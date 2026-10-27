"""GPU-resident dataset (Week 3, beyond A5): the whole split lives in GPU memory.

Why: after A5.3 the profile is loader-bound on Colab (2 CPU cores): AMP, channels_last and
compile give nothing alone. The decoded uint8 images of the full dataset take
~3,000 x 224 x 224 x 3 B = 0.45 GB, a few percent of a 16 GB T4. Copying them to the GPU ONCE
removes, for every epoch: the DataLoader workers, the collate, the pinned staging and the
host-to-device transfer (A5.10: PCIe bytes per epoch drop to zero).

Two modes (configs/exp_gpu_resident.yaml, configs/exp_precompute.yaml):
  - gpu_resident: uint8 images on the GPU; the batch is preprocessed on the GPU every step
    (same work and same order as the A5.3 path, only the loader changes);
  - gpu_resident + precompute (A5.9): the deterministic part (gamma, CLAHE, circular crop,
    normalize) is computed ONCE; each step only indexes the result (and augments on train).
    Stored in float16 when AMP is on (0.9 GB), float32 otherwise (1.8 GB).
    With augmentation the order changes: flip + rotation are applied AFTER the deterministic
    preprocessing. Flips and a <= 10 degree rotation about the centre commute with the centred
    circular crop; CLAHE tiles are computed on the un-rotated image, a small documented
    difference with the on-the-fly order.
Batches are drawn with torch.randperm on the GPU (seeded by set_seed), drop_last on train,
exactly like the DataLoader path.
"""
import os
import numpy as np
import torch

from src.data_loader import build_cache, DRDataset, get_splits
from src.paths import ROOT


class GpuResidentLoader:
    def __init__(self, x, y, batch_size, shuffle, drop_last, augment=False):
        self.x, self.y = x, y
        self.bs, self.shuffle, self.drop_last, self.augment = batch_size, shuffle, drop_last, augment
        n = len(y)
        self.n_batches = n // batch_size if drop_last else -(-n // batch_size)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        n = len(self.y)
        idx = (torch.randperm(n, device=self.y.device) if self.shuffle
               else torch.arange(n, device=self.y.device))
        for b in range(self.n_batches):
            i = idx[b * self.bs:(b + 1) * self.bs]
            x = self.x.index_select(0, i)                   # one gather kernel, no host copy
            if self.augment:
                from src.preprocess_gpu import augment_gpu
                x = augment_gpu(x.float()).to(self.x.dtype)
            yield x, self.y.index_select(0, i)


def _uint8_split(items, cfg, device):
    """(N,3,H,W) uint8 on `device`, from the A3 cache when data.cache_dir is set."""
    size = cfg["data"]["image_size"]
    cache_dir = cfg["data"].get("cache_dir")
    if cache_dir:
        cache_dir = cache_dir if os.path.isabs(cache_dir) else str(ROOT / cache_dir)
        u8 = np.asarray(build_cache(items, size, cache_dir))             # (N,H,W,3), read once
    else:
        ds = DRDataset(items, size)
        u8 = np.stack([ds.load(i) for i in range(len(items))])
    if not u8.flags.writeable:                      # read-only memmap: torch warns on from_numpy
        u8 = np.array(u8)
    t = torch.from_numpy(np.ascontiguousarray(u8)).to(device)          # one H2D copy per split
    return t.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def make_gpu_resident_loaders(cfg, device, which=("train", "val"), augment=False,
                              precompute=False, store_fp16=False, chunk=128):
    """Returns (loaders, splits). Loaders yield (x, y) already on `device`:
    x uint8 (preprocess with gpu_pre) or, with precompute, the preprocessed float tensor."""
    from src.preprocess_gpu import preprocess_batch_gpu
    splits = get_splits(cfg)
    bs = cfg["data"]["batch_size"]
    loaders = {}
    for k in which:
        items = splits[k]
        x = _uint8_split(items, cfg, device)
        y = torch.tensor([int(l) for _, l in items], device=device)
        aug_here = False
        if precompute:
            dtype = torch.float16 if store_fp16 else torch.float32
            x = torch.cat([preprocess_batch_gpu(x[i:i + chunk], cfg, augment=False, device=device)
                           .to(dtype) for i in range(0, len(x), chunk)])
            aug_here = augment and k == "train"
        loaders[k] = GpuResidentLoader(x, y, bs, shuffle=(k == "train"), drop_last=(k == "train"),
                                       augment=aug_here)
    return loaders, splits

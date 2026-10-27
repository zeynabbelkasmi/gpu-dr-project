"""Week 2, re-profiling step: A/B of the CLAHE histogram on real fundus images.

v1 used torch.bincount (implicit device->host sync to read max(ids)); v2 uses scatter_add_
(no sync, but global int64 atomics: the black background puts ~20% of each corner tile's
pixels in bin 0). Times the histogram alone and the whole clahe_gpu, B = 32 and 128.
Writes results/week2/ab_clahe_histogram.csv.

    python benchmarks/ab_clahe_histogram.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import yaml

from src import preprocess_gpu as pg
from src.data_loader import DRDataset, get_splits
from src.utils import GpuTimer, gpu_warmup


def hist_bincount(ids, nbins):
    return torch.bincount(ids, minlength=nbins)


def hist_scatter(ids, nbins):
    h = torch.zeros(nbins, dtype=torch.long, device=ids.device)
    return h.scatter_add_(0, ids, pg._ones(ids.numel(), ids.device))


def timeit(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        with GpuTimer("cuda") as t:
            fn()
        ts.append(t.elapsed)
    return 1e3 * np.mean(ts), 1e3 * np.std(ts)


def main():
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    items = get_splits(cfg)["train"][:128]
    ds = DRDataset(items, cfg["data"]["image_size"])
    u8 = torch.from_numpy(np.stack([ds.load(i) for i in range(len(items))])).permute(0, 3, 1, 2)
    x = pg.to_float_bchw(u8.contiguous(), "cuda")
    gpu_warmup("cuda", 3.0)
    rows = []
    for b in (32, 128):
        xb = x[:b]
        L = (pg._lab_luts(x.device)[0][((xb * 255).to(torch.uint8).long()
             * torch.tensor([65536, 256, 1], device="cuda").view(1, 3, 1, 1)).sum(1).view(-1)][:, 0]).view(b, 224, 224)
        g = pg._clahe_geometry(224, 224, 8, x.device)
        off = (torch.arange(b, device="cuda") * (64 * 256)).view(b, 1)
        ids = (off + g["tile_of"] * 256 + L.long().reshape(b, -1)).reshape(-1)
        nb = b * 64 * 256
        assert torch.equal(hist_bincount(ids, nb), hist_scatter(ids, nb))
        for name, fn in (("bincount (v1)", hist_bincount), ("scatter_add_ (v2)", hist_scatter)):
            m, s = timeit(lambda: fn(ids, nb))
            rows.append({"batch": b, "step": "histogram only", "impl": name, "mean_ms": m, "std_ms": s})
        m, s = timeit(lambda: pg.clahe_gpu(xb))
        rows.append({"batch": b, "step": "clahe_gpu total", "impl": "scatter_add_ (v2)", "mean_ms": m, "std_ms": s})
        zero_frac = (L == 0).float().mean().item()
        rows[-1]["frac_pixels_L0"] = zero_frac
    df = pd.DataFrame(rows)
    df.to_csv("results/week2/ab_clahe_histogram.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

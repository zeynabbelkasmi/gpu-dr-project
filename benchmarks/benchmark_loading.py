"""Week 2 (A3.5): redesign of Module A (loading). Every alternative is measured on the
same training images and compared with the reference (PIL decode + BILINEAR resize,
DRDataset.load) in time AND in pixel difference, because a faster decoder that
changes the input is a different experiment.

    python benchmarks/benchmark_loading.py [--n 256]

Alternatives (per image, single thread unless stated):
  pil_reference      Image.open().convert("RGB").resize(BILINEAR)      == DRDataset.load
  pil_threads        the same, ThreadPoolExecutor (PIL releases the GIL in decode/resize)
  pil_draft          JPEG DCT-domain downscale (Image.draft) before the resize
  cv2_reduced_4      cv2.imdecode(IMREAD_REDUCED_COLOR_4) + INTER_AREA resize
  nvjpeg_gpu         torchvision.io.decode_jpeg(device="cuda") batched + GPU antialiased resize
  uint8_cache        np.memmap of the decoded 224x224 images (built once, A3)
Plus the cache life cycle on the full training split: build time (first run), size on
disk, and one epoch of pure loading through a DataLoader with and without the cache.
Writes results/week2/loading.csv and results/week2/cache.json.
"""
import os, sys, io, json, time, argparse, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data_loader import DRDataset, build_cache, cache_path, get_splits
from src.paths import ROOT

OUT = "results/week2"


def pil_reference(p, size):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def pil_draft(p, size):
    with Image.open(p) as im:
        im.draft("RGB", size)                       # libjpeg decodes at 1/2, 1/4 or 1/8 scale
        return np.asarray(im.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def cv2_reduced(p, size, flag=cv2.IMREAD_REDUCED_COLOR_4):
    img = cv2.imdecode(np.fromfile(p, np.uint8), flag)
    return cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def nvjpeg_batch(paths, size, chunk=32):
    from torchvision.io import decode_jpeg
    out = []
    for i in range(0, len(paths), chunk):
        raw = [torch.frombuffer(bytearray(open(p, "rb").read()), dtype=torch.uint8)
               for p in paths[i:i + chunk]]
        for im in decode_jpeg(raw, device="cuda"):
            out.append(F.interpolate(im[None].float(), size=(size[1], size[0]), mode="bilinear",
                                     antialias=True, align_corners=False)
                       .round_().clamp_(0, 255).to(torch.uint8)[0])
    torch.cuda.synchronize()
    return out


def timed(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return r, float(np.mean(ts)), float(np.std(ts))


def epoch_loading(items, size, cache, nw):
    """One pass of pure loading (uint8, no preprocessing) through a DataLoader."""
    dl = DataLoader(DRDataset(items, size, raw_uint8=True, cache=cache), batch_size=32,
                    shuffle=True, num_workers=nw, drop_last=True, persistent_workers=False)
    t0 = time.perf_counter()
    n = sum(x.shape[0] for x, _ in dl)
    return time.perf_counter() - t0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    size = tuple(cfg["data"]["image_size"])
    os.makedirs(OUT, exist_ok=True)
    train = get_splits(cfg)["train"]
    paths = [p for p, _ in train[:args.n]]
    n = len(paths)
    workers = os.cpu_count() // 2

    ref = np.stack([pil_reference(p, size) for p in paths[:8]])            # warm-up + reference
    ref = np.stack([pil_reference(p, size) for p in paths])
    rows = []

    def add(name, fn, repeats=3):
        out, m, s = timed(fn, repeats)
        out = np.stack([o.permute(1, 2, 0).cpu().numpy() if torch.is_tensor(o) else o for o in out])
        d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
        rows.append({"variant": name, "n_images": n, "mean_s": m, "std_s": s,
                     "per_image_ms": 1e3 * m / n, "imgs_per_s": n / m,
                     "max_abs_levels_vs_ref": int(d.max()), "mean_abs_levels_vs_ref": float(d.mean()),
                     "frac_bit_exact": float((d == 0).mean())})
        print(rows[-1], flush=True)

    add("pil_reference (1 thread)", lambda: [pil_reference(p, size) for p in paths])
    with ThreadPoolExecutor(workers) as ex:
        add(f"pil_reference ({workers} threads)", lambda: list(ex.map(lambda p: pil_reference(p, size), paths)))
        add(f"pil_draft ({workers} threads)", lambda: list(ex.map(lambda p: pil_draft(p, size), paths)))
    add("pil_draft (1 thread)", lambda: [pil_draft(p, size) for p in paths])
    add("cv2_reduced_4 (1 thread)", lambda: [cv2_reduced(p, size) for p in paths])
    if torch.cuda.is_available():
        nvjpeg_batch(paths[:32], size)                                       # warm-up (nvJPEG handles)
        add("nvjpeg_gpu (batched, incl. file read)", lambda: nvjpeg_batch(paths, size))

    # ---- uint8 cache life cycle on the whole training split ----
    cache_dir = str(ROOT / "data" / "cache_bench_loading")
    shutil.rmtree(cache_dir, ignore_errors=True)
    t0 = time.perf_counter()
    cache = build_cache(train, size, cache_dir, workers=workers)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    cache = build_cache(train, size, cache_dir, workers=workers)            # second call: reuse
    reopen_s = time.perf_counter() - t0
    file_mb = os.path.getsize(cache_path(train, size, cache_dir)) / 1e6
    add("uint8_cache (memmap read)", lambda: [np.asarray(cache[i]) for i in range(n)], repeats=5)

    src_mb = sum(os.path.getsize(p) for p, _ in train) / 1e6
    epochs = {}
    for label, c, nw in (("decode, 0 workers", None, 0), ("decode, 2 workers (baseline)", None, 2),
                         ("cache, 0 workers", cache, 0), ("cache, 2 workers", cache, 2)):
        epoch_loading(train[:64], size, c, nw)                               # spawn / page-cache warm-up
        s, k = epoch_loading(train, size, c, nw)
        epochs[label] = {"epoch_s": s, "images": k, "imgs_per_s": k / s}
        print(label, epochs[label], flush=True)
    info = {"n_train_images": len(train), "image_size": size, "build_workers": workers,
            "build_time_s (first run)": build_s, "reopen_time_s (later runs)": reopen_s,
            "cache_file_mb": file_mb, "source_jpeg_mb": src_mb,
            "cache_vs_source_size": file_mb / src_mb,
            "full_dataset_cache_mb (2999 images)": 2999 * size[0] * size[1] * 3 / 1e6,
            "epoch_loading": epochs,
            "bit_exact_vs_decode": bool(all(np.array_equal(cache[i], pil_reference(train[i][0], size))
                                            for i in range(0, len(train), 97)))}
    del cache                                                                # Windows: unmap before delete
    import gc; gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/loading.csv", index=False)
    with open(f"{OUT}/cache.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

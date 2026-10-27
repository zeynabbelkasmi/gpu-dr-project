"""Week 1, Module A: split the loading stage into disk read, JPEG decode, resize and
array conversion, single-threaded, on the reference code path (PIL, as DRDataset.load).

    python benchmarks/profile_io.py [--n 200]

Two file sets:
  warm  - training images, read every epoch by the baseline, so they sit in the OS page cache
  first - test-split files, never opened by training or evaluation in this session; only
          their bytes are read and decoded for timing (no label, no model), so the test
          set stays unused for any decision. Windows cannot drop the page cache without
          admin tools, so "first" is the closest available proxy for a cold read.
Writes results/week1/loading_breakdown.json.
"""
import os, sys, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import psutil
from PIL import Image

from src.data_loader import shared_splits


def breakdown(paths, size=(224, 224)):
    t = {"read": 0.0, "decode": 0.0, "resize": 0.0, "to_array": 0.0}
    nbytes, npix = 0, 0
    io0 = psutil.Process().io_counters()
    for p in paths:
        t0 = time.perf_counter()
        with open(p, "rb") as f:
            buf = f.read()
        t1 = time.perf_counter()
        im = Image.open(io.BytesIO(buf))
        im = im.convert("RGB")                      # forces the full JPEG decode
        t2 = time.perf_counter()
        npix = max(npix, im.size[0] * im.size[1])
        im = im.resize(size, Image.Resampling.BILINEAR)
        t3 = time.perf_counter()
        a = np.asarray(im, dtype=np.uint8)
        t4 = time.perf_counter()
        t["read"] += t1 - t0; t["decode"] += t2 - t1; t["resize"] += t3 - t2; t["to_array"] += t4 - t3
        nbytes += len(buf)
    io1 = psutil.Process().io_counters()
    n = len(paths)
    total = sum(t.values())
    return {"n_images": n,
            "per_image_ms": {k: 1e3 * v / n for k, v in t.items()},
            "share": {k: v / total for k, v in t.items()},
            "total_per_image_ms": 1e3 * total / n,
            "mean_file_mb": nbytes / n / 1e6,
            "read_mb_per_s": nbytes / 1e6 / max(t["read"], 1e-9),
            "decode_mpix_per_s (max frame size)": n * npix / 1e6 / t["decode"],
            "decoded_frame_mb (uint8 RGB, full res)": npix * 3 / 1e6,
            "output_mb (224x224x3 uint8)": size[0] * size[1] * 3 / 1e6,
            "process_read_bytes": io1.read_bytes - io0.read_bytes,
            "imgs_per_s_single_thread": n / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    s = shared_splits()
    out = {"warm (train files)": breakdown([p for p, _ in s["train"][:args.n]]),
           "first touch (test files, bytes only)": breakdown([p for p, _ in s["test"][:args.n]])}
    os.makedirs("results/week1", exist_ok=True)
    with open("results/week1/loading_breakdown.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

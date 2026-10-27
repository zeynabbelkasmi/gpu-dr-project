"""Workload for NVIDIA Nsight Systems: the batched GPU preprocessing + filters, with NVTX
ranges so every stage is a labelled box on the timeline.

    nsys profile -t cuda,nvtx --stats=true -o results/week2/nsys_preprocess --force-overwrite true \
        .venv/Scripts/python benchmarks/nsys_preprocess.py
    (Windows: nsys is in "C:/Program Files/NVIDIA Corporation/Nsight Systems <ver>/target-windows-x64")

Input: 128 decoded training images, uploaded in pinned uint8 batches of --batch.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml

from src import filters_gpu as fg, preprocess_gpu as pg
from src.data_loader import DRDataset, get_splits

nvtx = torch.cuda.nvtx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    items = get_splits(cfg)["train"][:128]
    ds = DRDataset(items, cfg["data"]["image_size"])
    u8 = torch.from_numpy(np.stack([ds.load(i) for i in range(len(items))])).permute(0, 3, 1, 2)
    batches = [u8[i:i + args.batch].contiguous().pin_memory() for i in range(0, len(u8), args.batch)]

    for b in batches:                                     # warm-up: context, LUTs, kernels
        pg.preprocess_batch_gpu(b, cfg)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for it in range(args.iters):
        b = batches[it % len(batches)]
        nvtx.range_push(f"batch{it}")
        nvtx.range_push("h2d+to_float")
        x = pg.to_float_bchw(b, "cuda")
        nvtx.range_pop()
        nvtx.range_push("clahe")
        x = pg.clahe_gpu(x)
        nvtx.range_pop()
        nvtx.range_push("crop+normalize")
        y = pg.crop_normalize_fused(x)
        nvtx.range_pop()
        nvtx.range_push("filters")
        fg.gaussian_blur_gpu(x); fg.median_filter_gpu(x); fg.contrast_stretch_gpu(x)
        nvtx.range_pop()
        nvtx.range_push("augment")
        pg.augment_gpu(y)
        nvtx.range_pop()
        nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()

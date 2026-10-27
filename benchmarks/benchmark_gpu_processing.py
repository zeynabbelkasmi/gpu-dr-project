"""Week 2 (A4.11, A4.13-A4.15): CPU vs GPU for every image-processing stage, on real
fundus images, with correctness, transfer, memory and kernel-launch evidence.

    python benchmarks/benchmark_gpu_processing.py [--n 128] [--repeats 10]

Protocol (brief, section 7): same images for CPU and GPU, warm-up runs discarded
(CUDA context, cuDNN/cuBLAS handles, CuPy JIT, LUT construction), GpuTimer synchronizes
around every timed region, mean +/- std over repeats. CPU reference = the per-image
functions of preprocess_cpu / filters_cpu in one process (OpenCV with its default
thread pool); a thread-pool CPU row gives the best-effort multi-core CPU number.

Outputs (results/week2/):
  sweep.csv            time per batch and per image, batch sizes 1, 8, 32, 64, 128
  equivalence.csv      max/mean abs difference vs the CPU reference, fraction bit-exact
  h2d.csv              host-to-device copy: uint8 vs float32, pinned vs pageable
  gpu_resources.csv    peak GPU memory, allocator calls and CUDA kernels per call (B=128)
  torch_profiler_gpu_preprocess.txt   top CUDA kernels of the full GPU preprocessing
  summary.json         environment + headline numbers used by the report
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src import filters_cpu as fc, filters_gpu as fg, preprocess_cpu as pc, preprocess_gpu as pg
from src.data_loader import DRDataset, get_splits
from src.utils import GpuTimer, gpu_state, gpu_warmup, set_seed

OUT = "results/week2"
GPU = torch.device("cuda")
CPU = torch.device("cpu")
BATCHES = (1, 8, 32, 64, 128)


def bench(fn, device, repeats, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        with GpuTimer(device) as t:
            fn()
        ts.append(t.elapsed)
    return float(np.mean(ts)), float(np.std(ts))


def cpu_augment_fixed(img, flip, angle):
    """preprocess_cpu.augment_image with the random draws fixed (for equivalence)."""
    if flip:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--cpu-repeats", type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    os.makedirs(OUT, exist_ok=True)
    has_cupy = fg.cupy_available()

    items = get_splits(cfg)["train"][:args.n]
    ds = DRDataset(items, cfg["data"]["image_size"])
    u8 = np.stack([ds.load(i) for i in range(len(items))])               # N,H,W,3 uint8
    f32 = u8.astype(np.float32) / 255.0
    u8_nchw = torch.from_numpy(np.ascontiguousarray(u8.transpose(0, 3, 1, 2)))
    pinned_all = u8_nchw.pin_memory()
    t_all = pg.to_float_bchw(pinned_all, GPU)                             # N,3,H,W float on GPU
    state_idle = gpu_state()
    state_warm = gpu_warmup(GPU, 5.0)
    print("GPU state idle -> after warm-up:", state_idle, state_warm, flush=True)
    rng = np.random.default_rng(0)

    # op -> (CPU reference per image or None, GPU batch fn, CPU reference op name)
    ops = {
        "clahe": (pc.apply_clahe, lambda t: pg.clahe_gpu(t), "clahe"),
        "clahe_luma (GPU variant)": (None, lambda t: pg.clahe_gpu(t, mode="luma"), "clahe"),
        "circular_crop": (pc.circular_crop, lambda t: pg.circular_crop_gpu(t), "circular_crop"),
        "normalize": (pc.normalize, lambda t: pg.normalize_gpu(t), "normalize"),
        "crop+normalize": (lambda i: pc.normalize(pc.circular_crop(i)),
                           lambda t: pg.normalize_gpu(pg.circular_crop_gpu(t)), "crop+normalize"),
        "crop+normalize fused (GPU variant)": (None, pg.crop_normalize_fused, "crop+normalize"),
        "preprocess_full": (lambda i: pc.preprocess_image(i, cfg),
                            lambda t: pg.preprocess_batch_gpu(t, cfg, fuse=False), "preprocess_full"),
        "preprocess_full fused (GPU variant)": (None, lambda t: pg.preprocess_batch_gpu(t, cfg),
                                                "preprocess_full"),
        "gaussian_blur": (fc.gaussian_blur, lambda t: fg.gaussian_blur_gpu(t), "gaussian_blur"),
        "gaussian_blur 2D kernel (GPU variant)": (None, lambda t: fg.gaussian_blur_gpu(t, separable=False),
                                                  "gaussian_blur"),
        "median_filter": (fc.median_filter, fg.median_filter_gpu, "median_filter"),
        "green_channel": (lambda i: np.ascontiguousarray(fc.green_channel(i)),
                          lambda t: fg.green_channel_gpu(t).contiguous(), "green_channel"),
        "contrast_stretch": (fc.contrast_stretch, fg.contrast_stretch_gpu, "contrast_stretch"),
        "augmentation": (lambda i: pc.augment_image(i, rng), lambda t: pg.augment_gpu(t), "augmentation"),
    }
    if has_cupy:
        ops["median_filter CuPy RawKernel (GPU variant)"] = (None, fg.median_filter_cupy_kernel,
                                                             "median_filter")
        ops["median_filter cupyx.ndimage (GPU variant)"] = (None, fg.median_filter_cupy_ndimage,
                                                            "median_filter")

    # ---- 1. batch-size sweep (A4.13) ----
    rows = []
    for name, (cpu_fn, gpu_fn, ref) in ops.items():
        for b in BATCHES:
            imgs, t = f32[:b], t_all[:b]
            if cpu_fn is not None:
                m, s = bench(lambda: [cpu_fn(i) for i in imgs], CPU, args.cpu_repeats, 1)
                rows.append({"op": name, "ref_op": ref, "device": "cpu", "batch": b,
                             "mean_s": m, "std_s": s, "per_image_us": 1e6 * m / b})
            m, s = bench(lambda: gpu_fn(t), GPU, args.repeats, 3)
            rows.append({"op": name, "ref_op": ref, "device": "cuda", "batch": b,
                         "mean_s": m, "std_s": s, "per_image_us": 1e6 * m / b})
        print(f"[sweep] {name}", flush=True)

    # full preprocessing INCLUDING the host-to-device copy of the uint8 batch (pinned)
    for b in BATCHES:
        src = pinned_all[:b]
        for fuse in (False, True):
            m, s = bench(lambda: pg.preprocess_batch_gpu(src, cfg, fuse=fuse), GPU, args.repeats, 3)
            rows.append({"op": "preprocess_full + H2D" + (" fused (GPU variant)" if fuse else ""),
                         "ref_op": "preprocess_full", "device": "cuda", "batch": b,
                         "mean_s": m, "std_s": s, "per_image_us": 1e6 * m / b})
    # best-effort multi-core CPU: thread pool over images (cv2 releases the GIL)
    workers = os.cpu_count() // 2
    with ThreadPoolExecutor(workers) as ex:
        for b in BATCHES:
            imgs = f32[:b]
            m, s = bench(lambda: list(ex.map(lambda i: pc.preprocess_image(i, cfg), imgs)),
                         CPU, args.cpu_repeats, 1)
            rows.append({"op": f"preprocess_full CPU {workers} threads", "ref_op": "preprocess_full",
                         "device": "cpu-mt", "batch": b, "mean_s": m, "std_s": s,
                         "per_image_us": 1e6 * m / b})
    sweep = pd.DataFrame(rows)
    ref = sweep[(sweep.device == "cpu") & (sweep.op == sweep.ref_op)].set_index(["ref_op", "batch"])["mean_s"]
    sweep["speedup_vs_cpu"] = [ref.get((r.ref_op, r.batch), np.nan) / r.mean_s if r.device != "cpu" else np.nan
                               for r in sweep.itertuples()]
    sweep.to_csv(f"{OUT}/sweep.csv", index=False)

    # ---- 2. numerical equivalence on all N real images ----
    eq = []
    flips = rng.random(len(items)) < 0.5
    angles = rng.uniform(-10, 10, len(items))
    for name, (cpu_fn, gpu_fn, refname) in ops.items():
        if name == "augmentation":
            cpu_out = np.stack([cpu_augment_fixed(i, fl, a) for i, fl, a in zip(f32, flips, angles)])
            gpu_out = pg.augment_gpu(t_all, flip=torch.from_numpy(flips).to(GPU),
                                     angle=torch.from_numpy(angles).float().to(GPU))
        else:
            ref_fn = cpu_fn or ops[refname][0]
            cpu_out = np.stack([ref_fn(i) for i in f32])
            gpu_out = gpu_fn(t_all)
        g = gpu_out.cpu().numpy()                     # GPU: NCHW
        if refname == "green_channel":
            g = g[:, 0]                               # CPU: (N,H,W)
        elif refname != "preprocess_full":
            g = g.transpose(0, 2, 3, 1)               # CPU: NHWC (preprocess_image returns CHW)
        d = np.abs(cpu_out.astype(np.float64) - g.astype(np.float64))
        rec = {"op": name, "n_images": len(items), "max_abs": d.max(), "mean_abs": d.mean(),
               "frac_bit_exact": float((cpu_out == g).mean()),
               "frac_within_1_level (1/255)": float((d <= 1 / 255 + 1e-7).mean())}
        if refname == "clahe":
            rec["luminance_corr"] = float(np.corrcoef(cpu_out.mean(-1).ravel(), g.mean(-1).ravel())[0, 1])
            rec["max_abs_levels"] = float(np.abs(np.round(cpu_out * 255) - np.round(g * 255)).max())
        if name == "augmentation":
            inner = (slice(None), slice(8, -8), slice(8, -8))
            rec["max_abs_interior"] = d[inner].max()
        eq.append(rec)
    pd.DataFrame(eq).to_csv(f"{OUT}/equivalence.csv", index=False)
    print(pd.DataFrame(eq).to_string(index=False), flush=True)

    # ---- 3. host-to-device copies (A4.14) ----
    h2d = []
    for b in (32, 128):
        for dtype in ("uint8", "float32"):
            host = u8_nchw[:b] if dtype == "uint8" else torch.from_numpy(
                np.ascontiguousarray(f32[:b].transpose(0, 3, 1, 2)))
            for mem in ("pageable", "pinned"):
                h = host.pin_memory() if mem == "pinned" else host.clone()
                gpu_warmup(GPU, 1.0)                                         # keep the link at full speed
                m, s = bench(lambda: h.to(GPU, non_blocking=(mem == "pinned")), GPU, args.repeats, 3)
                nb = h.numel() * h.element_size()
                h2d.append({"batch": b, "dtype": dtype, "memory": mem, "bytes": nb,
                            "mean_ms": 1e3 * m, "std_ms": 1e3 * s, "gb_per_s": nb / m / 1e9})
    h2d = pd.DataFrame(h2d)
    h2d.to_csv(f"{OUT}/h2d.csv", index=False)
    print(h2d.to_string(index=False), flush=True)

    # ---- 4. memory, allocator activity and kernel launches per call (B=128) ----
    from torch.profiler import profile, ProfilerActivity
    res = []
    t = t_all
    for name, (_, gpu_fn, _) in ops.items():
        gpu_fn(t); torch.cuda.synchronize()                          # constants built
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        a0 = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = gpu_fn(t); torch.cuda.synchronize()
        a1 = torch.cuda.memory_stats()["allocation.all.allocated"]
        peak = torch.cuda.max_memory_allocated() - base
        del out
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            gpu_fn(t); torch.cuda.synchronize()
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                   and "memcpy" not in e.name.lower() and "memset" not in e.name.lower()]
        res.append({"op": name, "batch": t.shape[0], "peak_extra_mib": peak / 2**20,
                    "input_mib": t.numel() * 4 / 2**20, "allocations_per_call": a1 - a0,
                    "cuda_kernels_per_call": len(kernels)})
    def mib(v):
        if torch.is_tensor(v):
            return v.numel() * v.element_size() / 2**20
        if isinstance(v, dict):
            return sum(mib(x) for x in v.values())
        if isinstance(v, (tuple, list)):
            return sum(mib(x) for x in v)
        return 0.0
    const_mib = sum(mib(v) for v in pg._CONST.values())
    lut_mib = sum(mib(v) for k, v in pg._CONST.items() if k[0] == "lab_luts")
    pd.DataFrame(res).to_csv(f"{OUT}/gpu_resources.csv", index=False)
    print(pd.DataFrame(res).to_string(index=False), flush=True)

    # ---- 5. where does the GPU preprocessing time go? (new bottleneck) ----
    pg.preprocess_batch_gpu(pinned_all, cfg); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            pg.preprocess_batch_gpu(pinned_all, cfg)
        torch.cuda.synchronize()
    with open(f"{OUT}/torch_profiler_gpu_preprocess.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    summary = {"gpu_state_idle": state_idle, "gpu_state_after_warmup": state_warm,
               "gpu_state_end": gpu_state(), "n_images": len(items), "image_size": cfg["data"]["image_size"],
               "cv2_threads": cv2.getNumThreads(), "torch_cpu_threads": torch.get_num_threads(),
               "cpu_mt_workers": workers, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "cuda": torch.version.cuda, "opencv": cv2.__version__,
               "cupy": __import__("cupy").__version__ if has_cupy else None,
               "device_constants_mib": {"preprocess_gpu total": const_mib, "of which LAB LUTs": lut_mib}}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

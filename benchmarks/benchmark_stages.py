"""Stage-level benchmark: CPU vs GPU per pipeline stage on one real batch of
training images. Writes results/timing.csv (assignments A2 and A5).

Every stage processes the same N images, so per-stage times are comparable.
The CPU rows are provided. The GPU rows are yours.

Methodology (brief, section 7): warm-up runs, CUDA synchronized around every
timed region (GpuTimer), several repeats, mean +/- std.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
import torch

from src.utils import GpuTimer, gpu_warmup, set_seed
from src.data_loader import get_splits, DRDataset
from src import preprocess_cpu, filters_cpu
from src.model import build_model


def bench(fn, device, repeats, warmup=2):
    """Run fn `warmup` times (discarded), then `repeats` timed runs. Returns (mean, std) in s."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        with GpuTimer(device) as t:
            fn()
        ts.append(t.elapsed)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=64, help="images per stage")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/timing.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu")
    N, R = args.n, args.repeats
    rows = []

    def add(stage, dev, fn, device, repeats=R, warmup=2, n=N):
        m, s = bench(fn, device, repeats, warmup)
        rows.append({"stage": stage, "device": dev, "mean_s": m, "std_s": s,
                     "n_images": n, "imgs_per_s": n / m})
        print(f"{stage:20s} {dev:4s} {m * 1e3:10.2f} ms +/- {s * 1e3:.2f}", flush=True)

    train = get_splits(cfg)["train"][:N]
    ds = DRDataset(train, cfg["data"]["image_size"])

    # ---- CPU rows (given) ----
    add("loading", "cpu", lambda: [ds.load(i) for i in range(N)], cpu, repeats=3, warmup=1)
    u8 = np.stack([ds.load(i) for i in range(N)])            # N,H,W,3 uint8
    f32 = u8.astype(np.float32) / 255.0
    add("preprocessing", "cpu", lambda: preprocess_cpu.batch_preprocess_cpu(f32, cfg), cpu)
    add("filter_gaussian", "cpu", lambda: [filters_cpu.gaussian_blur(i) for i in f32], cpu)
    add("filter_median", "cpu", lambda: [filters_cpu.median_filter(i) for i in f32], cpu)
    add("filter_green", "cpu",
        lambda: [np.ascontiguousarray(filters_cpu.green_channel(i)) for i in f32], cpu)
    add("filter_contrast", "cpu", lambda: [filters_cpu.contrast_stretch(i) for i in f32], cpu)
    rng = np.random.default_rng(0)
    add("augmentation", "cpu", lambda: [preprocess_cpu.augment_image(i, rng) for i in f32], cpu)

    x = torch.from_numpy(preprocess_cpu.batch_preprocess_cpu(f32, cfg))
    y = torch.tensor([l for _, l in train])
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()

    def train_step():
        model.train()
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    def infer():
        model.eval()
        with torch.no_grad():
            model(x)

    add("train_step", "cpu", train_step, cpu, repeats=3, warmup=1)
    add("inference", "cpu", infer, cpu, repeats=3, warmup=1)

    # ---- GPU rows (A5.6) ----
    if has_cuda:
        from src import preprocess_gpu, filters_gpu
        from torchvision.io import decode_jpeg
        import torch.nn.functional as F
        gpu = torch.device("cuda")
        print("GPU state after warm-up:", gpu_warmup(gpu, 5.0), flush=True)
        h, w = cfg["data"]["image_size"][1], cfg["data"]["image_size"][0]

        # loading, GPU variant: file bytes read on the CPU, batched nvJPEG decode + antialiased
        # bilinear resize on the GPU (not bit-exact with PIL, see the Week 2 report).
        # The uint8 cache of A3 is reported separately as loading_cache (CPU, memmap read).
        def gpu_load():
            raw = [torch.frombuffer(bytearray(open(p, "rb").read()), dtype=torch.uint8)
                   for p, _ in train]
            imgs = decode_jpeg(raw, device=gpu)
            return [F.interpolate(im[None].float(), size=(h, w), mode="bilinear",
                                  antialias=True, align_corners=False) for im in imgs]
        add("loading", "cuda", gpu_load, gpu, repeats=3, warmup=1)
        cache_dir = os.path.join("data", "cache_bench")
        from src.data_loader import build_cache
        cache = build_cache(train, cfg["data"]["image_size"], cache_dir)
        add("loading_cache", "cpu", lambda: np.array(cache[:N]), cpu)   # np.array forces the read (a slice is a view)

        pinned = torch.from_numpy(np.ascontiguousarray(u8.transpose(0, 3, 1, 2))).pin_memory()
        add("h2d_transfer", "cuda", lambda: pinned.to(gpu, non_blocking=True), gpu)
        add("preprocessing", "cuda", lambda: preprocess_gpu.preprocess_batch_gpu(pinned, cfg), gpu)
        t = preprocess_gpu.to_float_bchw(pinned, gpu)
        add("filter_gaussian", "cuda", lambda: filters_gpu.gaussian_blur_gpu(t), gpu)
        add("filter_median", "cuda", lambda: filters_gpu.median_filter_gpu(t), gpu)
        add("filter_green", "cuda", lambda: filters_gpu.green_channel_gpu(t).contiguous(), gpu)
        add("filter_contrast", "cuda", lambda: filters_gpu.contrast_stretch_gpu(t), gpu)
        add("augmentation", "cuda", lambda: preprocess_gpu.augment_gpu(t), gpu)

        # same model, same batch: isolates the device effect; then AMP as a labelled variant
        xg, yg = x.to(gpu), y.to(gpu)
        model_g = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(gpu)
        opt_g = torch.optim.AdamW(model_g.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        def train_step_gpu(amp=False):
            model_g.train()
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = crit(model_g(xg), yg)
            scaler.scale(loss).backward() if amp else loss.backward()
            scaler.step(opt_g) if amp else opt_g.step()
            if amp:
                scaler.update()

        def infer_gpu(amp=False):
            model_g.eval()
            with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                model_g(xg)

        add("train_step", "cuda", train_step_gpu, gpu, warmup=3)
        add("train_step_amp", "cuda", lambda: train_step_gpu(True), gpu, warmup=3)
        add("inference", "cuda", infer_gpu, gpu, warmup=3)
        add("inference_amp", "cuda", lambda: infer_gpu(True), gpu, warmup=3)

    df = pd.DataFrame(rows)
    # A5.7: speedup of every GPU row against the CPU row of the same stage (AMP variants are
    # compared with the fp32 CPU row of their stage: train_step_amp vs train_step).
    cpu_mean = df[df.device == "cpu"].set_index("stage")["mean_s"]
    base = df.stage.str.replace("_amp", "", regex=False)
    df["speedup_vs_cpu"] = np.where(df.device == "cuda", base.map(cpu_mean) / df.mean_s, np.nan)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

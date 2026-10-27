"""A5.8 / A6.7 / A5.10: input-pipeline sweep on the real training split.

For each (batch_size, num_workers) the train loader is iterated once for warm-up and then
`--repeats` times; every batch goes to the GPU and, with --gpu-preprocess, through the GPU
preprocessing, so the number is the throughput the training loop actually sees.
Writes results/sweep_loader.csv and results/sweep_loader.png, and prints the bytes that
cross PCIe per epoch with CPU preprocessing (float32) and GPU preprocessing (uint8).

    python benchmarks/sweep_loader.py --config configs/optimized.yaml
    python benchmarks/sweep_loader.py --config configs/exp_cache.yaml --cpu-preprocess
"""
import os, sys, argparse, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
import torch

from src.data_loader import make_loaders, get_splits
from src.train import resolve_gpu_preprocess, make_gpu_pre
from src.utils import set_seed


def epoch_throughput(loader, device, pre):
    n = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, _ in loader:
        x = pre(x) if pre is not None else x.to(device, non_blocking=True)
        n += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimized.yaml")
    ap.add_argument("--workers", default="0,1,2,4,8")
    ap.add_argument("--batch-sizes", default="32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu-preprocess", dest="gpu_preprocess", action="store_true", default=None)
    ap.add_argument("--cpu-preprocess", dest="gpu_preprocess", action="store_false")
    ap.add_argument("--out", default="results/sweep_loader.csv")
    args = ap.parse_args()
    cfg0 = yaml.safe_load(open(args.config))
    set_seed(cfg0["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_pre_on = resolve_gpu_preprocess(cfg0, device, args.gpu_preprocess)
    cfg0["preprocess"]["augment"] = False           # same work in every configuration
    rows = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        for nw in [int(w) for w in args.workers.split(",")]:
            cfg = copy.deepcopy(cfg0)
            cfg["data"]["batch_size"], cfg["data"]["num_workers"] = bs, nw
            loader = make_loaders(cfg, device.type, gpu_preprocess=gpu_pre_on, which=("train",))[0]["train"]
            pre = make_gpu_pre(cfg, device, augment=False) if gpu_pre_on else None
            epoch_throughput(loader, device, pre)    # warm-up: worker start, cache, cuDNN
            ips = [epoch_throughput(loader, device, pre)[0] for _ in range(args.repeats)]
            rows.append({"batch_size": bs, "num_workers": nw, "gpu_preprocess": gpu_pre_on,
                         "imgs_per_s_mean": float(np.mean(ips)), "imgs_per_s_std": float(np.std(ips))})
            print(f"bs={bs:4d} workers={nw}: {np.mean(ips):8.1f} +/- {np.std(ips):.1f} img/s", flush=True)
            del loader
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for bs, g in df.groupby("batch_size"):
        ax.errorbar(g.num_workers, g.imgs_per_s_mean, yerr=g.imgs_per_s_std, marker="o",
                    capsize=3, label=f"batch {bs}")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("images / s (loader + H2D" + (" + GPU preprocess)" if gpu_pre_on else ")"))
    ax.set_title(f"A5.8 loader sweep ({device.type})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out.replace(".csv", ".png"), dpi=150)

    # A5.10: bytes over PCIe per training epoch
    w, h = cfg0["data"]["image_size"]
    n_img = len(get_splits(cfg0)["train"])
    u8, f32 = n_img * h * w * 3, n_img * h * w * 3 * 4
    print(f"A5.10: {n_img} train images/epoch -> uint8 (GPU preprocessing) {u8 / 2**20:.0f} MiB, "
          f"float32 (CPU preprocessing) {f32 / 2**20:.0f} MiB per epoch (4x less with uint8)")


if __name__ == "__main__":
    main()

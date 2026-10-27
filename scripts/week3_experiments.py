"""Week 3 (A6.9, A6.10): run the single-change experiments and build the comparison table.

Order follows A6.10: same model on CPU then GPU (device effect), then one optimization at a
time on GPU (pipeline and precision effects), then the model change, then everything together.
Runs whose results/<run>/history.json already exists are skipped (use --force to redo).

    python scripts/week3_experiments.py                 # all runs, 5 epochs each
    python scripts/week3_experiments.py --epochs 2      # quick check
    python scripts/week3_experiments.py --only exp_amp_gpu,exp_resnet18_gpu
Output: results/week3_table.csv (one row per run).
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch

from src.train import main as train_main

# (run name, config, device)
RUNS = [
    ("exp_cache_cpu", "configs/exp_cache.yaml", "cpu"),          # Week 2 reference (CPU)
    ("exp_cache_gpu", "configs/exp_cache.yaml", "cuda"),         # device effect only
    ("exp_gpu_pre_gpu", "configs/exp_gpu_pre.yaml", "cuda"),     # A5.3
    ("exp_prefetch_gpu", "configs/exp_prefetch.yaml", "cuda"),   # A5.5 (on top of A5.3)
    ("exp_amp_gpu", "configs/exp_amp.yaml", "cuda"),             # A6.1
    ("exp_channels_last_gpu", "configs/exp_channels_last.yaml", "cuda"),  # A6.2
    ("exp_compile_gpu", "configs/exp_compile.yaml", "cuda"),     # A6.3 (optional)
    ("exp_resnet18_gpu", "configs/exp_resnet18.yaml", "cuda"),   # A6.6 model effect
    ("optimized", "configs/optimized.yaml", "cuda"),             # everything on
]


def row(run):
    h = json.load(open(os.path.join("results", run, "history.json")))
    last = h["history"][-1]
    best = max(h["history"], key=lambda e: e["val_acc"])
    cfg = h["cfg"]
    return {"run": run, "model": cfg["train"]["model"], "device": h["device"],
            "precision": "amp" if h.get("amp") else "fp32",
            "preprocessing": "gpu" if h.get("gpu_preprocess") else "cpu",
            "prefetch_gpu": h.get("prefetch_gpu", False),
            "channels_last": h.get("channels_last", False), "compile": h.get("compile", False),
            "batch_size": cfg["data"]["batch_size"], "num_workers": cfg["data"]["num_workers"],
            "cold_epoch_s": h["cold_epoch_s"], "steady_epoch_s": h["steady_epoch_s"],
            "train_imgs_per_s": last["train_imgs_per_s"],
            "gpu_peak_mem_mb": last["gpu_peak_mem_mb"],
            "best_val_acc": h["best_val_acc"], "val_f1_at_best": best["val_f1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="default: epochs of each config")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu-epochs", type=int, default=2,
                    help="epochs of the CPU run: 2 is enough for a steady-state epoch time "
                         "(its accuracy is not comparable; the 5-epoch CPU reference is results/exp_cache)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    has_cuda = torch.cuda.is_available()
    done = []
    for run, cfg, dev in RUNS:
        if only and run not in only:
            continue
        if dev == "cuda" and not has_cuda:
            print(f"[skip] {run}: CUDA not available", flush=True)
            continue
        if run == "exp_compile_gpu" and sys.platform == "win32":
            print(f"[skip] {run}: torch.compile needs Triton (Linux)", flush=True)
            continue
        if args.force or not os.path.exists(os.path.join("results", run, "history.json")):
            ep = args.cpu_epochs if dev == "cpu" else args.epochs
            train_main(cfg, dev, run=run, epochs=ep, monitor=(run == "optimized"))
        done.append(run)
    df = pd.DataFrame([row(r) for r in done])
    ref = df.loc[df.run == "exp_cache_cpu", "steady_epoch_s"]
    if len(ref):
        df["speedup_vs_cpu"] = ref.iloc[0] / df.steady_epoch_s
    df.to_csv("results/week3_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

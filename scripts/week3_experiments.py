"""Week 3 (A6.9, A6.10): run the single-change experiments and build the comparison table.

Order follows A6.10: same model on CPU then GPU (device effect), then one optimization at a
time on GPU (pipeline and precision effects), then the model change, then the training recipe.
Runs whose results/<run>/history.json already exists are skipped (use --force to redo).

    python scripts/week3_experiments.py                      # all runs, seed 42
    python scripts/week3_experiments.py --only optimized --force
    python scripts/week3_experiments.py --only exp_r18_recipe_gpu,optimized --seeds 42,43,44
Seed 42 writes results/<run>/, another seed s writes results/<run>_s<s>/ (same split).
Outputs (always built from EVERY run found in results/, whatever --only says):
    results/week3_table.csv          one row per run and seed
    results/week3_table_seeds.csv    mean +/- std over seeds, one row per run
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
    ("exp_gpu_resident_gpu", "configs/exp_gpu_resident.yaml", "cuda"),  # data stays on the GPU
    ("exp_precompute_gpu", "configs/exp_precompute.yaml", "cuda"),      # A5.9 (on top of it)
    ("optimized_cnn", "configs/optimized_cnn.yaml", "cuda"),     # everything on, reference model
    ("optimized", "configs/optimized.yaml", "cuda"),             # fast pipeline + ResNet-18
    # A6.9 (3) training strategy, ResNet-18, one change at a time on top of optimized.yaml
    ("exp_r18_lr3e4_gpu", "configs/exp_r18_lr3e4.yaml", "cuda"),
    ("exp_r18_augment_gpu", "configs/exp_r18_augment.yaml", "cuda"),
    ("exp_r18_cosine_gpu", "configs/exp_r18_cosine.yaml", "cuda"),
    ("exp_r18_regul_gpu", "configs/exp_r18_regul.yaml", "cuda"),
    ("exp_r18_recipe_gpu", "configs/exp_r18_recipe.yaml", "cuda"),      # all four together
    ("exp_r18_recipe10_gpu", "configs/exp_r18_recipe10.yaml", "cuda"),  # same, 10 epochs
    ("exp_compile_ro_gpu", "configs/exp_compile_ro.yaml", "cuda"),  # + CUDA graphs (optional, last)
]
ORDER = [r for r, _, _ in RUNS]


def run_dir(run, seed):
    return run if seed == 42 else f"{run}_s{seed}"


def row(base, seed, d):
    h = json.load(open(os.path.join("results", d, "history.json")))
    last = h["history"][-1]
    best = max(h["history"], key=lambda e: e["val_acc"])       # the epoch that was checkpointed
    cfg = h["cfg"]
    t = cfg["train"]
    return {"run": base, "seed": seed, "model": t["model"], "device": h["device"],
            "precision": "amp" if h.get("amp") else "fp32",
            "preprocessing": "gpu" if h.get("gpu_preprocess") else "cpu",
            "prefetch_gpu": h.get("prefetch_gpu", False),
            "gpu_resident": h.get("gpu_resident", False), "precompute": h.get("precompute", False),
            "channels_last": h.get("channels_last", False), "compile": h.get("compile", False),
            "augment": h.get("augment", False), "lr": t["lr"],
            "scheduler": t.get("scheduler") or "none", "epochs": t["epochs"],
            "batch_size": cfg["data"]["batch_size"],
            "setup_time_s": h["setup_time_s"], "cold_epoch_s": h["cold_epoch_s"],
            "steady_epoch_s": h["steady_epoch_s"], "train_imgs_per_s": last["train_imgs_per_s"],
            "gpu_peak_mem_mb": last["gpu_peak_mem_mb"],
            "best_val_acc": h["best_val_acc"], "val_f1_at_best": best["val_f1"],
            "best_epoch": best["epoch"]}


def collect():
    rows = []
    for d in sorted(os.listdir("results")):
        if not os.path.exists(os.path.join("results", d, "history.json")):
            continue
        base, seed = d, 42
        if "_s" in d and d.rsplit("_s", 1)[1].isdigit():
            base, seed = d.rsplit("_s", 1)[0], int(d.rsplit("_s", 1)[1])
        if base in ORDER:
            rows.append(row(base, seed, d))
    df = pd.DataFrame(rows)
    df["order"] = df.run.map(ORDER.index)
    df = df.sort_values(["order", "seed"]).drop(columns="order")
    ref = df.loc[(df.run == "exp_cache_cpu") & (df.seed == 42), "steady_epoch_s"]
    if len(ref):
        df["speedup_vs_cpu"] = ref.iloc[0] / df.steady_epoch_s
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="default: epochs of each config")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,43,44")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu-epochs", type=int, default=2,
                    help="epochs of the CPU run: 2 is enough for a steady-state epoch time "
                         "(its accuracy is not comparable; the 5-epoch CPU reference is results/exp_cache)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    seeds = [int(s) for s in args.seeds.split(",")]
    has_cuda = torch.cuda.is_available()
    for run, cfg, dev in RUNS:
        if only and run not in only:
            continue
        if dev == "cuda" and not has_cuda:
            print(f"[skip] {run}: CUDA not available", flush=True)
            continue
        if run.startswith("exp_compile") and sys.platform == "win32":
            print(f"[skip] {run}: torch.compile needs Triton (Linux)", flush=True)
            continue
        for seed in seeds:
            d = run_dir(run, seed)
            if not args.force and os.path.exists(os.path.join("results", d, "history.json")):
                continue
            ep = args.cpu_epochs if dev == "cpu" else args.epochs
            try:
                train_main(cfg, dev, run=d, epochs=ep, seed=seed, monitor=(d == "optimized"))
            except Exception as e:                  # one failing variant must not stop the others
                print(f"[fail] {d}: {type(e).__name__}: {e}", flush=True)

    df = collect()
    df.to_csv("results/week3_table.csv", index=False)
    cols = ["steady_epoch_s", "train_imgs_per_s", "best_val_acc", "val_f1_at_best"]
    agg = df.groupby("run", sort=False)[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.to_csv("results/week3_table_seeds.csv")
    pd.set_option("display.width", 250)
    show = ["run", "seed", "model", "precision", "augment", "lr", "scheduler", "epochs",
            "steady_epoch_s", "train_imgs_per_s", "gpu_peak_mem_mb", "best_val_acc",
            "val_f1_at_best", "best_epoch"] + (["speedup_vs_cpu"] if "speedup_vs_cpu" in df else [])
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    multi = agg[agg["val_f1_at_best_count"] > 1]
    if len(multi):
        print("\nMean +/- std over seeds:")
        for run, r in multi.iterrows():
            print(f"  {run:24s} n={int(r.val_f1_at_best_count)}  "
                  f"val_acc={r.best_val_acc_mean:.3f}+/-{r.best_val_acc_std:.3f}  "
                  f"val_f1={r.val_f1_at_best_mean:.3f}+/-{r.val_f1_at_best_std:.3f}  "
                  f"epoch={r.steady_epoch_s_mean:.2f}s")


if __name__ == "__main__":
    main()

"""Week 4 ablation (A7.6): start from the final configuration and switch off ONE optimization
at a time, on the same GPU, same split, same 5-epoch budget, 3 seeds each.

    python scripts/week4_ablation.py                       # every variant, seeds 42,43,44
    python scripts/week4_ablation.py --only full,amp_off --seeds 42

The reference ("full" = configs/optimized.yaml) is re-run in the SAME session: Colab can give
a different T4 (clocks, driver) from one day to the next, so the Week 3 numbers are not reused
as the reference. Runs whose history.json exists are skipped (use --force to redo).

Outputs (built from every run found, whatever --only says):
    results/week4/ablation/<variant>_s<seed>/history.json     one training run
    results/week4/ablation_runs.csv                           one row per run
    results/week4/ablation.csv                                mean/std per variant + deltas vs full
Checkpoints go to checkpoints/week4/ (git-ignored). The test set is NOT touched here.
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch

from src.train import main as train_main

OUT = os.path.join("results", "week4", "ablation")
CKPT = os.path.join("checkpoints", "week4", "ablation")
# (variant, config, what is switched off)
VARIANTS = [
    ("full", "configs/optimized.yaml", "nothing (final configuration)"),
    ("resident_off", "configs/exp_resident_off.yaml", "GPU-resident data (+ precompute): DataLoader + per-step GPU preprocessing"),
    ("precompute_off", "configs/exp_precompute_off.yaml", "precompute: CLAHE/crop/normalize on the GPU every step"),
    ("amp_off", "configs/exp_amp_off.yaml", "mixed precision (float32)"),
    ("channels_last_off", "configs/exp_channels_last_off.yaml", "channels_last (NCHW)"),
    ("cudnn_benchmark_off", "configs/exp_cudnn_benchmark_off.yaml", "cuDNN autotuning"),
    ("fused_adamw_off", "configs/exp_fused_adamw_off.yaml", "fused AdamW"),
    ("input_pipeline_off", "configs/exp_input_pipeline_off.yaml", "combination: resident + GPU preprocessing (CPU preprocessing in workers)"),
]
ORDER = [v for v, _, _ in VARIANTS]


def row(variant, seed, d):
    h = json.load(open(os.path.join(OUT, d, "history.json")))
    best = max(h["history"], key=lambda e: e["val_acc"])      # the checkpointed epoch
    peak = max((e["gpu_peak_mem_mb"] or 0) for e in h["history"])
    return {"variant": variant, "seed": seed, "setup_time_s": h["setup_time_s"],
            "cold_epoch_s": h["cold_epoch_s"], "steady_epoch_s": h["steady_epoch_s"],
            "wall_time_s": h["wall_time_s"],
            "train_imgs_per_s": sum(e["train_imgs_per_s"] for e in h["history"][1:])
            / max(len(h["history"]) - 1, 1),
            "gpu_peak_mem_mb": peak, "best_val_acc": h["best_val_acc"],
            "val_f1_at_best": best["val_f1"], "best_epoch": best["epoch"]}


def collect():
    rows = []
    if os.path.isdir(OUT):
        for d in sorted(os.listdir(OUT)):
            if not os.path.exists(os.path.join(OUT, d, "history.json")):
                continue
            v, s = d.rsplit("_s", 1)
            if v in ORDER and s.isdigit():
                rows.append(row(v, int(s), d))
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df["order"] = df.variant.map(ORDER.index)
    df = df.sort_values(["order", "seed"]).drop(columns="order")
    cols = ["setup_time_s", "cold_epoch_s", "steady_epoch_s", "wall_time_s", "train_imgs_per_s",
            "gpu_peak_mem_mb", "best_val_acc", "val_f1_at_best"]
    agg = df.groupby("variant", sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n"] = df.groupby("variant", sort=False).size()
    agg["switched_off"] = [dict((v, w) for v, _, w in VARIANTS)[v] for v in agg.index]
    if "full" in agg.index:
        ref = agg.loc["full"]
        # slowdown > 1: switching the optimization off makes the epoch slower by that factor,
        # i.e. the optimization is worth that speedup on top of all the others.
        agg["slowdown_vs_full"] = agg.steady_epoch_s_mean / ref.steady_epoch_s_mean
        agg["delta_epoch_s"] = agg.steady_epoch_s_mean - ref.steady_epoch_s_mean
        agg["delta_val_f1"] = agg.val_f1_at_best_mean - ref.val_f1_at_best_mean
        agg["delta_mem_mb"] = agg.gpu_peak_mem_mb_mean - ref.gpu_peak_mem_mb_mean
    return df, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=None, help="default: epochs of the config (5)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    seeds = [int(s) for s in args.seeds.split(",")]
    if not torch.cuda.is_available():
        sys.exit("the ablation needs a CUDA GPU (Colab: Runtime > Change runtime type > T4 GPU)")
    from src.utils import gpu_warmup
    gpu_warmup("cuda")
    # seed-major order: a slow drift of the GPU (thermal, clocks) spreads over every variant
    for seed in seeds:
        for v, cfg, _ in VARIANTS:
            if only and v not in only:
                continue
            d = f"{v}_s{seed}"
            if not args.force and os.path.exists(os.path.join(OUT, d, "history.json")):
                continue
            print(f"\n[ablation] {d}  ({cfg})", flush=True)
            try:
                train_main(cfg, "cuda", run=d, epochs=args.epochs, seed=seed,
                           monitor=(d == "full_s42"), out_root=OUT, ckpt_root=CKPT)
            except Exception as e:                      # one failing variant must not stop the rest
                print(f"[fail] {d}: {type(e).__name__}: {e}", flush=True)

    df, agg = collect()
    if df is None:
        print("no run found")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(os.path.join("results", "week4", "ablation_runs.csv"), index=False)
    agg.to_csv(os.path.join("results", "week4", "ablation.csv"))
    pd.set_option("display.width", 250)
    show = [c for c in ["n", "steady_epoch_s_mean", "steady_epoch_s_std", "slowdown_vs_full",
                        "gpu_peak_mem_mb_mean", "val_f1_at_best_mean", "val_f1_at_best_std",
                        "delta_val_f1"] if c in agg]
    print("\n" + agg[show].to_string(float_format=lambda v: f"{v:.3f}"))
    print("-> results/week4/ablation.csv, results/week4/ablation_runs.csv")


if __name__ == "__main__":
    main()

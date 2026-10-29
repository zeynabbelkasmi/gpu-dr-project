"""End-to-end benchmark (assignment A7.1, A7.5): complete training runs of the CPU baseline
and of the optimized GPU configuration ON THE SAME MACHINE, including loader start-up, worker
spawn, cache build or GPU upload, host/device transfers and validation.

Protocol: one discarded warm-up run per configuration (file cache, cuDNN autotune, weight
download, decoded-image cache build), then N timed repeats. Each repeat records
  wall_s          whole train.main() call (cold: setup + all epochs)
  setup_s         config, loaders, cache / GPU upload / precompute, model
  cold_epoch_s    epoch 0 (cuDNN autotuning, worker start-up)
  steady_epoch_s  mean epoch time excluding epoch 0 (train + validation, synchronized)
  total_5ep_s     setup + cold + 4 x steady: the 5-epoch training budget of both configs.
                  Measured directly when --epochs 5, extrapolated otherwise (column `extrapolated`).

    python benchmarks/benchmark_end_to_end.py                       # baseline CPU + optimized GPU
    python benchmarks/benchmark_end_to_end.py --only optimized_gpu --repeats 5

Writes results/week4/end_to_end_runs.csv (every repeat) and results/week4/end_to_end.csv
(mean/std per configuration + speedups vs baseline_cpu). Training outputs of these runs go to
results/week4/e2e/ so they never overwrite real runs; their accuracy is not used.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch

from src.train import main as train_main
from src.utils import load_json

CONFIGS = [("baseline_cpu", "configs/baseline.yaml", "cpu"),
           ("optimized_gpu", "configs/optimized.yaml", "cuda")]
BUDGET = 5            # epochs of both configs


def one_run(name, cfg_path, device, epochs, rep, out):
    run = f"{name}_r{rep}"
    train_main(cfg_path, device, run=run, epochs=epochs, out_root=os.path.join(out, "e2e"),
               ckpt_root=os.path.join("checkpoints", "week4", "e2e"))
    h = load_json(os.path.join(out, "e2e", run, "history.json"))
    return {"config": name, "device": device, "repeat": rep, "epochs": epochs,
            "wall_s": h["wall_time_s"], "setup_s": h["setup_time_s"],
            "epochs_s": h["wall_time_s"] - h["setup_time_s"],
            "cold_epoch_s": h["cold_epoch_s"], "steady_epoch_s": h["steady_epoch_s"],
            "total_5ep_s": h["setup_time_s"] + h["cold_epoch_s"] + (BUDGET - 1) * h["steady_epoch_s"],
            "extrapolated": epochs != BUDGET}


def aggregate(df):
    """A7.1: mean and std per configuration, and speedups relative to baseline_cpu."""
    cols = ["wall_s", "setup_s", "cold_epoch_s", "steady_epoch_s", "total_5ep_s"]
    agg = df.groupby("config", sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["repeats"] = df.groupby("config", sort=False).size()
    agg["epochs"] = df.groupby("config", sort=False)["epochs"].first()
    if "baseline_cpu" in agg.index:
        ref = agg.loc["baseline_cpu"]
        # speedup_steady_epoch: repeated cost only (what an extra epoch costs);
        # speedup_wall: whole measured run with its one-time costs (setup, cold epoch);
        # speedup_total_5ep: the full 5-epoch training budget, the number quoted as end-to-end.
        agg["speedup_steady_epoch"] = ref.steady_epoch_s_mean / agg.steady_epoch_s_mean
        agg["speedup_wall"] = ref.wall_s_mean / agg.wall_s_mean
        agg["speedup_total_5ep"] = ref.total_5ep_s_mean / agg.total_5ep_s_mean
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2,
                    help="epochs per timed run (2 = one cold + one steady; 5 = full budget)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu-repeats", type=int, default=None,
                    help="repeats of the CPU baseline (default: --repeats); it is slow on Colab")
    ap.add_argument("--only", default=None, help="baseline_cpu and/or optimized_gpu")
    ap.add_argument("--out", default=os.path.join("results", "week4"))
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    runs_csv = os.path.join(args.out, "end_to_end_runs.csv")
    old = pd.read_csv(runs_csv) if os.path.exists(runs_csv) else None
    runs = []
    for name, cfg_path, device in CONFIGS:
        if only and name not in only:
            continue
        if device == "cuda" and not torch.cuda.is_available():
            print(f"[e2e] skip {name}: no CUDA", flush=True)
            continue
        if device == "cuda":
            from src.utils import gpu_warmup
            gpu_warmup("cuda")
        print(f"[e2e] warmup {name} (discarded)", flush=True)
        one_run(name, cfg_path, device, 1, "warmup", args.out)
        n_rep = args.cpu_repeats if (device == "cpu" and args.cpu_repeats) else args.repeats
        for rep in range(n_rep):
            print(f"[e2e] {name} repeat {rep}", flush=True)
            runs.append(one_run(name, cfg_path, device, args.epochs, rep, args.out))
    df = pd.DataFrame(runs)
    if old is not None:                      # keep the other configuration from an earlier call
        keep = old[~old.config.isin(df.config.unique())] if len(df) else old
        df = pd.concat([keep, df], ignore_index=True)
    if df.empty:
        print("nothing ran")
        return
    order = [c for c, _, _ in CONFIGS]
    df = df.sort_values("config", key=lambda s: s.map(order.index), kind="stable")
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(runs_csv, index=False)
    agg = aggregate(df)
    agg.to_csv(os.path.join(args.out, "end_to_end.csv"))
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n" + agg.T.to_string(float_format=lambda v: f"{v:.3f}"))
    # Question for the report: speedup_steady_epoch is the larger number (it hides the
    # one-time costs: cache build, GPU upload, precompute, cuDNN autotuning); speedup_total_5ep
    # is the honest end-to-end number for our 5-epoch budget and is the one we quote.


if __name__ == "__main__":
    main()

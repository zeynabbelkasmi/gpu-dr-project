"""Week 4 final evaluation (A7.8): the ONLY script that touches the test set.

Run it once, after the configuration is frozen (profiling + ablation done, decisions taken on
validation only). For each model it
  1. trains the frozen config with seeds 42, 43, 44 (same split, same 5-epoch budget),
     checkpoint = best VALIDATION accuracy (src/train.py);
  2. sanity check on VALIDATION: src/evaluate.py on the val split must reproduce the val accuracy
     stored in the checkpoint (same preprocessing, precision and layout as training);
  3. evaluates every checkpoint on the TEST split, once, and reports all seeds (no seed is chosen
     after seeing the test numbers: the mean +/- std over the three is the result).

    python scripts/week4_test_eval.py                 # baseline + final, seeds 42,43,44

The baseline model (configs/baseline.yaml, reference CNN, fp32, CPU preprocessing) is trained
on the GPU by default: identical config and recipe, only the device changes, which does not
change what the model learns beyond run-to-run noise (Week 3: exp_cache_cpu vs exp_cache_gpu).
Pass --baseline-device cpu to train it on the CPU (slow on Colab).

A lock file results/week4/TEST_SET_USED.json records when and with which configs the test set
was used; the script refuses to run again unless --i-understand-test-reuse is given.
Outputs: results/week4/test/<model>_s<seed>/{history.json, metrics.json, metrics_val.json},
         results/week4/test_summary.csv (one row per model, mean and std over seeds),
         results/week4/test_runs.csv (one row per checkpoint).
"""
import os, sys, argparse, json, hashlib, datetime, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from src.train import main as train_main
from src.evaluate import main as eval_main
from src.utils import save_json

OUT = os.path.join("results", "week4", "test")
CKPT = os.path.join("checkpoints", "week4", "test")
LOCK = os.path.join("results", "week4", "TEST_SET_USED.json")


def sha(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-config", default="configs/optimized.yaml")
    ap.add_argument("--baseline-config", default="configs/baseline.yaml")
    ap.add_argument("--baseline-device", default="cuda")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--i-understand-test-reuse", action="store_true")
    args = ap.parse_args()
    if os.path.exists(LOCK) and not args.i_understand_test_reuse:
        sys.exit(f"{LOCK} exists: the test set was already used ({json.load(open(LOCK))['when']}). "
                 "Re-running it to pick a better number is test-set selection. Pass "
                 "--i-understand-test-reuse only to reproduce the SAME frozen configs.")
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU for the final model")
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [("baseline", args.baseline_config, args.baseline_device),
              ("final", args.final_config, "cuda")]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = None
    rows = []
    for name, cfg, dev in models:
        for seed in seeds:
            run = f"{name}_s{seed}"
            ckpt = os.path.join(CKPT, run, "best.pt")
            if not os.path.exists(ckpt):
                print(f"\n[test-eval] train {run} ({cfg}, {dev})", flush=True)
                train_main(cfg, dev, run=run, seed=seed, out_root=OUT, ckpt_root=CKPT)
            # evaluation device: the GPU for both (inference speed is not compared here)
            val = eval_main(cfg, run=run, device_str="cuda", checkpoint=ckpt, out_root=OUT,
                            split="val")
            gap = abs(val["accuracy"] - val["val_acc"])
            print(f"[check] {run}: val acc re-computed {val['accuracy']:.4f} vs checkpoint "
                  f"{val['val_acc']:.4f} (gap {gap:.4f})", flush=True)
            test = eval_main(cfg, run=run, device_str="cuda", checkpoint=ckpt, out_root=OUT,
                             split="test")
            rows.append({"model": name, "config": cfg, "train_device": dev, "seed": seed,
                         "checkpoint_epoch": test["checkpoint_epoch"],
                         "val_acc_ckpt": val["val_acc"], "val_acc_recomputed": val["accuracy"],
                         "val_macro_f1": val["macro_f1"], "val_check_gap": gap,
                         "test_accuracy": test["accuracy"],
                         "test_macro_precision": test["macro_precision"],
                         "test_macro_recall": test["macro_recall"],
                         "test_macro_f1": test["macro_f1"],
                         **{f"test_recall_c{c}": test["per_class"][str(c)]["recall"]
                            for c in range(5)},
                         "test_inference_imgs_per_s": test["inference_imgs_per_s"]})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(os.path.join("results", "week4", "test_runs.csv"), index=False)
    num = [c for c in df.columns if df[c].dtype != object and c not in ("seed",)]
    agg = df.groupby("model", sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    # confusion matrices summed over seeds (each seed sees the same 450 test images)
    for name, _, _ in models:
        cms = [np.array(json.load(open(os.path.join(OUT, f"{name}_s{s}", "metrics.json")))
                        ["confusion_matrix"]) for s in seeds]
        save_json({"seeds": seeds, "sum": sum(cms).tolist(), "per_seed": [c.tolist() for c in cms]},
                  os.path.join("results", "week4", f"confusion_{name}.json"))
    agg.to_csv(os.path.join("results", "week4", "test_summary.csv"))
    save_json({"when": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
               "configs": {n: {"path": c, "md5": sha(c)} for n, c, _ in models}, "seeds": seeds},
              LOCK)
    pd.set_option("display.width", 250)
    show = ["test_accuracy_mean", "test_accuracy_std", "test_macro_f1_mean", "test_macro_f1_std",
            "test_macro_precision_mean", "test_macro_recall_mean", "val_macro_f1_mean",
            "val_check_gap_mean"]
    print("\n" + agg[show].T.to_string(float_format=lambda v: f"{v:.4f}"))
    print("-> results/week4/test_summary.csv, test_runs.csv, confusion_*.json")


if __name__ == "__main__":
    main()

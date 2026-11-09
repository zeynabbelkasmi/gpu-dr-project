"""Ordinal view of the test results: within-one-grade accuracy and quadratic weighted kappa.

DR grades are ordered (0 < 1 < 2 < 3 < 4), so predicting grade 2 for a grade-3 image is a
smaller error than predicting grade 0. Accuracy and macro-F1 count both the same. This script
adds two standard ordinal metrics, computed ONLY from the confusion matrices that
scripts/week4_test_eval.py already wrote (results/week4/confusion_{baseline,final}.json).
It does not load a model, does not touch the test images and changes no decision: it is a
second reading of the single test evaluation, added after that evaluation.

  within_one   fraction of images whose predicted grade is at most one grade from the truth
  off_by_2     fraction off by two grades or more (= 1 - within_one)
  qwk          quadratic weighted Cohen's kappa, weights (i - j)^2, observed vs chance
               agreement from the row/column totals (same as scikit-learn
               cohen_kappa_score(weights="quadratic")); 0 = chance level, 1 = perfect

Per seed, then mean and sample standard deviation over the seeds (as test_summary.csv).

    python scripts/ordinal_metrics.py        -> results/week4/ordinal_metrics.csv
"""
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import RESULTS_DIR

WEEK4 = RESULTS_DIR / "week4"
MODELS = ("baseline", "final")


def ordinal_metrics(conf):
    """conf: (C, C) list of lists, rows = true grade, columns = predicted grade."""
    c = len(conf)
    n = sum(map(sum, conf))
    rows = [sum(r) for r in conf]
    cols = [sum(conf[i][j] for i in range(c)) for j in range(c)]
    cells = [(i, j) for i in range(c) for j in range(c)]
    accuracy = sum(conf[i][i] for i in range(c)) / n
    within_one = sum(conf[i][j] for i, j in cells if abs(i - j) <= 1) / n
    observed = sum((i - j) ** 2 * conf[i][j] for i, j in cells) / n
    expected = sum((i - j) ** 2 * rows[i] * cols[j] for i, j in cells) / n / n
    return {"n": n, "accuracy": accuracy, "within_one": within_one,
            "off_by_2": 1 - within_one, "qwk": 1 - observed / expected}


def main():
    keys = ("accuracy", "within_one", "off_by_2", "qwk")
    out_rows = []
    for model in MODELS:
        with open(WEEK4 / f"confusion_{model}.json") as f:
            d = json.load(f)
        per_seed = [ordinal_metrics(m) for m in d["per_seed"]]
        for seed, m in zip(d["seeds"], per_seed):
            out_rows.append({"model": model, "seed": seed, **m})
        summary = {"model": model, "seed": "mean", "n": per_seed[0]["n"]}
        spread = {"model": model, "seed": "std", "n": per_seed[0]["n"]}
        for k in keys:
            vals = [m[k] for m in per_seed]
            summary[k] = statistics.mean(vals)
            spread[k] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out_rows += [summary, spread]
        print(f"{model:8s} " + "  ".join(
            f"{k}={summary[k]:.3f}+/-{spread[k]:.3f}" for k in keys))

    out = WEEK4 / "ordinal_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "n", *keys])
        w.writeheader()
        w.writerows(out_rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

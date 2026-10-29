"""Build the mandatory figures and tables from results/ (assignment A8.1-A8.6).

Read only measured artifacts. Never type a number by hand.

    python scripts/make_figures.py

Inputs (a figure whose inputs are missing is skipped with a message, or uses the documented
fallback):
  results/profile_stages.json            baseline time per stage, laptop CPU   (benchmarks/profile_baseline.py)
  results/timing.csv                     per-stage CPU vs GPU, 64 images        (benchmarks/benchmark_stages.py)
  results/week3_table.csv                every Week 3 run and seed, Colab T4    (scripts/week3_experiments.py)
  results/week4/end_to_end.csv           baseline CPU vs optimized GPU, same machine (benchmarks/benchmark_end_to_end.py)
  results/week4/ablation.csv             one optimization off at a time        (scripts/week4_ablation.py)
  results/week4/test_summary.csv         test metrics, baseline and final      (scripts/week4_test_eval.py)
  results/week4/confusion_*.json         test confusion matrices (summed over seeds)
  results/week4/profile_optimized/summary.json   GPU profile of the final config (benchmarks/profile_gpu.py)
  results/baseline/resources.csv, results/week4/ablation/full_s42/resources.csv (or results/optimized/)
Outputs: results/figures/fig{1..5}_*.png, results/figures/fig6_confusion.png and results/tables.md
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join("results", "figures")
W4 = os.path.join("results", "week4")
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f5bd6"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a3a19a"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})
MACHINE_LAPTOP = "RTX 4070 Laptop + 8-core Ryzen (Weeks 1-2)"
MACHINE_COLAB = "Colab T4 + 1-core CPU (Weeks 3-4)"


def _p(*a):
    return os.path.join(*a)


def _have(path, what):
    if os.path.exists(path):
        return True
    print(f"  skip {what}: {path} missing")
    return False


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(_p(OUT, name), dpi=170)
    plt.close(fig)
    print(f"  -> {_p(OUT, name)}")


def _read(path):
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


def _week3():
    """Week 3 runs aggregated over seeds: mean/std of steady epoch and val F1, model, size."""
    path = _p("results", "week3_table.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    g = df.groupby("run", sort=False)
    out = g.agg(model=("model", "first"), device=("device", "first"),
                image_size=("image_size", "first"), batch_size=("batch_size", "first"),
                epochs=("epochs", "first"), n=("seed", "size"),
                epoch=("steady_epoch_s", "mean"), epoch_std=("steady_epoch_s", "std"),
                f1=("val_f1_at_best", "mean"), f1_std=("val_f1_at_best", "std"),
                mem=("gpu_peak_mem_mb", "mean"))
    return out


# ----------------------------------------------------------------------------- Figure 1
def fig1_baseline_breakdown():
    """Figure 1. Baseline runtime per stage (share of one CPU epoch), from profile_stages.json."""
    path = _p("results", "profile_stages.json")
    if not _have(path, "Figure 1"):
        return
    d = json.load(open(path))
    top = {k: v for k, v in d["stages_s"].items() if not k.startswith(" ")}
    sub = {k.strip(): v for k, v in d["stages_s"].items() if k.startswith(" ")}
    names = list(top)[::-1]
    vals = [top[k] for k in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [C2 if k.startswith("loading") else C1 for k in names]
    ax.barh(names, vals, color=colors)
    for i, (k, v) in enumerate(zip(names, vals)):
        ax.text(v + d["total_s"] * 0.01, i, f"{v:.1f} s  ({100 * v / d['total_s']:.1f} %)",
                va="center", fontsize=9, color=INK2)
    if sub:
        detail = ", ".join(f"{k} {v:.1f} s" for k, v in sub.items())
        ax.text(0.99, 0.02, f"preprocessing = {detail}", transform=ax.transAxes, ha="right",
                fontsize=8, color=INK2)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("time in one baseline epoch (s)")
    ax.set_title(f"Fig. 1 - Baseline CPU epoch: {d['total_s']:.0f} s, loading dominates")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, f"cProfile, 1 epoch, 0 workers, reference CNN 224 px. {MACHINE_LAPTOP}.",
             fontsize=7, color=MUTED)
    _save(fig, "fig1_baseline_breakdown.png")


# ----------------------------------------------------------------------------- Figure 2
def _stage_speedups():
    """(label, speedup, kind) per stage from timing.csv (64 images, same laptop for CPU/GPU)."""
    path = _p("results", "timing.csv")
    if not os.path.exists(path):
        return []
    t = pd.read_csv(path)
    cpu = t[t.device == "cpu"].set_index("stage").mean_s
    gpu = t[t.device == "cuda"].set_index("stage").mean_s
    rows = []
    if "loading_cache" in cpu and "loading" in cpu:
        rows.append(("loading: decoded cache (CPU)", cpu["loading"] / cpu["loading_cache"], "algorithm"))
    if "loading" in gpu:
        rows.append(("loading: GPU decode", cpu["loading"] / gpu["loading"], "gpu"))
    labels = {"preprocessing": "preprocessing (CLAHE, crop, norm.)", "augmentation": "augmentation",
              "filter_gaussian": "filter: Gaussian", "filter_median": "filter: median",
              "filter_green": "filter: green channel", "filter_contrast": "filter: contrast",
              "train_step": "train step fp32", "train_step_amp": "train step AMP",
              "inference": "inference fp32", "inference_amp": "inference AMP"}
    for k, lab in labels.items():
        base = k.replace("_amp", "")
        if k in gpu and base in cpu:
            rows.append((lab, cpu[base] / gpu[k], "gpu"))
    return rows


def _end_to_end():
    """End-to-end speedups, same machine. week4/end_to_end.csv, else Week 3 steady epochs."""
    e = _read(_p(W4, "end_to_end.csv"))
    rows, src = [], None
    if e is not None and "speedup_steady_epoch" in e and "optimized_gpu" in e.index:
        r = e.loc["optimized_gpu"]
        rows += [("END-TO-END: steady epoch", r.speedup_steady_epoch, "e2e"),
                 ("END-TO-END: 5-epoch training", r.speedup_total_5ep, "e2e")]
        src = "results/week4/end_to_end.csv"
    else:
        w3 = _week3()
        if w3 is not None and "exp_cache_cpu" in w3.index:
            ref = w3.loc["exp_cache_cpu", "epoch"]
            for run, lab in (("optimized_cnn", "END-TO-END epoch, same model (CNN)"),
                             ("optimized", "END-TO-END epoch, final design")):
                if run in w3.index:
                    rows.append((lab, ref / w3.loc[run, "epoch"], "e2e"))
            src = "results/week3_table.csv (fallback: week4 end_to_end.csv missing)"
    return rows, src


def fig2_speedup():
    """Figure 2. Stage-level speedups next to the end-to-end speedup (log scale, line at 1)."""
    stages = _stage_speedups()
    e2e, src = _end_to_end()
    rows = stages + e2e
    if not rows:
        print("  skip Figure 2: no timing.csv and no end-to-end results")
        return
    colors = {"algorithm": C3, "gpu": C1, "e2e": C2}
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[colors[r[2]] for r in rows])
    for yi, r in zip(y, rows):
        ax.text(r[1] * 1.08, yi, f"{r[1]:.1f}x", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xscale("log")
    ax.axvline(1, color=INK, lw=1)
    ax.set_xlim(0.5, max(r[1] for r in rows) * 4)
    ax.set_xlabel("speedup = T_baseline / T_optimized (log scale)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C3, label="algorithmic (no GPU)"),
                       Patch(color=C1, label="GPU, isolated stage (64 images)"),
                       Patch(color=C2, label="end-to-end, whole training")],
              loc="lower right", fontsize=8)
    ax.set_title("Fig. 2 - Stage speedups are much larger than the end-to-end speedup")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.005, f"Stages: results/timing.csv, {MACHINE_LAPTOP}. End-to-end: {src}, "
             f"{MACHINE_COLAB}.", fontsize=6.5, color=MUTED)
    _save(fig, "fig2_speedup.png")


# ----------------------------------------------------------------------------- Figure 3
def fig3_accuracy_vs_time():
    """Figure 3. One point per configuration: steady epoch time vs validation macro-F1
    (mean +/- std over seeds). Test macro-F1 of the baseline and final model as stars."""
    w3 = _week3()
    if w3 is None:
        print("  skip Figure 3: results/week3_table.csv missing")
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    mcol = {"reference_cnn": C1, "resnet18": C2, "efficientnet_b0": C3, "resnet50": C5}
    keep = {"exp_cache_gpu": "ref. CNN, GPU", "optimized_cnn": "optimized_cnn",
            "exp_compile_ro_gpu": "CNN + CUDA graphs", "exp_resnet18_gpu": "ResNet-18, default pipeline",
            "exp_r18_cosine_gpu": "R18 224 px cosine", "exp_r18_augment_gpu": "R18 + augment",
            "exp_r18_regul_gpu": "R18 + regularization", "exp_effb0_gpu": "EfficientNet-B0",
            "exp_r50_gpu": "ResNet-50", "exp_r18_res384_bs64_gpu": "R18 384 px",
            "optimized": "FINAL: R18 320 px, batch 64"}
    w3 = w3.drop(index=[r for r in w3.index if w3.loc[r, "device"] == "cpu"])  # 2-epoch timing run
    for run, r in w3.iterrows():
        c = mcol.get(r.model, MUTED)
        ax.errorbar(r.epoch, r.f1, yerr=(r.f1_std if r.n > 1 else None),
                    xerr=(r.epoch_std if r.n > 1 else None), fmt="o", color=c,
                    ms=5 + 1.5 * (r.n > 1), capsize=2, lw=1, alpha=0.9)
        if run in keep:
            off = {"exp_effb0_gpu": (6, -12), "exp_r50_gpu": (6, -2), "optimized": (-40, 14),
                   "exp_r18_augment_gpu": (6, -10)}.get(run, (5, 4))
            ax.annotate(keep[run], (r.epoch, r.f1), xytext=off, textcoords="offset points",
                        fontsize=7.5, color=INK if run == "optimized" else INK2,
                        fontweight="bold" if run == "optimized" else "normal")
        if run == "optimized":
            ax.plot(r.epoch, r.f1, "o", ms=14, mfc="none", mec=INK, mew=1.2)
    test = _read(_p(W4, "test_summary.csv"))
    if test is not None:
        abl = _read(_p(W4, "ablation.csv"))
        e2e = _read(_p(W4, "end_to_end.csv"))
        x = {"final": (abl.loc["full", "steady_epoch_s_mean"] if abl is not None and "full" in abl.index
                       else w3.loc["optimized", "epoch"] if "optimized" in w3.index else None),
             "baseline": (e2e.loc["baseline_cpu", "steady_epoch_s_mean"]
                          if e2e is not None and "baseline_cpu" in e2e.index
                          else w3.loc["exp_cache_cpu", "epoch"] if "exp_cache_cpu" in w3.index else None)}
        for name, col in (("baseline", C1), ("final", C2)):
            if name in test.index and x[name] is not None:
                r = test.loc[name]
                ax.errorbar(x[name], r.test_macro_f1_mean, yerr=r.test_macro_f1_std, fmt="*",
                            ms=16, color=col, mec=INK, capsize=3)
                ax.annotate(f"TEST {name}: F1 {r.test_macro_f1_mean:.3f}", (x[name], r.test_macro_f1_mean),
                            xytext=(8, -14), textcoords="offset points", fontsize=8.5,
                            fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("steady-state epoch time, train + val (s, log scale)")
    ax.set_ylabel("macro-F1 (validation: dots; test: stars)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in mcol.items()]
              + [Line2D([], [], marker="*", ls="", color=MUTED, mec=INK, ms=12, label="test set (once)")],
              loc="center right", fontsize=8)
    ax.set_title("Fig. 3 - Accuracy vs execution time (error bars: std over 3 seeds)")
    fig.text(0.01, 0.005, f"Validation points: results/week3_table.csv; test: results/week4/test_summary.csv. "
             f"{MACHINE_COLAB}. Unlabelled orange dots: other single-change ResNet-18 runs (week3_table.csv).", fontsize=6.5, color=MUTED)
    _save(fig, "fig3_accuracy_vs_time.png")


# ----------------------------------------------------------------------------- Figure 4
def fig4_ablation():
    """Figure 4. Ablation from the final configuration: one optimization off per bar."""
    a = _read(_p(W4, "ablation.csv"))
    if a is None:
        print(f"  skip Figure 4: {W4}/ablation.csv missing (run scripts/week4_ablation.py)")
        return
    names = list(a.index)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.45 * len(names) + 1.8), sharey=True,
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(names))[::-1]
    col = [C2 if n == "full" else (C4 if "input_pipeline" in n else C1) for n in names]
    ax1.barh(y, a.steady_epoch_s_mean, xerr=a.steady_epoch_s_std.fillna(0), color=col, capsize=2)
    for yi, n in zip(y, names):
        r = a.loc[n]
        txt = f"{r.steady_epoch_s_mean:.2f} s" + (f"  (x{r.slowdown_vs_full:.2f})" if n != "full" and
                                                  "slowdown_vs_full" in a else "")
        ax1.text(r.steady_epoch_s_mean + (r.steady_epoch_s_std if r.steady_epoch_s_std == r.steady_epoch_s_std else 0)
                 + a.steady_epoch_s_mean.max() * 0.02, yi, txt, va="center", fontsize=8.5, color=INK2)
    ax1.set_yticks(y, [("FULL (final config)" if n == "full" else n.replace("_off", " OFF"))
                       for n in names])
    ax1.set_xlim(0, a.steady_epoch_s_mean.max() * 1.45)
    ax1.set_xlabel("steady epoch, train + val (s)")
    ax1.set_title("Epoch time (x: slowdown vs full)")
    ax1.grid(axis="y", visible=False)
    if "full" in a.index:
        ax1.axvline(a.loc["full", "steady_epoch_s_mean"], color=C2, ls="--", lw=1)
    ax2.barh(y, a.val_f1_at_best_mean, xerr=a.val_f1_at_best_std.fillna(0), color=col, capsize=2,
             alpha=0.85)
    lo = max(0, (a.val_f1_at_best_mean - a.val_f1_at_best_std.fillna(0)).min() - 0.05)
    ax2.set_xlim(lo, min(1, a.val_f1_at_best_mean.max() + 0.06))
    if "full" in a.index:
        ax2.axvline(a.loc["full", "val_f1_at_best_mean"], color=C2, ls="--", lw=1)
    ax2.set_xlabel("validation macro-F1")
    ax2.set_title("Quality (unchanged = pure speed)")
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Fig. 4 - Ablation: final configuration with one optimization switched off "
                 f"(mean +/- std, n = {int(a.n.max())} seeds)", fontweight="bold", fontsize=11)
    fig.text(0.01, 0.005, f"results/week4/ablation.csv. {MACHINE_COLAB}. Orange bar: defined combination.",
             fontsize=6.5, color=MUTED)
    _save(fig, "fig4_ablation.png")


# ----------------------------------------------------------------------------- Figure 5
def fig5_resources():
    """Figure 5. CPU and GPU utilization and memory over time, baseline vs optimized."""
    base = _p("results", "baseline", "resources.csv")
    opt = next((p for p in (_p(W4, "ablation", "full_s42", "resources.csv"),
                            _p("results", "optimized", "resources.csv")) if os.path.exists(p)), None)
    runs = [(n, p, m) for n, p, m in (("Baseline (CPU)", base, MACHINE_LAPTOP),
                                       ("Optimized (GPU)", opt, MACHINE_COLAB)) if p and os.path.exists(p)]
    if not runs:
        print("  skip Figure 5: no resources.csv")
        return
    fig, axes = plt.subplots(3, len(runs), figsize=(5.2 * len(runs), 6.6), squeeze=False)
    for j, (name, path, machine) in enumerate(runs):
        r = pd.read_csv(path)
        t = r.t_s
        a = axes[0, j]
        a.plot(t, r.cpu_percent, color=C1, lw=1)
        a.set_ylim(0, 105)
        a.set_ylabel("CPU util. (%)")
        a.set_title(f"{name}\n{os.path.relpath(path)}", fontsize=9)
        a = axes[1, j]
        a.plot(t, r.gpu_util_percent, color=C2, lw=1)
        a.set_ylim(0, 105)
        a.set_ylabel("GPU util. (%)")
        a.text(0.99, 0.05, f"mean {r.gpu_util_percent.mean():.0f} %", transform=a.transAxes,
               ha="right", fontsize=8, color=INK2)
        axes[0, j].text(0.99, 0.05, f"mean {r.cpu_percent.mean():.0f} %", transform=axes[0, j].transAxes,
                        ha="right", fontsize=8, color=INK2)
        a = axes[2, j]
        a.plot(t, r.proc_tree_rss_gb, color=C1, lw=1, label="process RAM (GB)")
        a.set_ylabel("RAM (GB)")
        a2 = a.twinx()
        a2.plot(t, r.gpu_mem_used_mb / 1024, color=C2, lw=1, label="GPU memory (GB)")
        a2.set_ylabel("GPU mem (GB)", color=C2)
        a2.grid(False)
        a.set_xlabel(f"time (s) - {machine}", fontsize=8)
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right", frameon=True, framealpha=0.9, edgecolor=GRID)
    fig.suptitle("Fig. 5 - Resource utilization (1 s samples, benchmarks/resource_monitor.py)",
                 fontweight="bold", fontsize=11)
    _save(fig, "fig5_resources.png")


def fig6_confusion():
    """Extra: test confusion matrices (summed over seeds), baseline vs final."""
    mats = [(n, _p(W4, f"confusion_{n}.json")) for n in ("baseline", "final")]
    mats = [(n, json.load(open(p))) for n, p in mats if os.path.exists(p)]
    if not mats:
        print("  skip Figure 6: no results/week4/confusion_*.json")
        return
    fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 4.2), squeeze=False)
    for ax, (n, d) in zip(axes[0], mats):
        m = np.array(d["sum"], dtype=float)
        norm = m / m.sum(1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for k in range(5):
                ax.text(k, i, f"{100 * norm[i, k]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, k] > 0.5 else INK)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xlabel("predicted grade")
        ax.set_ylabel("true grade")
        ax.grid(False)
        ax.set_title(f"{n} - test, {len(d['seeds'])} seeds summed (% of row)", fontsize=9)
    fig.suptitle("Fig. 6 - Test confusion matrices", fontweight="bold", fontsize=11)
    _save(fig, "fig6_confusion.png")


# ----------------------------------------------------------------------------- Tables
def _f(v, d=3, s=""):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}{s}"


def tables():
    """Tables 2 and 3 of the brief, the ablation table and the Amdahl analysis -> results/tables.md."""
    L = ["# Tables (generated by scripts/make_figures.py - do not edit by hand)", ""]
    # --- Table 2: stages (laptop, 64 images) + epoch/total (Colab, end-to-end)
    L += ["## Table 2 - Stage-level performance", "",
          f"Stage rows: `results/timing.csv`, 64 images, {MACHINE_LAPTOP}, CPU and GPU on the same "
          f"machine. Epoch and total rows: `results/week4/end_to_end.csv`, {MACHINE_COLAB}, both "
          "configurations on the same machine.", "",
          "| Pipeline stage | CPU baseline (s) | Optimized (s) | Speedup | Optimized = |", "|---|---|---|---|---|"]
    tpath = _p("results", "timing.csv")
    if os.path.exists(tpath):
        t = pd.read_csv(tpath)
        cpu = t[t.device == "cpu"].set_index("stage").mean_s
        gpu = t[t.device == "cuda"].set_index("stage").mean_s
        def row(label, c, g, how):
            L.append(f"| {label} | {_f(c, 4)} | {_f(g, 4)} | {_f(c / g, 1, 'x')} | {how} |")
        row("Image loading (64 img)", cpu["loading"], cpu["loading_cache"], "decoded uint8 cache, CPU (A3)")
        row("Preprocessing (64 img)", cpu["preprocessing"], gpu["preprocessing"], "batched GPU kernels (A4)")
        fl = [k for k in ("filter_gaussian", "filter_median", "filter_green", "filter_contrast") if k in gpu]
        row("Filtering, 4 filters (64 img)", sum(cpu[k] for k in fl), sum(gpu[k] for k in fl), "GPU (A4)")
        row("Augmentation (64 img)", cpu["augmentation"], gpu["augmentation"], "GPU flip + rotation (A4)")
        row("Train step (64 img, ref. CNN)", cpu["train_step"], gpu["train_step_amp"], "GPU + AMP")
        row("Evaluation / inference (64 img)", cpu["inference"], gpu["inference_amp"], "GPU + AMP")
    e = _read(_p(W4, "end_to_end.csv"))
    if e is not None and {"baseline_cpu", "optimized_gpu"} <= set(e.index):
        b, o = e.loc["baseline_cpu"], e.loc["optimized_gpu"]
        L.append(f"| Training per epoch (steady, train+val) | {_f(b.steady_epoch_s_mean, 2)} +/- {_f(b.steady_epoch_s_std, 2)} | "
                 f"{_f(o.steady_epoch_s_mean, 2)} +/- {_f(o.steady_epoch_s_std, 2)} | {_f(o.speedup_steady_epoch, 1, 'x')} | final config |")
        L.append(f"| One-time setup (cache/upload/precompute) | {_f(b.setup_s_mean, 1)} | {_f(o.setup_s_mean, 1)} | "
                 f"{_f(b.setup_s_mean / o.setup_s_mean, 1, 'x')} | |")
        ext = " (extrapolated: setup + cold + 4 x steady)" if bool(e.get("epochs", pd.Series([5])).iloc[0] != 5) else ""
        L.append(f"| **Total pipeline, 5-epoch training{ext}** | {_f(b.total_5ep_s_mean, 1)} | "
                 f"{_f(o.total_5ep_s_mean, 1)} | **{_f(o.speedup_total_5ep, 1, 'x')}** | final config |")
    else:
        L.append("| Training per epoch / Total pipeline | pending | pending | | run benchmarks/benchmark_end_to_end.py |")
    L += ["", "The baseline trains the reference CNN at 224 px; the final configuration trains ResNet-18 at 320 px. "
          "The decomposition below separates the GPU/pipeline gain from the model change.", ""]
    w3 = _week3()
    if w3 is not None and {"exp_cache_cpu", "exp_cache_gpu", "optimized_cnn", "optimized"} <= set(w3.index):
        ref = w3.loc["exp_cache_cpu", "epoch"]
        L += ["| Step (same Colab machine, steady epoch) | Run | Epoch (s) | vs CPU | Val macro-F1 |", "|---|---|---|---|---|"]
        for lab, run in (("CPU reference (cache)", "exp_cache_cpu"), ("+ GPU, same model (device effect)", "exp_cache_gpu"),
                         ("+ GPU pipeline + AMP etc., same model", "optimized_cnn"),
                         ("+ ResNet-18, cosine, 320 px, batch 64 (final)", "optimized")):
            r = w3.loc[run]
            f1 = f"{r.f1:.3f}" + (f" +/- {r.f1_std:.3f}" if r.n > 1 else "") if run != "exp_cache_cpu" else "n/a (2 epochs)"
            L.append(f"| {lab} | `{run}` | {r.epoch:.2f} | {ref / r.epoch:.1f}x | {f1} |")
    # --- Table 3
    L += ["", "## Table 3 - Classification quality (TEST set, used once, mean +/- std over seeds)", ""]
    ts = _read(_p(W4, "test_summary.csv"))
    if ts is not None and {"baseline", "final"} <= set(ts.index):
        L += ["| Metric | Baseline model | Final model |", "|---|---|---|"]
        for lab, k in (("Accuracy", "test_accuracy"), ("Macro-precision", "test_macro_precision"),
                       ("Macro-recall", "test_macro_recall"), ("Macro-F1", "test_macro_f1")):
            L.append(f"| {lab} | {_f(ts.loc['baseline', k + '_mean'])} +/- {_f(ts.loc['baseline', k + '_std'])} | "
                     f"{_f(ts.loc['final', k + '_mean'])} +/- {_f(ts.loc['final', k + '_std'])} |")
        for c in range(5):
            k = f"test_recall_c{c}"
            if k + "_mean" in ts:
                L.append(f"| Recall grade {c} | {_f(ts.loc['baseline', k + '_mean'])} | {_f(ts.loc['final', k + '_mean'])} |")
        L.append("| Confusion matrix | Fig. 6, `results/week4/confusion_baseline.json` | Fig. 6, `results/week4/confusion_final.json` |")
        L.append(f"| Validation macro-F1 (same checkpoints) | {_f(ts.loc['baseline', 'val_macro_f1_mean'])} | {_f(ts.loc['final', 'val_macro_f1_mean'])} |")
    else:
        L.append("pending: run `scripts/week4_test_eval.py` once the configuration is frozen.")
    # --- ablation
    a = _read(_p(W4, "ablation.csv"))
    L += ["", "## Ablation (A7.6) - final configuration with one optimization off", ""]
    if a is not None:
        L += ["| Variant | Switched off | n | Steady epoch (s) | Slowdown vs full | Peak GPU mem (MB) | Val macro-F1 |",
              "|---|---|---|---|---|---|---|"]
        for n, r in a.iterrows():
            L.append(f"| `{n}` | {r.switched_off} | {int(r.n)} | {_f(r.steady_epoch_s_mean, 2)} +/- {_f(r.steady_epoch_s_std, 2)} | "
                     f"{_f(r.get('slowdown_vs_full'), 2, 'x')} | {_f(r.gpu_peak_mem_mb_mean, 0)} | "
                     f"{_f(r.val_f1_at_best_mean)} +/- {_f(r.val_f1_at_best_std)} |")
    else:
        L.append("pending: run `scripts/week4_ablation.py`.")
    # --- profile
    prof = _p(W4, "profile_optimized", "summary.json")
    L += ["", "## GPU profile of the final configuration (A7.2-A7.4, A7.7)", ""]
    if os.path.exists(prof):
        s = json.load(open(prof))
        p = s["profiler"]
        L += [f"- {s['gpu']}, torch {s['torch']}: train step {s['step_ms_unsynchronized']:.1f} ms "
              f"({s['imgs_per_s_train_step']:.0f} img/s); GPU busy {p['gpu_busy_ms_per_step']:.1f} ms/step, "
              f"**idle {100 * p['gpu_idle_fraction']:.1f} %**; {p['kernels_per_step']:.0f} kernels/step "
              f"(mean {p['mean_kernel_us']:.0f} us).",
              "- Host-to-device per step: " + (", ".join(f"{k} {v['bytes_per_step'] / 2**20:.2f} MiB in {v['ms_per_step']:.3f} ms"
                                                     for k, v in p["memcpy"].items()) or "none") + ".", "",
              "| GPU kernel category | ms / step | share |", "|---|---|---|"]
        for k, v in p["kernel_categories_ms_per_step"].items():
            L.append(f"| {k} | {v:.2f} | {100 * p['kernel_categories_share'][k]:.1f} % |")
        ph = s["phases_ms_synchronized"]
        L += ["", "| Phase (synchronized) | ms / step | share |", "|---|---|---|"]
        for k, v in ph.items():
            L.append(f"| {k} | {v:.2f} | {100 * s['phases_share_synchronized'][k]:.1f} % |")
        if s.get("epoch"):
            ep = s["epoch"]
            L += ["", f"Epoch decomposition: train {ep['train_epoch_s']:.2f} s ({ep['train_steps']} steps) + "
                  f"validation {ep['val_epoch_s']:.2f} s = {ep['epoch_s']:.2f} s (validation {100 * ep['val_share']:.0f} %)."]
    else:
        L.append("pending: run `benchmarks/profile_gpu.py`.")
    # --- Amdahl
    L += ["", "## Amdahl analysis", ""]
    ps = _p("results", "profile_stages.json")
    if os.path.exists(ps) and os.path.exists(tpath):
        fr = json.load(open(ps))["fractions"]
        t = pd.read_csv(tpath)
        cpu = t[t.device == "cpu"].set_index("stage").mean_s
        gpu = t[t.device == "cuda"].set_index("stage").mean_s
        p_load = fr["loading (decode+resize)"]
        s_load = cpu["loading"] / cpu["loading_cache"]
        S1 = 1 / ((1 - p_load) + p_load / s_load)
        L += [f"- Main optimization, decoded-image cache: p = {p_load:.3f} of the baseline epoch, stage speedup "
              f"s = {s_load:.0f}x -> S = 1 / ((1 - p) + p / s) = **{S1:.2f}x** (bound 1 / (1 - p) = {1 / (1 - p_load):.2f}x). "
              "A huge stage speedup buys less than 3x because 35 % of the epoch is untouched."]
        parts = {"loading (decode+resize)": s_load, "preprocessing (total)": cpu["preprocessing"] / gpu["preprocessing"]}
        s_train = cpu["train_step"] / gpu["train_step_amp"]
        for k in ("forward", "backward", "optimizer_step"):
            parts[k] = s_train
        denom = sum(fr[k] / v for k, v in parts.items()) + (1 - sum(fr[k] for k in parts))
        S2 = 1 / denom
        L.append(f"- All measured stages at their isolated speedups (loading {s_load:.0f}x, preprocessing "
                 f"{parts['preprocessing (total)']:.1f}x, training step {s_train:.1f}x, 'other' "
                 f"{100 * (1 - sum(fr[k] for k in parts)):.1f} % unchanged): generalized Amdahl "
                 f"S = 1 / sum(p_i / s_i) = **{S2:.1f}x**.")
        if w3 is not None and {"exp_cache_cpu", "optimized_cnn"} <= set(w3.index):
            L.append(f"- Measured, same model, same Colab machine: {w3.loc['exp_cache_cpu', 'epoch'] / w3.loc['optimized_cnn', 'epoch']:.0f}x "
                     "per steady epoch (`optimized_cnn` vs `exp_cache_cpu`). The prediction uses laptop-CPU fractions "
                     "and speedups; on Colab the CPU has 1 core, which makes every CPU stage slower and the "
                     "measured ratio larger: the two numbers are not from the same machine.")
        if e is not None and "optimized_gpu" in e.index:
            o = e.loc["optimized_gpu"]
            L.append(f"- Measured end-to-end, final design, same machine: {o.speedup_steady_epoch:.1f}x per steady epoch, "
                     f"{o.speedup_total_5ep:.1f}x for the 5-epoch training including one-time costs (the model is 3.6x "
                     "more FLOPs per image at 224 px and 2x the pixels at 320 px).")
    os.makedirs("results", exist_ok=True)
    open(_p("results", "tables.md"), "w").write("\n".join(L) + "\n")
    print("  -> results/tables.md")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig1_baseline_breakdown, fig2_speedup, fig3_accuracy_vs_time,
               fig4_ablation, fig5_resources, fig6_confusion, tables):
        print(fn.__name__)
        fn()

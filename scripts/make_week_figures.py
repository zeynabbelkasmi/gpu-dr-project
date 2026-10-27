"""Figures of the Week 1 and Week 2 reports, built only from files in results/.

    python scripts/make_week_figures.py

Outputs results/figures/week1_*.png and results/figures/week2_*.png. A figure whose
input file is missing is skipped with a message (run the matching benchmark first).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.join("results", "figures")
# categorical slots in fixed order (CPU = slot 1, GPU = slot 2, third series = slot 3)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _need(path):
    if not os.path.exists(path):
        print(f"skip: {path} missing")
        return False
    return True


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def week1_breakdown():
    """Figure 1: share of the profiled baseline epoch per stage (cProfile)."""
    p = "results/profile_stages.json"
    if not _need(p):
        return
    d = json.load(open(p))
    top = ["loading (decode+resize)", "preprocessing (total)", "forward", "backward",
           "optimizer_step", "other"]
    s = pd.Series({k: d["stages_s"][k] for k in top}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(s.index, s.values, color=C1, height=0.6)
    for y, (k, v) in enumerate(s.items()):
        ax.text(v, y, f"  {v:.1f} s ({100 * v / d['total_s']:.1f}%)", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("time in one profiled epoch (train + val), s")
    ax.set_title(f"Baseline runtime breakdown, cProfile, total {d['total_s']:.0f} s")
    ax.set_xlim(0, s.max() * 1.35)
    ax.grid(axis="y", visible=False)
    _save(fig, "week1_fig1_baseline_breakdown.png")


def week1_loading():
    p = "results/week1/loading_breakdown.json"
    if not _need(p):
        return
    d = json.load(open(p))
    parts = ["read", "decode", "resize", "to_array"]
    sets = list(d)
    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.38
    for j, (k, col) in enumerate(zip(sets, (C1, C2))):
        vals = [d[k]["per_image_ms"][q] for q in parts]
        xs = [i + (j - 0.5) * width for i in range(len(parts))]
        ax.bar(xs, vals, width=width - 0.04, color=col, label=k)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(parts)), ["disk read", "JPEG decode", "resize", "to ndarray"])
    ax.set_ylabel("ms per image (1 thread)")
    ax.set_title("Module A: where loading time goes")
    ax.legend()
    ax.grid(axis="x", visible=False)
    _save(fig, "week1_loading_breakdown.png")


def week1_resources():
    p = "results/baseline/resources.csv"
    if not _need(p):
        return
    r = pd.read_csv(p)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
    a1.plot(r.t_s, r.cpu_percent, color=C1, lw=1.5)
    a1.set_ylabel("system CPU, %")
    a1.set_ylim(0, 100)
    a1.set_title("Baseline run (train_cpu.py): CPU utilization and memory")
    a2.plot(r.t_s, r.proc_tree_rss_gb, color=C1, lw=1.5, label="training process tree RSS")
    a2.plot(r.t_s, r.ram_used_gb, color=C2, lw=1.5, label="system RAM used")
    a2.set_ylabel("GB")
    a2.set_xlabel("time since start, s")
    a2.legend(loc="center right")
    _save(fig, "week1_resources.png")


def week2_sweep():
    p = "results/week2/sweep.csv"
    if not _need(p):
        return
    s = pd.read_csv(p)
    panels = [("clahe", ["clahe"]), ("median_filter", ["median_filter",
              "median_filter CuPy RawKernel (GPU variant)"]), ("preprocess_full",
              ["preprocess_full", "preprocess_full + H2D fused (GPU variant)",
               "preprocess_full CPU 8 threads"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, (title, ops) in zip(axes, panels):
        series = []
        for op in ops:
            for dev in ("cpu", "cpu-mt", "cuda"):
                g = s[(s.op == op) & (s.device == dev)]
                if len(g):
                    variant = op[len(title):].replace("(GPU variant)", "").replace("CPU ", "").strip()
                    lab = {"cpu": "CPU", "cpu-mt": "CPU", "cuda": "GPU"}[dev]
                    series.append((f"{lab} {variant}".strip() if variant else f"{lab} (reference)" if dev == "cpu" else lab, g))
        for (lab, g), col in zip(series, (C1, C2, C3, C4)):
            ax.plot(g.batch, g.per_image_us, marker="o", ms=4, lw=2, color=col, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 8, 32, 64, 128], ["1", "8", "32", "64", "128"])
        ax.set_xlabel("batch size")
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("time per image, µs (log)")
    _save(fig, "week2_batch_sweep.png")


def week2_speedup():
    p = "results/week2/sweep.csv"
    if not _need(p):
        return
    s = pd.read_csv(p)
    g = s[(s.batch == 128) & (s.device != "cpu")].dropna(subset=["speedup_vs_cpu"])
    g = g.sort_values("speedup_vs_cpu")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(g) + 1.2))
    cols = [C2 if d == "cuda" else C1 for d in g.device]
    ax.barh(g.op, g.speedup_vs_cpu, color=cols, height=0.6)
    for y, v in enumerate(g.speedup_vs_cpu):
        ax.text(v, y, f" {v:.1f}x", va="center", fontsize=8, color=INK2)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("speedup vs single-process CPU reference at batch 128 (log)")
    ax.set_title("Speedup at batch 128 (orange = GPU, blue = CPU 8 threads)", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, "week2_speedup_b128.png")


def week2_loading():
    p = "results/week2/loading.csv"
    if not _need(p):
        return
    l = pd.read_csv(p).sort_values("per_image_ms")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cols = [C3 if v == 1.0 else C2 for v in l.frac_bit_exact]
    ax.barh(l.variant, l.per_image_ms, color=cols, height=0.6)
    for y, (v, e) in enumerate(zip(l.per_image_ms, l.mean_abs_levels_vs_ref)):
        ax.text(v, y, f" {v:.3g} ms  (mean |diff| {e:.2f} levels)", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("ms per image (log)")
    ax.set_title("Module A alternatives (green = bit-exact with the reference)", loc="left")
    ax.set_xlim(right=l.per_image_ms.max() * 40)
    ax.grid(axis="y", visible=False)
    _save(fig, "week2_loading.png")


if __name__ == "__main__":
    for f in (week1_breakdown, week1_loading, week1_resources, week2_sweep, week2_speedup, week2_loading):
        f()

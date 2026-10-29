"""PyTorch Profiler on the optimized GPU training loop (assignment A7.2-A7.4, A7.7).

Goal: find out where wall time goes during steady-state training: GPU kernels,
host-to-device copies, or GPU idle time caused by the input pipeline.

    python benchmarks/profile_gpu.py                                   # configs/optimized.yaml
    python benchmarks/profile_gpu.py --config configs/exp_resident_off.yaml   # DataLoader path

Writes to results/week4/profile_<config>/ :
  summary.json   step time, GPU busy/idle, kernel/memcpy time and bytes, kernel categories,
                 synchronized phase times, epoch decomposition (train steps vs validation)
  top_ops.txt    top 30 operators by GPU time (torch.profiler key_averages)
  kernels.csv    every GPU kernel name with total time and count over the profiled steps
  timeline.png   two training steps: CPU phases and GPU activity on one time axis
  trace.json     Chrome trace (git-ignored, large): open it in https://ui.perfetto.dev

Three views, because each one hides something the others show:
  1. torch.profiler (asynchronous, as in training): GPU busy time = UNION of kernel/memcpy
     intervals (events on different streams overlap, so durations are not summed), idle
     fraction = 1 - busy / wall. Tells whether the GPU waits for the host.
  2. Synchronized phase timing (outside the profiler): cuda.synchronize between data,
     forward, backward and optimizer so each phase gets its own time. Breaks the overlap,
     so the sum is slightly larger than an unsynchronized step: used for attribution only.
  3. Epoch decomposition with src.train.run_epoch: train epoch vs validation epoch, to
     compare with steady_epoch_s in history.json.
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import yaml
import torch
import torch.nn as nn
from torch.profiler import profile, schedule, ProfilerActivity, record_function

from src.utils import set_seed, save_json, GpuTimer, gpu_warmup
from src.model import build_model
from src.train import run_epoch, make_gpu_pre, resolve_gpu_preprocess

PHASES = ("data", "forward", "backward", "optimizer")


# ----------------------------------------------------------------------------- setup
def build(cfg, device):
    """Same objects as src/train.py main(), for the same config (kept in sync by hand:
    every flag read here is read there with the same default)."""
    set_seed(cfg["seed"], cudnn_benchmark=bool(cfg["train"].get("cudnn_benchmark", False)))
    amp = bool(cfg["train"].get("amp", False))
    channels_last = bool(cfg["train"].get("channels_last", False))
    augment = bool(cfg["preprocess"].get("augment", False))
    gpu_pre_on = resolve_gpu_preprocess(cfg, device)
    prefetch_on = bool(cfg["data"].get("prefetch_gpu", False))
    resident = bool(cfg["data"].get("gpu_resident", False))
    precompute = resident and bool(cfg["preprocess"].get("precompute", False))
    if resident:
        gpu_pre_on, prefetch_on = True, False
    pre_tr = make_gpu_pre(cfg, device, augment=augment) if gpu_pre_on and not precompute else None
    pre_va = make_gpu_pre(cfg, device, augment=False) if gpu_pre_on and not precompute else None
    if resident:
        from src.gpu_resident import make_gpu_resident_loaders
        loaders, _ = make_gpu_resident_loaders(cfg, device, ("train", "val"), augment=augment,
                                               precompute=precompute, store_fp16=amp)
    else:
        from src.data_loader import make_loaders
        loaders, _ = make_loaders(cfg, "cuda", gpu_preprocess=gpu_pre_on, which=("train", "val"))
    if prefetch_on:
        from src.prefetcher import CudaPrefetcher
        loaders = {k: CudaPrefetcher(v, device) for k, v in loaders.items()}
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    net = (torch.compile(model, mode=cfg["train"].get("compile_mode", "default"))
           if cfg["train"].get("compile", False) else model)
    fused = bool(cfg["train"].get("fused_adamw", False))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 0.0),
                            **({"fused": True} if fused else {}))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    flags = {"amp": amp, "channels_last": channels_last, "gpu_preprocess": gpu_pre_on,
             "gpu_resident": resident, "precompute": precompute, "prefetch_gpu": prefetch_on,
             "fused_adamw": fused, "augment": augment,
             "cudnn_benchmark": torch.backends.cudnn.benchmark,
             "batch_size": cfg["data"]["batch_size"], "image_size": cfg["data"]["image_size"],
             "model": cfg["train"]["model"]}
    return dict(loaders=loaders, pre_tr=pre_tr, pre_va=pre_va, net=net, opt=opt, crit=crit,
                scaler=scaler, amp=amp, channels_last=channels_last, flags=flags)


def batches(loader):
    """Endless iterator over a loader (a new epoch starts when one ends)."""
    while True:
        for b in loader:
            yield b


def train_step(S, x, y, device, sync=False, times=None):
    """One training step, identical to the body of run_epoch (train=True).
    With sync=True, each phase is closed by cuda.synchronize and timed into `times`."""
    def mark(name, t0):
        if sync:
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times[name].append(t1 - t0)
            return t1
        return t0

    t = time.perf_counter()
    with record_function("h2d+preprocess"):
        y = y.to(device, non_blocking=True)
        x = S["pre_tr"](x) if S["pre_tr"] is not None else x.to(device, non_blocking=True)
        if S["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)
    t = mark("data", t)
    with record_function("forward"):
        with torch.autocast("cuda", enabled=S["amp"]):
            logits = S["net"](x)
            loss = S["crit"](logits, y)
    t = mark("forward", t)
    with record_function("backward"):
        S["opt"].zero_grad(set_to_none=True)
        S["scaler"].scale(loss).backward()
    t = mark("backward", t)
    with record_function("optimizer"):
        S["scaler"].step(S["opt"])
        S["scaler"].update()
    mark("optimizer", t)
    return loss.detach()


# ----------------------------------------------------------------------------- trace analysis
CATEGORIES = [  # (category, substrings of the kernel name, lower case); first match wins
    ("batch norm", ("batch_norm", "batchnorm", "bn_fw", "bn_bw", "welford")),
    ("layout (NCHW<->NHWC)", ("nchwtonhwc", "nhwctonchw", "transpose", "permute")),
    ("pooling", ("pool",)),
    ("AMP scaler", ("_amp_", "non_finite", "unscale", "amp_update_scale")),
    ("optimizer (AdamW)", ("adam", "multi_tensor", "foreach")),
    ("data (index/gather, preprocess)", ("index", "gather", "randperm", "scatter", "histc",
                                         "grid_sampler", "sort", "cumsum")),
    ("conv (cuDNN)", ("conv", "implicit_gemm", "xmma", "fprop", "dgrad", "wgrad", "winograd",
                      "cudnn", "precomputed_convolve", "scudnn", "hcudnn", "s1688", "h1688", "s884",
                      "h884", "cutlass")),
    ("gemm (fc)", ("gemm", "gemv", "cublas")),
    ("elementwise / activation", ("elementwise", "vectorized", "relu", "threshold", "add_kernel",
                                  "mul_kernel", "copy", "cast", "fill")),
    ("reduction / loss", ("reduce", "softmax", "nll", "log_softmax", "cross_entropy", "argmax",
                          "bincount")),
]


def category(name):
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def union_length(intervals):
    """Total length covered by a list of (start, end) intervals (overlaps counted once)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def analyze_trace(trace_path, n_steps):
    """Numbers of A7.3 from the exported Chrome trace (ts and dur in microseconds)."""
    ev = json.load(open(trace_path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ev = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    steps = sorted((e for e in ev if str(e.get("name", "")).startswith("ProfilerStep#")),
                   key=lambda e: e["ts"])
    for e in ev:                      # category names changed case across torch versions
        e["cat"] = str(e.get("cat", "")).lower()
    gpu = [e for e in ev if e["cat"] in ("kernel", "gpu_memcpy", "gpu_memset", "memcpy", "memset")]
    for e in gpu:
        e["cat"] = {"memcpy": "gpu_memcpy", "memset": "gpu_memset"}.get(e["cat"], e["cat"])
    if not steps or not gpu:
        from collections import Counter
        raise RuntimeError(f"trace has {len(steps)} ProfilerStep and {len(gpu)} GPU events; "
                           f"event categories: {Counter(e['cat'] for e in ev).most_common(12)}")
    t0 = steps[0]["ts"]
    t1 = max(steps[-1]["ts"] + steps[-1]["dur"], max(e["ts"] + e["dur"] for e in gpu))
    wall = t1 - t0
    gpu = [e for e in gpu if e["ts"] >= t0 - 1]
    kern = [e for e in gpu if e["cat"] == "kernel"]
    memcpy = [e for e in gpu if e["cat"] == "gpu_memcpy"]
    memset = [e for e in gpu if e["cat"] == "gpu_memset"]

    def kind(e):
        n = e.get("name", "")
        return "h2d" if "HtoD" in n else "d2h" if "DtoH" in n else "d2d"

    mc = defaultdict(lambda: {"ms": 0.0, "bytes": 0, "count": 0})
    for e in memcpy:
        k = mc[kind(e)]
        k["ms"] += e["dur"] / 1e3
        k["bytes"] += int(e.get("args", {}).get("bytes", 0) or 0)
        k["count"] += 1
    busy = union_length([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    by_cat, by_name = defaultdict(float), defaultdict(lambda: [0.0, 0])
    for e in kern:
        by_cat[category(e["name"])] += e["dur"]
        by_name[e["name"]][0] += e["dur"]
        by_name[e["name"]][1] += 1
    kern_total = sum(e["dur"] for e in kern)
    # CPU-side phase annotations (record_function), mean per step
    cpu_ph = defaultdict(float)
    for e in ev:
        if e.get("name") in ("h2d+preprocess", "forward", "backward", "optimizer") and \
                e.get("cat") in ("user_annotation", "cpu_op", None):
            cpu_ph[e["name"]] += e["dur"]
    # gaps: GPU idle intervals longer than 50 us inside the window (what the GPU waits for)
    ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    gaps, end = [], t0
    for s, e in ivs:
        if s - end > 50:
            gaps.append(s - end)
        end = max(end, e)
    summary = {
        "profiled_steps": n_steps,
        "wall_ms_per_step": wall / 1e3 / n_steps,
        "gpu_busy_ms_per_step": busy / 1e3 / n_steps,
        "gpu_idle_fraction": 1 - busy / wall,
        "kernel_ms_per_step": kern_total / 1e3 / n_steps,
        "kernels_per_step": len(kern) / n_steps,
        "mean_kernel_us": kern_total / max(len(kern), 1),
        "memcpy": {k: {"ms_per_step": v["ms"] / n_steps, "bytes_per_step": v["bytes"] / n_steps,
                       "count_per_step": v["count"] / n_steps} for k, v in mc.items()},
        "memset_ms_per_step": sum(e["dur"] for e in memset) / 1e3 / n_steps,
        "kernel_categories_ms_per_step": {k: v / 1e3 / n_steps for k, v in
                                          sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "kernel_categories_share": {k: v / kern_total for k, v in
                                    sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "cpu_annotation_ms_per_step": {k: v / 1e3 / n_steps for k, v in cpu_ph.items()},
        "gpu_gaps_over_50us": {"count_per_step": len(gaps) / n_steps,
                               "total_ms_per_step": sum(gaps) / 1e3 / n_steps,
                               "max_ms": max(gaps) / 1e3 if gaps else 0.0},
    }
    names = sorted(by_name.items(), key=lambda kv: -kv[1][0])
    return summary, names, ev, steps, gpu


def plot_timeline(path, ev, steps, gpu, n_show=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(steps) < n_show:
        n_show = len(steps)
    t0 = steps[0]["ts"]
    t1 = steps[n_show - 1]["ts"] + steps[n_show - 1]["dur"]
    t1 = max([t1] + [e["ts"] + e["dur"] for e in gpu if e["ts"] < t1])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    colors = {"h2d+preprocess": "#2a78d6", "forward": "#1baf7a", "backward": "#eb6834",
              "optimizer": "#eda100"}
    for e in ev:
        if e.get("name") in colors and t0 <= e["ts"] <= t1:
            ax.barh(2, e["dur"] / 1e3, left=(e["ts"] - t0) / 1e3, height=0.6,
                    color=colors[e["name"]])
    cat_colors = {}
    palette = ["#1baf7a", "#eb6834", "#2a78d6", "#eda100", "#8f5bd6", "#d63f74", "#52514e",
               "#4fb3c8", "#a3a19a", "#7a5c2e", "#0b0b0b"]
    for e in gpu:
        if t0 <= e["ts"] <= t1:
            c = ("memcpy/memset" if e.get("cat") != "kernel" else category(e["name"]))
            cat_colors.setdefault(c, palette[len(cat_colors) % len(palette)])
            ax.barh(1, max(e["dur"], 2) / 1e3, left=(e["ts"] - t0) / 1e3, height=0.6,
                    color=cat_colors[c])
    for s in steps[:n_show]:
        ax.axvline((s["ts"] - t0) / 1e3, color="#a3a19a", lw=0.8, ls="--")
    ax.set_yticks([1, 2], ["GPU (kernels, copies)", "CPU (phases)"])
    ax.set_xlabel("time (ms) from the start of the first profiled step")
    from matplotlib.patches import Patch
    handles = [Patch(color=v, label=k) for k, v in colors.items()]
    handles += [Patch(color=v, label=k) for k, v in cat_colors.items()]
    ax.legend(handles=handles, ncol=4, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.set_title(f"{n_show} training steps (dashed lines: step boundaries)")
    ax.set_ylim(0.4, 2.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimized.yaml")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--phase-steps", type=int, default=20, help="synchronized phase timing steps")
    ap.add_argument("--no-epochs", action="store_true", help="skip the epoch decomposition")
    ap.add_argument("--out", default=None, help="default results/week4/profile_<config>")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("profile_gpu needs a CUDA GPU (Colab: Runtime > Change runtime type > T4 GPU)")
    name = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join("results", "week4", f"profile_{name}")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda")
    cfg = yaml.safe_load(open(args.config))
    gpu_state = gpu_warmup(device)
    t_setup = time.perf_counter()
    S = build(cfg, device)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - t_setup
    it = batches(S["loaders"]["train"])
    steps_per_epoch = len(S["loaders"]["train"])
    print(f"[profile_gpu] {args.config}: {S['flags']}, {steps_per_epoch} steps/epoch, "
          f"setup {setup_s:.1f}s", flush=True)

    # 0) warm-up outside any measurement: cuDNN autotuning (cudnn_benchmark), allocator, compile
    for _ in range(max(args.warmup, 10)):
        x, y = next(it)
        train_step(S, x, y, device)
    torch.cuda.synchronize()

    # 1) unsynchronized step time (the reference for the other views)
    n = args.steps
    with GpuTimer(device) as t:
        for _ in range(n):
            x, y = next(it)
            train_step(S, x, y, device)
    step_ms = t.elapsed / n * 1e3

    # 2) torch.profiler, asynchronous like training
    trace = os.path.join(out, "trace.json")
    # repeat=1: ONE warmup+active cycle. With the default repeat=0 the schedule starts a new
    # cycle after the active steps and the events of the finished cycle are dropped unless they
    # are handled in on_trace_ready, so the trace is exported there, at the end of the cycle.
    got = {}

    def on_ready(p):
        p.export_chrome_trace(trace)
        try:
            got["table"] = p.key_averages().table(sort_by="self_device_time_total", row_limit=30)
        except Exception:
            got["table"] = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=args.warmup, active=args.steps, repeat=1),
                 on_trace_ready=on_ready, record_shapes=False) as prof:
        for _ in range(args.warmup + args.steps):
            x, y = next(it)
            train_step(S, x, y, device)
            prof.step()
    torch.cuda.synchronize()
    if "table" not in got:            # older torch: no callback fired, export what is recorded
        on_ready(prof)
    table = got["table"]
    open(os.path.join(out, "top_ops.txt"), "w").write(table)
    summ, names, ev, steps, gpu = analyze_trace(trace, args.steps)
    with open(os.path.join(out, "kernels.csv"), "w") as f:
        f.write("kernel,category,total_ms_per_step,count_per_step,mean_us\n")
        for k, (dur, cnt) in names:
            f.write(f"\"{k.replace(chr(34), chr(39))}\",{category(k)},{dur / 1e3 / args.steps:.4f},"
                    f"{cnt / args.steps:.2f},{dur / cnt:.2f}\n")
    plot_timeline(os.path.join(out, "timeline.png"), ev, steps, gpu)

    # 3) synchronized phase timing
    times = defaultdict(list)
    for _ in range(args.phase_steps):
        x, y = next(it)
        train_step(S, x, y, device, sync=True, times=times)
    phases = {k: 1e3 * sum(v) / len(v) for k, v in times.items()}

    # 4) epoch decomposition (one warm epoch first, then one measured epoch)
    epochs = None
    if not args.no_epochs:
        kw = dict(scaler=S["scaler"], amp=S["amp"], channels_last=S["channels_last"])
        for rep in range(2):
            with GpuTimer(device) as ttr:
                run_epoch(S["net"], S["loaders"]["train"], S["crit"], device, S["opt"],
                          gpu_pre=S["pre_tr"], **kw)
            conf = torch.zeros(5, 5, dtype=torch.long, device=device)
            with GpuTimer(device) as tva:
                run_epoch(S["net"], S["loaders"]["val"], S["crit"], device, conf=conf,
                          gpu_pre=S["pre_va"], amp=S["amp"], channels_last=S["channels_last"])
        epochs = {"train_epoch_s": ttr.elapsed, "val_epoch_s": tva.elapsed,
                  "epoch_s": ttr.elapsed + tva.elapsed,
                  "train_steps": steps_per_epoch,
                  "train_steps_x_step_time_s": steps_per_epoch * step_ms / 1e3,
                  "val_share": tva.elapsed / (ttr.elapsed + tva.elapsed)}

    summary = {"config": args.config, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "gpu_state_after_warmup": gpu_state,
               "flags": S["flags"], "setup_s": setup_s,
               "step_ms_unsynchronized": step_ms,
               "imgs_per_s_train_step": cfg["data"]["batch_size"] / step_ms * 1e3,
               "profiler": summ,
               "phases_ms_synchronized": phases,
               "phases_share_synchronized": {k: v / sum(phases.values()) for k, v in phases.items()},
               "epoch": epochs,
               "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    save_json(summary, os.path.join(out, "summary.json"))

    p = summ
    print(f"\n=== {args.config} on {summary['gpu']} ===")
    print(f"step: {step_ms:.1f} ms ({summary['imgs_per_s_train_step']:.0f} img/s), "
          f"profiled wall {p['wall_ms_per_step']:.1f} ms/step")
    print(f"GPU busy {p['gpu_busy_ms_per_step']:.1f} ms/step -> idle {100 * p['gpu_idle_fraction']:.1f} %"
          f" | {p['kernels_per_step']:.0f} kernels/step, mean {p['mean_kernel_us']:.0f} us")
    for k, v in p["memcpy"].items():
        print(f"memcpy {k}: {v['ms_per_step']:.3f} ms/step, {v['bytes_per_step'] / 2**20:.2f} MiB/step")
    print("GPU kernel time by category:")
    for k, v in p["kernel_categories_ms_per_step"].items():
        print(f"  {k:34s} {v:7.2f} ms/step  {100 * p['kernel_categories_share'][k]:5.1f} %")
    print("synchronized phases: " + ", ".join(f"{k} {v:.1f} ms" for k, v in phases.items()))
    if epochs:
        print(f"epoch: train {epochs['train_epoch_s']:.2f} s + val {epochs['val_epoch_s']:.2f} s = "
              f"{epochs['epoch_s']:.2f} s (val {100 * epochs['val_share']:.0f} %); "
              f"{steps_per_epoch} steps x {step_ms:.0f} ms = {epochs['train_steps_x_step_time_s']:.2f} s")
    print(f"-> {out}/summary.json, top_ops.txt, kernels.csv, timeline.png, trace.json")


if __name__ == "__main__":
    main()

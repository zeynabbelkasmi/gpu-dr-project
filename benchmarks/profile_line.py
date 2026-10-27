"""Week 1 (A2.7): two views that cProfile cannot give.

1. line_profiler on the per-image hot functions (DRDataset.load, preprocess_image,
   apply_clahe, circular_crop, normalize): time per *line*, so the cost of each
   conversion and intermediate array inside a function is visible.
2. torch.profiler on the CPU training step of the reference CNN: time and memory per
   ATen operator (conv forward, conv backward, max-pool, AdamW...), which cProfile only
   sees as one opaque `backward` call.

    python benchmarks/profile_line.py [--n 64]

Writes results/week1/line_profile.txt, results/week1/torch_profiler_cpu_ops.txt,
results/week1/torch_profiler_cpu_summary.json and results/week1/torch_profiler_cpu/trace.json.
"""
import os, sys, io, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml
from line_profiler import LineProfiler

from src import preprocess_cpu
from src.data_loader import DRDataset, get_splits
from src.model import build_model
from src.utils import set_seed

OUT = "results/week1"


def line_profile(cfg, items):
    ds = DRDataset(items, cfg["data"]["image_size"])
    lp = LineProfiler()
    for fn in (DRDataset.load, preprocess_cpu.preprocess_image, preprocess_cpu.apply_clahe,
               preprocess_cpu.circular_crop, preprocess_cpu.normalize):
        lp.add_function(fn)
    lp.enable_by_count()
    for i in range(len(items)):
        x = ds.load(i).astype(np.float32) / 255.0
        preprocess_cpu.preprocess_image(x, cfg)
    lp.disable_by_count()
    s = io.StringIO()
    lp.print_stats(stream=s, output_unit=1e-3)
    with open(f"{OUT}/line_profile.txt", "w") as f:
        f.write(s.getvalue())
    print(s.getvalue())


def torch_profile(cfg, items, steps=6, warmup=2):
    ds = DRDataset(items, cfg["data"]["image_size"],
                   transform=lambda im: preprocess_cpu.preprocess_image(im, cfg))
    bs = cfg["data"]["batch_size"]
    x = torch.stack([ds[i][0] for i in range(bs)])
    y = torch.tensor([ds.items[i][1] for i in range(bs)])
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    from torch.profiler import profile, ProfilerActivity, record_function

    def step():
        with record_function("forward"):
            loss = crit(model(x), y)
        with record_function("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with record_function("optimizer_step"):
            opt.step()

    for _ in range(warmup):                      # thread pool, allocator, AdamW state
        step()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        for _ in range(steps):
            step()
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    with open(f"{OUT}/torch_profiler_cpu_ops.txt", "w") as f:
        f.write(table)
    print(table)
    ev = {e.key: e for e in prof.key_averages()}
    us = lambda k: ev[k].cpu_time_total / steps if k in ev else 0.0
    step_us = us("forward") + us("backward") + us("optimizer_step")
    summary = {"batch_size": bs, "steps": steps, "torch_threads": torch.get_num_threads(),
               "per_step_ms": {k: us(k) / 1e3 for k in ("forward", "backward", "optimizer_step")},
               "share_of_step": {k: us(k) / step_us for k in ("forward", "backward", "optimizer_step")},
               "imgs_per_s_compute_only": bs / (step_us / 1e6)}
    top = sorted(ev.values(), key=lambda e: -e.self_cpu_time_total)[:8]
    summary["top_self_ops_ms_per_step"] = {e.key: e.self_cpu_time_total / steps / 1e3 for e in top}
    with open(f"{OUT}/torch_profiler_cpu_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    os.makedirs(f"{OUT}/torch_profiler_cpu", exist_ok=True)
    prof.export_chrome_trace(f"{OUT}/torch_profiler_cpu/trace.json")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    os.makedirs(OUT, exist_ok=True)
    items = get_splits(cfg)["train"][:args.n]
    line_profile(cfg, items)
    torch_profile(cfg, items)


if __name__ == "__main__":
    main()

"""cProfile the CPU baseline: one training epoch + validation, in-process.

num_workers is forced to 0: with worker processes, decode/CLAHE run in
children that cProfile cannot see and the profile would only show queue
waits. Writes:
  results/profile_baseline.txt   top-40 functions by cumulative time
  results/profile_stages.json    time attributed to each pipeline stage
                                 (Figure 1 and the Amdahl fraction p)
"""
import os, sys, cProfile, pstats, io, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from src.train import main as train_main

# stage -> (function name, file substring); cumulative time of each
STAGES = {
    "loading (decode+resize)": ("load", "data_loader.py"),
    "preprocessing (total)": ("preprocess_image", "preprocess_cpu.py"),
    "  clahe": ("apply_clahe", "preprocess_cpu.py"),
    "  circular_crop": ("circular_crop", "preprocess_cpu.py"),
    "  normalize": ("normalize", "preprocess_cpu.py"),
    "forward": ("_call_impl", "module.py"),
    "backward": ("backward", "_tensor.py"),
    "optimizer_step": ("step", "torch/optim/"),
}


def stage_times(stats):
    out = {}
    for stage, (fn, fsub) in STAGES.items():
        best = 0.0
        for (fname, _, func), (_, _, _, ct, callers) in stats.stats.items():
            if func == fn and fsub in fname.replace("\\", "/"):
                # top-level calls only for recursive wrappers (nn.Module._call_impl)
                if fn == "_call_impl":
                    ct = sum(v[3] for k, v in callers.items() if k[2] != "_call_impl"
                             and k[2] != "_wrapped_call_impl") or ct
                best = max(best, ct)
        out[stage] = best
    return out


def main():
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["data"]["num_workers"] = 0
    cfg["train"]["epochs"] = 1
    os.makedirs("results", exist_ok=True)
    tmp = "results/_profile_cfg.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)

    pr = cProfile.Profile()
    pr.enable()
    train_main(tmp, "cpu", run="profile_baseline")
    pr.disable()
    os.remove(tmp)

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("cumulative").print_stats(40)
    with open("results/profile_baseline.txt", "w") as f:
        f.write(s.getvalue())

    total = max(v[3] for v in st.stats.values())
    stages = stage_times(st)
    top = ["loading (decode+resize)", "preprocessing (total)", "forward", "backward",
           "optimizer_step"]
    stages["other"] = max(total - sum(stages[k] for k in top), 0.0)
    out = {"total_s": total, "stages_s": stages,
           "fractions": {k: v / total for k, v in stages.items()}}
    with open("results/profile_stages.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in stages.items():
        print(f"{k:28s} {v:8.2f}s  {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()

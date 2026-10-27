"""PyTorch Profiler on the optimized GPU training loop (assignment A7).

Goal: find out where wall time goes during steady-state training: GPU kernels,
host-to-device copies, or GPU idle time caused by the input pipeline.
Writes results/torch_profiler/{summary.json, top_ops.txt, trace.json}.
trace.json opens in https://ui.perfetto.dev or chrome://tracing.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimized.yaml")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--out", default="results/torch_profiler")
    args = ap.parse_args()

    # TODO(A7.2): build loaders/model/optimizer from the config exactly like src/train.py does
    #   (reuse make_loaders and make_gpu_pre), then run `warmup + steps` training steps
    #   under torch.profiler.profile(activities=[CPU, CUDA], schedule=schedule(wait=0,
    #   warmup=args.warmup, active=args.steps)). Call prof.step() once per training step.
    # TODO(A7.3): from prof.events() collect the CUDA events and compute, over the profiled steps:
    #   - GPU kernel time, host-to-device memcpy time and bytes, device-to-host memcpy time
    #   - GPU busy time = length of the UNION of those intervals. Do not just add durations:
    #     events on different streams overlap and summing them over-counts.
    #   - GPU idle fraction = 1 - busy / wall time
    #   Save these numbers in summary.json, the top 30 ops by CUDA time in top_ops.txt,
    #   and export the Chrome trace.
    # TODO(A7.4): open trace.json and describe in the report what the GPU does between two
    #   training steps. Is the GPU waiting for data? Which ablation would confirm it?
    raise NotImplementedError("TODO(A7.2-A7.4): profile_gpu")


if __name__ == "__main__":
    main()

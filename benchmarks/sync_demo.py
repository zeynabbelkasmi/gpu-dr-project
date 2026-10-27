"""A2.1: why GPU timings need torch.cuda.synchronize().

Times the same large matmul (a) with perf_counter only and (b) with GpuTimer, which
synchronizes before and after. (a) only measures the kernel *launch*. Writes
results/sync_demo.json.

    python benchmarks/sync_demo.py
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from src.utils import GpuTimer


def main(n=8192, repeats=10):
    dev = torch.device("cuda")
    a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
    for _ in range(3):
        a @ b                                       # warm-up: context, cuBLAS handle, kernel load
    torch.cuda.synchronize()
    no_sync, with_sync = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = a @ b
        no_sync.append(time.perf_counter() - t0)    # the call returns as soon as the kernel is queued
        torch.cuda.synchronize()
        with GpuTimer(dev) as t:
            c = a @ b
        with_sync.append(t.elapsed)
    flops = 2 * n ** 3
    out = {"n": n, "repeats": repeats,
           "no_sync_ms": 1e3 * sum(no_sync) / repeats,
           "with_sync_ms": 1e3 * sum(with_sync) / repeats,
           "tflops_no_sync (wrong)": flops / (sum(no_sync) / repeats) / 1e12,
           "tflops_with_sync": flops / (sum(with_sync) / repeats) / 1e12}
    os.makedirs("results", exist_ok=True)
    with open("results/sync_demo.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

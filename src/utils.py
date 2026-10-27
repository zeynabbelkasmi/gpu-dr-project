"""Shared utilities: seeding, timing, JSON helpers."""
import os, random, time, json
import numpy as np
import torch


def set_seed(seed: int, cudnn_benchmark: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # A1.5 decision. Baseline and accuracy runs: benchmark=False, so the convolution
    # algorithm (and therefore the floating-point summation order) is the same on every
    # run and two seeds differ only by their seed. Throughput benchmarks may pass
    # cudnn_benchmark=True: input shapes are fixed (224x224, drop_last=True on train), so
    # the one-off autotuning cost falls in the discarded warm-up and steady-state epochs
    # use the fastest kernel. The flag has no effect on the CPU baseline.
    torch.backends.cudnn.benchmark = cudnn_benchmark


class GpuTimer:
    """Wall-clock timer. Use as: `with GpuTimer(device) as t: ...` then `t.elapsed`.

    CUDA kernels are asynchronous, so perf_counter alone only measures the time to
    launch them. On a CUDA device the timer synchronizes before reading the clock in
    __enter__ (work queued earlier is not billed to this region) and in __exit__
    (work queued inside the region is finished). See benchmarks/sync_demo.py (A2.1).
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.elapsed = 0.0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.elapsed = time.perf_counter() - self._t0
        return False


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def gpu_state():
    """PCIe link generation/width, SM clock and P-state from nvidia-smi (None if unavailable)."""
    import subprocess
    q = "pcie.link.gen.current,pcie.link.width.current,clocks.sm,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        gen, width, sm, pstate = (v.strip() for v in out.split(","))
        return {"pcie_gen": int(gen), "pcie_width": int(width), "sm_clock_mhz": int(sm), "pstate": pstate}
    except Exception:
        return None


def gpu_warmup(device="cuda", seconds=3.0):
    """Sustained load before GPU benchmarks.

    A laptop GPU idles in P8 with the PCIe link trained down to Gen1 and the SM clock at a
    few hundred MHz; the first seconds of any benchmark then measure the ramp-up, not the
    workload (measured: 3.3 GB/s host-to-device before warm-up vs 13 GB/s after). Keeping
    copies and matmuls in flight for a few seconds brings link and clocks to steady state.
    """
    dev = torch.device(device)
    a = torch.randn(4096, 4096, device=dev)
    host = torch.empty(64 * 2**20, dtype=torch.uint8).pin_memory()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        host.to(dev, non_blocking=True)
        a = a @ a
        a /= a.norm()
    torch.cuda.synchronize(dev)
    return gpu_state()

"""Background CPU/GPU utilization + memory sampler (Figure 5).

Cross-platform replacement for `top -b` / `nvidia-smi dmon`: psutil for the
CPU side, `nvidia-smi --query-gpu` for the GPU side, sampled at 1 Hz.

    with ResourceMonitor("results/optimized/resources.csv"):
        train(...)
"""
import csv, os, shutil, subprocess, threading, time

import psutil

_GPU_FIELDS = "utilization.gpu,utilization.memory,memory.used"


def _gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        util, mem_util, mem_used = (float(v) for v in out.split(","))
        return util, mem_util, mem_used
    except Exception:
        return None, None, None


class ResourceMonitor:
    def __init__(self, path, interval=1.0):
        self.path = path
        self.interval = interval
        self.has_gpu = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        proc = psutil.Process()
        psutil.cpu_percent(None)
        t0 = time.perf_counter()
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "cpu_percent", "ram_used_gb", "proc_tree_rss_gb",
                        "gpu_util_percent", "gpu_mem_util_percent", "gpu_mem_used_mb"])
            while not self._stop.wait(self.interval):
                try:
                    rss = proc.memory_info().rss + sum(
                        c.memory_info().rss for c in proc.children(recursive=True))
                except psutil.Error:
                    rss = 0
                gpu = _gpu_sample() if self.has_gpu else (None, None, None)
                w.writerow([round(time.perf_counter() - t0, 2), psutil.cpu_percent(None),
                            round(psutil.virtual_memory().used / 1e9, 3), round(rss / 1e9, 3),
                            *gpu])
                f.flush()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        return False

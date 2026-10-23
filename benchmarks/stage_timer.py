"""Petit outil pour mesurer le temps de chaque etape du pipeline.
Utilisation :
    timer = StageTimer()
    with timer.stage("loading"):
        ...
    timer.save("results/timing.csv")
"""
import csv, time
from contextlib import contextmanager

class StageTimer:
    def __init__(self, cuda_sync=False):
        self.records = {}          # stage -> liste de durees
        self.cuda_sync = cuda_sync # mettre True des qu'il y a du GPU

    def _sync(self):
        if self.cuda_sync:
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        self.records.setdefault(name, []).append(time.perf_counter() - t0)

    def summary(self):
        total = sum(sum(v) for v in self.records.values()) or 1e-9
        print(f"{'Stage':<22}{'Total (s)':>12}{'Calls':>8}{'% total':>10}")
        for k, v in sorted(self.records.items(), key=lambda kv: -sum(kv[1])):
            s = sum(v)
            print(f"{k:<22}{s:>12.3f}{len(v):>8}{100*s/total:>9.1f}%")
        print(f"{'TOTAL':<22}{total:>12.3f}")

    def save(self, path):
        total = sum(sum(v) for v in self.records.values()) or 1e-9
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "total_s", "calls", "mean_s", "percent_of_total"])
            for k, v in self.records.items():
                s = sum(v)
                w.writerow([k, f"{s:.4f}", len(v), f"{s/len(v):.4f}", f"{100*s/total:.1f}"])

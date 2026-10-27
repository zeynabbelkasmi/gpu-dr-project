"""Show which assignment steps still have a TODO(A<n>.<m>) tag in the code.

    python scripts/progress.py

When you finish a step, delete its TODO tag from the code (keep the comment text if it
helps readers). Steps with no file of their own are tagged in docs/checklist.md.
"""
import os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = ["src", "benchmarks", "scripts", "configs", "environment", "docs", "report", "README.md"]
TAG = re.compile(r"TODO\(A(\d+)\.(\d+)(?:-A?\d+\.?\d*)?\)")
TITLES = {
    1: "Baseline execution and verification",
    2: "Profiling and bottleneck identification",
    3: "Loading redesign: decoded-image cache",
    4: "GPU image processing (preprocessing and filters)",
    5: "End-to-end input pipeline",
    6: "Training acceleration and model redesign",
    7: "Final benchmarking and GPU profiling",
    8: "Figures, tables and report",
}


OPTIONAL = {"A4.11", "A6.3"}


def main():
    found = defaultdict(lambda: defaultdict(list))      # assignment -> tag -> locations
    for d in SCAN:
        top = os.path.join(ROOT, d)
        walk = [(ROOT, [], [d])] if os.path.isfile(top) else os.walk(top)
        for dirpath, _, files in walk:
            for f in files:
                if not f.endswith((".py", ".yaml", ".yml", ".md")):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8") as fh:
                    for no, line in enumerate(fh, 1):
                        for m in TAG.finditer(line):
                            loc = f"{os.path.relpath(path, ROOT)}:{no}"
                            locs = found[int(m.group(1))][f"A{m.group(1)}.{m.group(2)}"]
                            if loc not in locs:
                                locs.append(loc)
    total = 0
    for a in sorted(TITLES):
        tags = sorted(found.get(a, {}), key=lambda t: int(t.split(".")[1]))
        required = [t for t in tags if t not in OPTIONAL]
        total += len(required)
        status = "done" if not tags else f"{len(required)} TODO left"
        print(f"A{a}  {TITLES[a]:50s} {status}")
        for t in tags:
            note = "  (optional)" if t in OPTIONAL else ""
            print(f"      {t:6s} {', '.join(found[a][t][:3])}{note}")
    print(f"\n{total} required steps left (a step is done when its tag is gone from every file)."
          "\nRun `python -m pytest tests -v` for the automatic checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

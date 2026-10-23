"""Genere automatiquement environment/hardware_software.md (Table 1 du rapport).
Usage : python environment/collect_env.py
"""
import platform, sys, os
from pathlib import Path

lines = ["# Hardware & Software configuration", ""]
lines.append(f"- OS: {platform.platform()}")
lines.append(f"- Python: {sys.version.split()[0]}")
lines.append(f"- CPU: {platform.processor() or 'n/a'} ({os.cpu_count()} logical cores)")

try:
    import psutil
    lines.append(f"- RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
except ImportError:
    lines.append("- RAM: (pip install psutil pour l'obtenir)")

for name in ["numpy", "cv2", "torch", "torchvision", "sklearn", "cupy", "numba"]:
    try:
        mod = __import__(name)
        lines.append(f"- {name}: {getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        lines.append(f"- {name}: not installed")

try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines.append(f"- GPU: {p.name} ({p.total_memory/1e9:.1f} GB)")
        lines.append(f"- CUDA (torch): {torch.version.cuda}")
    else:
        lines.append("- GPU: none available (CPU only)")
except ImportError:
    pass

out = Path(__file__).parent / "hardware_software.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nEcrit dans {out}")

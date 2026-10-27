"""Record hardware/software configuration (Table 1) -> results/environment.json."""
import os, sys, json, platform, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psutil
import torch, torchvision, numpy, cv2, sklearn, PIL


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor()


def gpu_info():
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        drv = "nvidia-smi not found"
    return {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
            "compute_capability": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
            "driver": drv}


def main(out="results/environment.json"):
    env = {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9),
        "gpu": gpu_info(),
        "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "opencv": cv2.__version__, "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__, "torch_cpu_threads": torch.get_num_threads(),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()

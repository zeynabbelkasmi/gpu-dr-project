"""Why the frozen configuration is 3x slower on the RTX 4070 under Windows (portability check).

The same training step as configs/optimized.yaml (ResNet-18, batch 64, 320x320, fused AdamW,
cudnn.benchmark) on synthetic data already on the GPU, so only the model's kernels are timed:

  1. step throughput for AMP on/off x channels_last on/off (3 repeats of 40 steps, synchronized);
  2. one BatchNorm2d layer alone, fp16, NHWC vs NCHW, through cuDNN vs the native PyTorch kernel;
  3. the batch-norm kernels cuDNN actually launches with channels_last (PyTorch Profiler).

Run it on each platform with the same GPU, e.g. Windows and WSL2 (Linux) on the same laptop:

    python benchmarks/bench_channels_last.py --out results/rtx4070/channels_last_windows.json
    python benchmarks/bench_channels_last.py --out results/rtx4070/channels_last_wsl.json
"""
import os, sys, argparse, platform, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from src.model import build_model
from src.utils import GpuTimer, save_json, gpu_warmup, gpu_state

BATCH, SIZE = 64, 320


def make_step(amp, channels_last, dev):
    torch.manual_seed(0)
    m = build_model("resnet18", 5, pretrained=False).to(dev)
    x = torch.randn(BATCH, 3, SIZE, SIZE, device=dev)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 5, (BATCH,), device=dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss()

    def step():
        with torch.autocast("cuda", enabled=amp):
            loss = crit(m(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return step


def time_steps(step, dev, steps=40, repeats=3):
    for _ in range(10):                             # cuDNN autotuning, allocator
        step()
    ms = []
    for _ in range(repeats):
        with GpuTimer(dev) as t:
            for _ in range(steps):
                step()
        ms.append(t.elapsed / steps * 1000)
    return ms


def bn_ms(cudnn, channels_last, dev, c=64, hw=160, iters=20, repeats=3):
    """Forward + backward of one BatchNorm2d on a fp16 (64, c, hw, hw) activation."""
    torch.backends.cudnn.enabled = cudnn
    bn = nn.BatchNorm2d(c).to(dev)
    t = torch.randn(BATCH, c, hw, hw, device=dev, dtype=torch.half)
    if channels_last:
        t = t.contiguous(memory_format=torch.channels_last)
    t.requires_grad_(True)
    for _ in range(5):
        bn(t).float().sum().backward()
    ms = []
    for _ in range(repeats):
        with GpuTimer(dev) as tm:
            for _ in range(iters):
                bn(t).float().sum().backward()
        ms.append(tm.elapsed / iters * 1000)
    torch.backends.cudnn.enabled = True
    return ms


def bn_kernels(dev, steps=5):
    step = make_step(True, True, dev)
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_time_total > 0 and not e.key.startswith("aten::")]
    total = sum(e.device_time_total for e in ev)
    bn = sorted((e for e in ev if "bn_" in e.key or "batchnorm" in e.key.lower()),
                key=lambda e: -e.device_time_total)
    return {"step_gpu_ms": total / steps / 1000,
            "batch_norm_ms": sum(e.device_time_total for e in bn) / steps / 1000,
            "kernels": [{"name": e.key[:160], "ms_per_step": e.device_time_total / steps / 1000} for e in bn]}


def stats(v):
    return {"mean": statistics.mean(v), "std": statistics.stdev(v), "runs": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    gpu_warmup(dev)
    res = {"platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0), "gpu_state": gpu_state(),
           "batch": BATCH, "image_size": SIZE, "step": {}, "batch_norm_alone": {}}
    for amp in (True, False):
        for cl in (True, False):
            ms = time_steps(make_step(amp, cl, dev), dev)
            k = f"amp_{'on' if amp else 'off'}_channels_last_{'on' if cl else 'off'}"
            res["step"][k] = {"ms_per_step": stats(ms), "imgs_per_s": BATCH * 1000 / statistics.mean(ms)}
            print(f"{k:34s} {statistics.mean(ms):7.1f} ms/step  {res['step'][k]['imgs_per_s']:5.0f} img/s", flush=True)
    for cl in (True, False):
        for cudnn in (True, False):
            k = f"{'nhwc' if cl else 'nchw'}_{'cudnn' if cudnn else 'native'}"
            res["batch_norm_alone"][k] = stats(bn_ms(cudnn, cl, dev))
            print(f"BatchNorm fp16 {k:13s} {res['batch_norm_alone'][k]['mean']:7.2f} ms fwd+bwd", flush=True)
    s = res["step"]
    res["channels_last_speedup_amp"] = (s["amp_on_channels_last_off"]["ms_per_step"]["mean"]
                                        / s["amp_on_channels_last_on"]["ms_per_step"]["mean"])
    res["bn_kernels_channels_last_amp"] = bn_kernels(dev)
    print(f"channels_last speedup (AMP): {res['channels_last_speedup_amp']:.2f}x; "
          f"batch norm {res['bn_kernels_channels_last_amp']['batch_norm_ms']:.1f} ms of "
          f"{res['bn_kernels_channels_last_amp']['step_gpu_ms']:.1f} ms GPU time per step")
    save_json(res, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

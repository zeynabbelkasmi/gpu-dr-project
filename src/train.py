"""Training loop. The CPU reference works as is; assignments A5 and A6 add the GPU path.

Outputs are per run name so configurations never overwrite each other:
    results/<run>/history.json, checkpoints/<run>/best.pt
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import torch.nn as nn

from src.utils import set_seed, GpuTimer, save_json
from src.data_loader import make_loaders
from src.model import build_model


def macro_f1(conf):
    """Macro-F1 from a (C, C) confusion matrix (rows: true class, columns: predicted)."""
    tp = conf.diag().float()
    fp, fn = conf.sum(0) - conf.diag(), conf.sum(1) - conf.diag()
    return (2 * tp / (2 * tp + fp + fn).clamp(min=1)).mean().item()


def run_epoch(model, loader, criterion, device, optimizer=None, conf=None):
    """One pass over `loader`. Returns (mean loss, accuracy, n_images).
    If `conf` (a (C, C) long tensor on `device`) is given, it is updated in place with the
    confusion matrix, so the caller can compute macro-F1 with `macro_f1`.

    TODO(A6.1): add mixed precision. New parameters: scaler, amp.
        Wrap the forward pass and the loss in torch.autocast(device.type, enabled=amp)
        and replace backward/step with the GradScaler pattern
        (scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()).
    TODO(A6.2): add `channels_last` (convert the input batch with
        x.contiguous(memory_format=torch.channels_last)).
    TODO(A5.3): add `gpu_pre`, a callable(uint8 batch) -> preprocessed NCHW tensor on
        `device`. When it is given, use it instead of x.to(device).
    TODO(A5.4): make host-device copies asynchronous: x.to(device, non_blocking=True).
        Then explain in the report why this only helps with pinned memory.
    A2.2: the loss and accuracy are accumulated on the device so that the host waits
        once per epoch (the .item() calls in the return statement); see the comment in the loop.
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            x = x.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # A2.2: this stays a device tensor. `loss_sum += loss.item() * x.size(0)` would copy
            # the scalar to the host, i.e. block the CPU until every queued kernel of this
            # step has finished, once per batch, and the loader could no longer run ahead.
            loss_sum += loss.detach().float() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum()
            if conf is not None:
                conf += torch.bincount(y * conf.size(0) + pred,
                                       minlength=conf.numel()).view_as(conf)
            total += x.size(0)
    return loss_sum.item() / max(total, 1), correct.item() / max(total, 1), total


def resolve_gpu_preprocess(cfg, device, override=None):
    """TODO(A5.3): return True when GPU preprocessing should be used: the config flag
    cfg["preprocess"]["gpu"] (or `override` when it is not None) AND device.type == "cuda".
    """
    return False


def make_gpu_pre(cfg, device, augment):
    """TODO(A5.3): return a callable(batch) -> preprocessed tensor on `device`, built on
    src.preprocess_gpu.preprocess_batch_gpu with the given `augment` flag.
    (A lambda is fine here: it is used in the main process, not in DataLoader workers.)
    """
    raise NotImplementedError("TODO(A5.3): make_gpu_pre")


def main(cfg_path, device_str, run=None, gpu_preprocess=None, epochs=None,
         monitor=False, out_root="results", ckpt_root="checkpoints"):
    t_start = time.perf_counter()
    cfg = yaml.safe_load(open(cfg_path))
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    run = run or os.path.splitext(os.path.basename(cfg_path))[0]
    set_seed(cfg["seed"])
    device = torch.device(device_str)
    gpu_pre_on = resolve_gpu_preprocess(cfg, device, gpu_preprocess)
    augment = bool(cfg["preprocess"].get("augment", False))
    # TODO(A6.1): amp = cfg["train"].get("amp", False) and device.type == "cuda"; create a
    #             torch.amp.GradScaler("cuda", enabled=amp) and pass both to run_epoch.
    # TODO(A6.2): read cfg["train"].get("channels_last", False); convert the model with
    #             model.to(memory_format=torch.channels_last) and pass the flag to run_epoch.
    # TODO(A6.3, optional): read cfg["train"].get("compile", False); when true on CUDA, train through
    #             torch.compile(model) but save the state_dict of the ORIGINAL model so
    #             checkpoint keys stay clean. Record the cold-epoch cost in the report.
    #             torch.compile needs Triton (Linux). On Windows skip it and say so in the report.
    # TODO(A5.3): when gpu_pre_on, build TWO callables: make_gpu_pre(cfg, device, augment=augment)
    #             for the train loop and make_gpu_pre(cfg, device, augment=False) for validation.
    # TODO(A5.5): read cfg["data"].get("prefetch_gpu", False); when true on CUDA, wrap every loader
    #             in src.prefetcher.CudaPrefetcher(loader, device).

    loaders, splits = make_loaders(cfg, device.type, gpu_preprocess=gpu_pre_on, which=("train", "val"))
    split_sizes = {k: len(v) for k, v in splits.items()}
    print(f"[train:{run}] device={device} gpu_preprocess={gpu_pre_on} "
          f"augment={augment} splits={split_sizes}", flush=True)

    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 0.0))
    sched = None
    if cfg["train"].get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))

    ckpt_dir, out_dir = os.path.join(ckpt_root, run), os.path.join(out_root, run)
    os.makedirs(ckpt_dir, exist_ok=True)
    setup_s = time.perf_counter() - t_start

    mon = None
    if monitor:
        from benchmarks.resource_monitor import ResourceMonitor
        mon = ResourceMonitor(os.path.join(out_dir, "resources.csv")).__enter__()
    history, best_val = [], -1.0
    try:
        for epoch in range(cfg["train"]["epochs"]):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # A5.3, A6.1, A6.2: pass gpu_pre, scaler, amp and channels_last to both run_epoch calls
            with GpuTimer(device) as ttr:
                tr_loss, tr_acc, n_tr = run_epoch(model, loaders["train"], crit, device, opt)
            conf = torch.zeros(cfg["data"]["num_classes"], cfg["data"]["num_classes"],
                               dtype=torch.long, device=device)
            with GpuTimer(device) as tva:
                va_loss, va_acc, n_va = run_epoch(model, loaders["val"], crit, device, conf=conf)
            va_f1 = macro_f1(conf)
            if sched is not None:
                sched.step()
            peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20
                       if device.type == "cuda" else None)
            history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
                            "train_time_s": ttr.elapsed, "val_time_s": tva.elapsed,
                            "train_imgs_per_s": n_tr / ttr.elapsed,
                            "val_imgs_per_s": n_va / tva.elapsed,
                            "gpu_peak_mem_mb": peak_mb})
            print(f"epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"val_acc={va_acc:.4f} val_f1={va_f1:.4f} train_time={ttr.elapsed:.2f}s "
                  f"({n_tr / ttr.elapsed:.0f} img/s)", flush=True)
            if va_acc > best_val:          # model selection on VAL only, never on test
                best_val = va_acc
                torch.save({"model": model.state_dict(), "cfg": cfg, "val_acc": va_acc,
                            "epoch": epoch}, os.path.join(ckpt_dir, "best.pt"))
    finally:
        if mon is not None:
            mon.__exit__(None, None, None)

    wall_s = time.perf_counter() - t_start
    epoch_times = [h["train_time_s"] + h["val_time_s"] for h in history]
    steady = epoch_times[1:] or epoch_times
    save_json({"run": run, "config": cfg_path, "cfg": cfg, "device": device_str,
               "gpu_preprocess": gpu_pre_on, "augment": augment, "splits": split_sizes,
               "setup_time_s": setup_s, "wall_time_s": wall_s,
               "cold_epoch_s": epoch_times[0], "steady_epoch_s": sum(steady) / len(steady),
               "best_val_acc": best_val, "history": history},
              os.path.join(out_dir, "history.json"))
    print(f"[train:{run}] best val_acc={best_val:.4f}; wall={wall_s:.1f}s; "
          f"history -> {out_dir}/history.json", flush=True)
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run", default=None, help="output name (default: config basename)")
    ap.add_argument("--gpu-preprocess", dest="gpu_preprocess", action="store_true", default=None,
                    help="force GPU preprocessing (default: preprocess.gpu from config)")
    ap.add_argument("--cpu-preprocess", dest="gpu_preprocess", action="store_false")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--monitor", action="store_true", help="log CPU/GPU utilization")
    args = ap.parse_args()
    main(args.config, args.device, args.run, args.gpu_preprocess, args.epochs, args.monitor)

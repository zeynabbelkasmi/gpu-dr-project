"""Final test-set evaluation: accuracy, macro P/R/F1, per-class, confusion matrix,
and inference throughput. Reads checkpoints/<run>/best.pt, writes results/<run>/metrics.json.

The test set is used here and nowhere else: never select hyperparameters with it.
`--split val` computes the same report on the validation split (results/<run>/metrics_val.json)
so the baseline's classification quality can be checked in week 1 without touching test.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from src.data_loader import make_loaders
from src.model import build_model
from src.train import resolve_gpu_preprocess, make_gpu_pre
from src.utils import GpuTimer, save_json


def main(cfg_path, run=None, device_str="cpu", gpu_preprocess=None, checkpoint=None,
         out_root="results", split="test"):
    cfg = yaml.safe_load(open(cfg_path))
    run = run or os.path.splitext(os.path.basename(cfg_path))[0]
    checkpoint = checkpoint or os.path.join("checkpoints", run, "best.pt")
    device = torch.device(device_str)
    gpu_pre_on = resolve_gpu_preprocess(cfg, device, gpu_preprocess)
    loaders, splits = make_loaders(cfg, device.type, gpu_preprocess=gpu_pre_on, which=(split,))
    pre = make_gpu_pre(cfg, device, augment=False) if gpu_pre_on else None

    ckpt = torch.load(checkpoint, map_location=device)
    if ckpt["cfg"]["train"]["model"] != cfg["train"]["model"]:
        raise ValueError(f"checkpoint model {ckpt['cfg']['train']['model']} != "
                         f"config model {cfg['train']['model']}")
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # A6.4: same precision and memory format as training, so the test metrics describe the
    # model as it was trained (an AMP-trained model evaluated in fp32, or the reverse, can give
    # slightly different predictions). Inference time is reported separately from training
    # time: no backward, no optimizer step and no augmentation, so it is not an epoch time.
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    channels_last = bool(cfg["train"].get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    ys, ps = [], []
    with torch.no_grad(), GpuTimer(device) as t:
        for x, y in loaders[split]:
            x = pre(x) if pre is not None else x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
            ys.extend(y.tolist())
            ps.extend(logits.argmax(1).cpu().tolist())
    labels = list(range(cfg["data"]["num_classes"]))
    metrics = {
        "run": run, "config": cfg_path, "checkpoint": checkpoint, "device": device_str,
        "split": split,
        "checkpoint_epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc"),
        "n_test" if split == "test" else f"n_{split}": len(ys),
        "accuracy": accuracy_score(ys, ps),
        "macro_precision": precision_score(ys, ps, average="macro", zero_division=0),
        "macro_recall": recall_score(ys, ps, average="macro", zero_division=0),
        "macro_f1": f1_score(ys, ps, average="macro", zero_division=0),
        "per_class": classification_report(ys, ps, labels=labels, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": confusion_matrix(ys, ps, labels=labels).tolist(),
        "inference_time_s": t.elapsed,
        "inference_imgs_per_s": len(ys) / t.elapsed,
    }
    out = os.path.join(out_root, run, "metrics.json" if split == "test" else f"metrics_{split}.json")
    save_json(metrics, out)
    print(f"[eval:{run}:{split}] n={len(ys)} acc={metrics['accuracy']:.4f} "
          f"P={metrics['macro_precision']:.4f} R={metrics['macro_recall']:.4f} "
          f"F1={metrics['macro_f1']:.4f} ({metrics['inference_imgs_per_s']:.0f} img/s) -> {out}",
          flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--run", default=None, help="run name (default: config basename)")
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/<run>/best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-preprocess", dest="gpu_preprocess", action="store_true", default=None)
    ap.add_argument("--cpu-preprocess", dest="gpu_preprocess", action="store_false")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    args = ap.parse_args()
    main(args.config, args.run, args.device, args.gpu_preprocess, args.checkpoint, split=args.split)

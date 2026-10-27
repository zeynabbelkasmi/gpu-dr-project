import os
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src import data_loader

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")))


@pytest.fixture
def tiny_dataset(tmp_path):
    """24 tiny PNGs: 12 patients x 2 images, labels 0-4, named <patient>_<index>.png."""
    rng = np.random.default_rng(0)
    for i in range(12):
        label = i % 5
        d = tmp_path / str(label)
        d.mkdir(exist_ok=True)
        for j in range(2):
            arr = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"P{i:03d}_{j:02d}.png")
    return tmp_path


def test_find_images_reads_labels_from_folders(tiny_dataset):
    items = data_loader.find_images(str(tiny_dataset))
    assert len(items) == 24
    assert {l for _, l in items} == {0, 1, 2, 3, 4}


def test_split_is_patient_safe(tiny_dataset):
    items = data_loader.find_images(str(tiny_dataset))
    s = data_loader.split_by_patient(items, 0.5, 0.25, seed=1)
    pats = {k: {data_loader.patient_id(p) for p, _ in v} for k, v in s.items()}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"]
                or pats["val"] & pats["test"])
    assert sum(len(v) for v in s.values()) == len(items)


def test_cache_matches_direct_decode(tiny_dataset, tmp_path):
    items = data_loader.find_images(str(tiny_dataset))
    cache = data_loader.build_cache(items, (32, 32), str(tmp_path / "cache"), workers=2)
    ds = data_loader.DRDataset(items, (32, 32))
    assert cache.shape == (24, 32, 32, 3)
    for i in (0, 7, 23):
        assert np.array_equal(cache[i], ds.load(i))
    cached = data_loader.DRDataset(items, (32, 32), cache=cache)
    assert np.array_equal(cached.load(5), ds.load(5))


def test_augment_transform_changes_train_images_only():
    img = np.random.default_rng(0).uniform(0.2, 0.9, (64, 64, 3)).astype(np.float32)
    cfg = {"preprocess": {"clahe": False, "circular_crop": False, "normalize": False}}
    plain = data_loader.CpuTransform(cfg, augment=False)
    aug = data_loader.CpuTransform(cfg, augment=True)
    assert np.array_equal(plain(img), plain(img))
    outs = [aug(img) for _ in range(4)]
    assert all(o.shape == (3, 64, 64) for o in outs)
    assert any(not np.array_equal(o, plain(img)) for o in outs)


def test_validation_split_is_not_augmented(tiny_dataset):
    cfg = {"seed": 0, "data": {"root": str(tiny_dataset), "image_size": [32, 32],
                               "num_classes": 5, "train_frac": 0.5, "val_frac": 0.25,
                               "batch_size": 4, "num_workers": 0},
           "preprocess": {"clahe": False, "circular_crop": False, "normalize": False,
                          "gamma": 1.0, "augment": True}}
    loaders, _ = data_loader.make_loaders(cfg, "cpu", which=("val",))
    first = [x.clone() for x, _ in loaders["val"]]
    second = [x.clone() for x, _ in loaders["val"]]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_run_epoch_on_tiny_batch_cpu(tiny_dataset):
    from src.model import ReferenceCNN
    from src.train import run_epoch
    cfg = {"seed": 0, "data": {"root": str(tiny_dataset), "image_size": [32, 32],
                               "num_classes": 5, "train_frac": 0.7, "val_frac": 0.15,
                               "batch_size": 4, "num_workers": 0},
           "preprocess": {"clahe": True, "circular_crop": True, "normalize": True, "gamma": 1.0}}
    loaders, _ = data_loader.make_loaders(cfg, "cpu", which=("train",))
    model = ReferenceCNN(5)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    loss, acc, n = run_epoch(model, loaders["train"], torch.nn.CrossEntropyLoss(),
                             torch.device("cpu"), opt)
    assert n > 0 and np.isfinite(loss) and 0.0 <= acc <= 1.0


def test_macro_f1_from_confusion_matrix():
    from src.train import macro_f1
    assert macro_f1(torch.diag(torch.tensor([3, 2, 1]))) == pytest.approx(1.0)
    conf = torch.tensor([[2, 0], [2, 0]])
    assert macro_f1(conf) == pytest.approx((2 * 2 / (2 * 2 + 2 + 0)) / 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetcher_yields_same_batches_on_device():
    from torch.utils.data import DataLoader, TensorDataset
    from src.prefetcher import CudaPrefetcher
    ds = TensorDataset(torch.arange(20).float().view(10, 2), torch.arange(10))
    loader = DataLoader(ds, batch_size=3, pin_memory=True)
    got = list(CudaPrefetcher(loader, "cuda"))
    ref = list(loader)
    assert len(got) == len(ref) == len(CudaPrefetcher(loader, "cuda"))
    for (x, y), (rx, ry) in zip(got, ref):
        assert x.is_cuda and y.is_cuda
        assert torch.equal(x.cpu(), rx) and torch.equal(y.cpu(), ry)

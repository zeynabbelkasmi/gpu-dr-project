"""Point d'entrée unique pour accéder au dataset FDR26 + DataLoaders du pipeline.

Deux couches :
  1. Données de l'équipe (métadonnées + split partagé versionné) :
        from src.data_loader import load_split, load_image
        train = load_split("train")                  # DataFrame : image_id, patient_id, label, path, ...
        img = load_image(train.path.iloc[0])         # np.ndarray RGB uint8 (H, W, 3)
  2. Pipeline de l'enseignant (DRDataset, CpuTransform, build_cache, make_loaders).
     make_loaders lit le split partagé (data.split_csv) au lieu de re-mélanger les patients,
     pour que tout le monde entraîne/évalue sur exactement les mêmes images.

Structure attendue (voir docs/dataset_structure.txt) :
    DATA_DIR/images/<image_id>.jpg
    METADATA : feuille 'Toutes', colonnes image_id, classe (0-4), grade_DR, sexe, age, type_diabete
"""
import hashlib
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.paths import DATA_DIR, METADATA, ROOT, SPLITS_CSV

IMAGES_DIR = DATA_DIR / "images"
SHEET = "Toutes"
NUM_CLASSES = 5
CLASS_NAMES = ["DR0", "DR1", "DR2", "DR3", "DR4"]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------------------------
# 1. Données de l'équipe : métadonnées + split partagé
# ---------------------------------------------------------------------------------------------

def patient_id_of(image_id: str) -> str:
    """image_id = <code patient sur 8 caractères> + <numéro d'image sur 2 chiffres>, ex. T9NT4V5U02."""
    return image_id[:-2]


def load_metadata(drop_missing: bool = True) -> pd.DataFrame:
    """Métadonnées + chemin de chaque image. Colonne `label` = classe 0..4."""
    df = pd.read_excel(METADATA, sheet_name=SHEET)
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df = df.rename(columns={"classe": "label"})
    df["patient_id"] = df["image_id"].map(patient_id_of)
    df["path"] = df["image_id"].map(lambda i: str(IMAGES_DIR / f"{i}.jpg"))

    missing = ~df["path"].map(lambda p: Path(p).is_file())
    if missing.any():
        msg = f"{missing.sum()} image(s) listée(s) dans les métadonnées mais absente(s) : {df.loc[missing, 'image_id'].tolist()}"
        if not drop_missing:
            raise FileNotFoundError(msg)
        warnings.warn(msg + " -> ignorée(s)")
        df = df[~missing]

    cols = ["image_id", "patient_id", "label", "path", "grade_DR", "sexe", "age", "type_diabete"]
    return df[cols].reset_index(drop=True)


def load_split(name: str) -> pd.DataFrame:
    """Images d'un split ('train', 'val' ou 'test') d'après splits/splits.csv (versionné)."""
    if not SPLITS_CSV.is_file():
        raise FileNotFoundError(f"{SPLITS_CSV} absent : git pull (il est versionné), ne pas le régénérer chacun de son côté.")
    splits = pd.read_csv(SPLITS_CSV, dtype={"image_id": str})
    if name not in {"train", "val", "test"}:
        raise ValueError(f"split inconnu : {name}")
    meta = load_metadata()
    df = meta.merge(splits[["image_id", "split"]], on="image_id", how="inner")
    return df[df["split"] == name].drop(columns="split").reset_index(drop=True)


def load_image(path, size: int | None = None) -> np.ndarray:
    """Lit une image en RGB uint8. `size` : redimensionne en size x size (INTER_AREA)."""
    buf = np.fromfile(str(path), dtype=np.uint8)          # fromfile + imdecode : marche aussi avec accents dans le chemin (Windows)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Impossible de décoder {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def shared_splits(split_csv=SPLITS_CSV):
    """Le split partagé de l'équipe au format du pipeline : {"train": [(path, label), ...], ...}.

    Trié par image_id pour un ordre identique sur toutes les machines (le shuffle du
    DataLoader d'entraînement, seedé, fait le reste).
    """
    splits = pd.read_csv(split_csv, dtype={"image_id": str})
    meta = load_metadata().merge(splits[["image_id", "split"]], on="image_id", how="inner")
    out = {"train": [], "val": [], "test": []}
    for path, label, split in meta.sort_values("image_id")[["path", "label", "split"]].itertuples(index=False):
        out[split].append((path, int(label)))
    return out


# ---------------------------------------------------------------------------------------------
# 2. Pipeline de référence (code enseignant) : dataset, transform CPU, cache, loaders
# ---------------------------------------------------------------------------------------------

def find_images(root):
    """Walk root/<class_id>/*.png and return [(path, label), ...] (layout of prepare_fdr26.py)."""
    items = []
    for dirpath, _, fnames in os.walk(root):
        try:
            label = int(os.path.basename(dirpath))
        except ValueError:
            continue
        if not 0 <= label <= 4:
            continue
        for f in sorted(fnames):
            if f.lower().endswith(IMG_EXTS):
                items.append((os.path.join(dirpath, f), label))
    return items


def patient_id(path):
    """Patient id of a pipeline item. Folder layout: prefix before the LAST underscore
    (<patient>_<image>.png). Team layout (no underscore, <patient8><nn>.jpg): image_id[:-2]."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else patient_id_of(stem)


def split_by_patient(items, train_frac, val_frac, seed):
    by_pat = {}
    for p, l in items:
        by_pat.setdefault(patient_id(p), []).append((p, l))
    pats = sorted(by_pat)
    random.Random(seed).shuffle(pats)
    n = len(pats)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_p = set(pats[:n_train])
    val_p = set(pats[n_train:n_train + n_val])
    splits = {"train": [], "val": [], "test": []}
    for p in pats:
        key = "train" if p in train_p else "val" if p in val_p else "test"
        splits[key].extend(by_pat[p])
    return splits


def get_splits(cfg):
    """Splits used by every script. data.split_csv set -> the team's shared CSV (default);
    otherwise the teacher's folder walk + random patient shuffle on data.root."""
    d = cfg["data"]
    if d.get("split_csv"):
        path = Path(d["split_csv"])
        return shared_splits(path if path.is_absolute() else ROOT / path)
    items = find_images(d["root"])
    if not items:
        raise FileNotFoundError(
            f"No images found under {d['root']}/<class>/...  "
            "Set data.split_csv (team split) or point data.root at a class-folder dataset.")
    return split_by_patient(items, d["train_frac"], d["val_frac"], d.get("split_seed", cfg["seed"]))


class DRDataset(Dataset):
    """Loads, decodes and resizes an image; returns (C,H,W) tensor + label.

    CPU preprocessing is injected through `transform`.
    With `raw_uint8=True` the image is returned untouched (uint8, CHW) so a
    GPU can preprocess the whole batch after a 4x smaller host-to-device copy.
    """

    def __init__(self, items, image_size, transform=None, raw_uint8=False, cache=None):
        self.items = items
        self.image_size = tuple(image_size)
        self.transform = transform
        self.raw_uint8 = raw_uint8
        self.cache = cache

    # Windows DataLoader workers are spawned and receive a pickled copy of the dataset.
    # Pickling an np.memmap copies its whole content; send the file name instead and
    # re-map it inside the worker (the OS page cache is then shared by all workers).
    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.cache, np.memmap):
            state["cache"] = ("__memmap__", self.cache.filename)
        return state

    def __setstate__(self, state):
        c = state.get("cache")
        if isinstance(c, tuple) and c and c[0] == "__memmap__":
            state["cache"] = np.load(c[1], mmap_mode="r")
        self.__dict__.update(state)

    def __len__(self):
        return len(self.items)

    def load(self, idx):
        """Decode + resize only (the 'loading' stage). Returns HWC uint8."""
        if self.cache is not None:
            return np.asarray(self.cache[idx])
        path, _ = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            return np.asarray(im, dtype=np.uint8)

    def __getitem__(self, idx):
        u8 = self.load(idx)
        label = int(self.items[idx][1])
        if self.raw_uint8:
            return torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1))), label
        x = u8.astype(np.float32) / 255.0
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = x.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), label


class CpuTransform:
    """CPU preprocessing for one image: HWC float [0,1] -> CHW float.

    A class (not a lambda) so it pickles into spawned DataLoader workers.
    """

    def __init__(self, cfg, augment=False):
        self.cfg = cfg
        self.augment = augment
        self._rng = None

    def __call__(self, img):
        from src import preprocess_cpu
        if self.augment:
            if self._rng is None:      # created inside the worker, seeded per worker
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            img = preprocess_cpu.augment_image(img, self._rng)
        return preprocess_cpu.preprocess_image(img, self.cfg)


def get_preprocess_transform(cfg, augment=False):
    return CpuTransform(cfg, augment)


def cache_path(items, image_size, cache_dir):
    """Cache file name = hash of (image list, size): a changed list or size never reuses a stale file."""
    w, h = tuple(image_size)
    key = hashlib.md5(f"{w}x{h}\n".encode() + "\n".join(p for p, _ in items).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"u8_{w}x{h}_{len(items)}_{key}.npy")


def build_cache(items, image_size, cache_dir, workers=8):
    """Decode + resize every image ONCE into a uint8 (N, H, W, 3) .npy and return it memory-mapped.

    cache[i] == DRDataset(items, image_size).load(i) bit for bit (same decoder, same resize).
    Written to a temporary file then renamed, so an interrupted build never leaves a
    half-written cache behind. PIL releases the GIL while decoding, so threads scale.
    """
    os.makedirs(cache_dir, exist_ok=True)
    final = cache_path(items, image_size, cache_dir)
    if os.path.exists(final):
        return np.load(final, mmap_mode="r")
    w, h = tuple(image_size)
    tmp = final + f".tmp{os.getpid()}.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(items), h, w, 3))
    ds = DRDataset(items, image_size)

    def fill(i):
        arr[i] = ds.load(i)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fill, range(len(items))))
    arr.flush()
    del arr                         # Windows: a mapped file cannot be renamed
    os.replace(tmp, final)
    return np.load(final, mmap_mode="r")


def _worker_init(_):
    # one OpenCV thread per worker: N workers x many cv2 threads oversubscribes the CPU
    import cv2
    cv2.setNumThreads(0)


def make_loaders(cfg, device="cpu", gpu_preprocess=False, which=("train", "val", "test")):
    """Build DataLoaders for the requested splits.

    gpu_preprocess=True: workers only decode/resize and return uint8; the
    training loop preprocesses the batch on the GPU (assignment A5).
    """
    splits = get_splits(cfg)
    nw = cfg["data"]["num_workers"]
    augment = bool(cfg["preprocess"].get("augment", False))
    cache_dir = cfg["data"].get("cache_dir")
    if cache_dir and not os.path.isabs(cache_dir):
        cache_dir = str(ROOT / cache_dir)
    loaders = {}
    for k in which:
        # augmentation is for the training split only
        tf = None if gpu_preprocess else get_preprocess_transform(cfg, augment and k == "train")
        cache = build_cache(splits[k], cfg["data"]["image_size"], cache_dir) if cache_dir else None
        ds = DRDataset(splits[k], cfg["data"]["image_size"], transform=tf,
                       raw_uint8=gpu_preprocess, cache=cache)
        # TODO(A5.1): add the DataLoader options that feed a GPU efficiently:
        #   pin_memory (only when device == "cuda"), persistent_workers (when nw > 0),
        #   prefetch_factor (cfg["data"].get("prefetch_factor", 2), only valid when nw > 0),
        #   worker_init_fn=_worker_init (when nw > 0).
        loaders[k] = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=(k == "train"),
            num_workers=nw,
            drop_last=(k == "train"),
        )
    return loaders, splits

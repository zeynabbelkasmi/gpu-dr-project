"""Point d'entrée unique pour accéder au dataset FDR26. Tous les scripts passent par ici.

    from src.data_loader import load_split, load_image
    train = load_split("train")                  # DataFrame : image_id, patient_id, label, path, ...
    img = load_image(train.path.iloc[0])         # np.ndarray RGB uint8 (H, W, 3)

Structure attendue (voir docs/dataset_structure.txt) :
    DATA_DIR/images/<image_id>.jpg
    METADATA : feuille 'Toutes', colonnes image_id, classe (0-4), grade_DR, sexe, age, type_diabete
"""
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.paths import DATA_DIR, METADATA, SPLITS_CSV

IMAGES_DIR = DATA_DIR / "images"
SHEET = "Toutes"
NUM_CLASSES = 5
CLASS_NAMES = ["DR0", "DR1", "DR2", "DR3", "DR4"]


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

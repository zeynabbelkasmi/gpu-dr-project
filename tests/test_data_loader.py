"""Vérifie les invariants du split partagé. Usage : python -m pytest tests/test_data_loader.py"""
import pandas as pd
import pytest

from src.data_loader import NUM_CLASSES, load_image, load_split
from src.paths import SPLITS_CSV

pytestmark = pytest.mark.skipif(not SPLITS_CSV.exists(), reason="splits/splits.csv absent")


def test_no_patient_leak():
    s = pd.read_csv(SPLITS_CSV, dtype={"image_id": str})
    assert s.groupby("patient_id")["split"].nunique().max() == 1


def test_splits_disjoint_and_complete():
    parts = {n: load_split(n) for n in ("train", "val", "test")}
    ids = [set(p.image_id) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for p in parts.values():
        assert set(p.label) == set(range(NUM_CLASSES))


def test_load_image_rgb_uint8():
    row = load_split("val").iloc[0]
    img = load_image(row.path, size=224)
    assert img.shape == (224, 224, 3) and img.dtype == "uint8"

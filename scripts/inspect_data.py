"""Affiche la structure du dataset et les colonnes du fichier de métadonnées.
Sert à écrire src/data_loader.py sur la vraie structure.

Usage : python scripts/inspect_data.py > docs/dataset_structure.txt
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import DATA_DIR, METADATA  # noqa: E402

if not DATA_DIR.is_dir():
    sys.exit(f"{DATA_DIR} introuvable : lance d'abord  python scripts/download_data.py")

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

print("=== Arborescence (3 niveaux, 5 entrées max par dossier) ===")
def tree(d: Path, depth=0):
    if depth > 3:
        return
    kids = sorted(d.iterdir())
    for k in kids[:5]:
        print("  " * depth + k.name + ("/" if k.is_dir() else ""))
        if k.is_dir():
            tree(k, depth + 1)
    if len(kids) > 5:
        print("  " * depth + f"... (+{len(kids) - 5} autres)")
tree(DATA_DIR)

imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMG_EXT]
print(f"\n=== {len(imgs)} images ===")
print("Extensions :", Counter(p.suffix.lower() for p in imgs))
print("Images par dossier parent :", Counter(str(p.parent.relative_to(DATA_DIR)) for p in imgs))
print("Exemples de noms :", [p.name for p in imgs[:5]])

print(f"\n=== Métadonnées : {METADATA.name} ===")
for sheet, df in pd.read_excel(METADATA, sheet_name=None).items():
    print(f"\n--- feuille '{sheet}' : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(df.dtypes.to_string())
    print(df.head(5).to_string())
    for c in df.columns:
        n = df[c].nunique()
        if n <= 10:
            print(f"{c} -> {df[c].value_counts(dropna=False).to_dict()}")
        else:
            print(f"{c} -> {n} valeurs distinctes")

"""Chemins centralisés du projet. Tous les scripts importent d'ici, aucun chemin en dur ailleurs.

Emplacement des données :
  - par défaut : <racine du repo>/data
  - sinon      : variable d'environnement FDR26_DATA (ex. Colab : /content/data)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("FDR26_DATA", ROOT / "data")).expanduser().resolve()
ZIP_PATH = DATA_ROOT / "FDR26_classification.zip"
DATA_DIR = DATA_ROOT / "FDR26_classification"          # images extraites
METADATA = DATA_ROOT / "metadata_classification_DR.xlsx"

SPLITS_CSV = ROOT / "splits" / "splits.csv"            # versionné dans git : même split pour tous
RESULTS_DIR = ROOT / "results"

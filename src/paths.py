import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FDR26_DIR", ROOT / "data" / "FDR26_classification"))
METADATA = ROOT / "data" / "metadata_classification_DR.xlsx"
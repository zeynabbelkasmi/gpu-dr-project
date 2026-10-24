"""Télécharge, vérifie (MD5) et extrait le dataset FDR26 depuis Zenodo (record 20605487).

Usage (depuis la racine du repo) :
    python scripts/download_data.py

Relançable sans risque : ce qui est déjà téléchargé/vérifié/extrait est sauté.
Destination : ./data, ou le dossier donné par la variable d'environnement FDR26_DATA.
"""
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import DATA_DIR, DATA_ROOT, ZIP_PATH  # noqa: E402

BASE = "https://zenodo.org/records/20605487/files/"
FILES = {
    "FDR26_classification.zip": "db90631129dc3294480a9f10a7bb744a",
    "metadata_classification_DR.xlsx": "128b725a17177d38e419993c423ad4b9",
}
DONE = DATA_DIR / ".extraction_ok"  # marqueur : extraction terminée sans erreur


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")  # un téléchargement coupé ne laisse jamais un faux fichier complet
    with requests.get(f"{BASE}{name}?download=1", stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(part, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)


def extract() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        sys.exit(f"{DATA_DIR} existe mais l'extraction n'est pas marquée terminée "
                 f"(interrompue ou faite à la main) : supprime ce dossier et relance.")
    tmp = DATA_ROOT / "_extract_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    print("Extraction...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(tmp)
    entries = [e for e in tmp.iterdir() if e.name != "__MACOSX"]
    # Si le zip contient un seul dossier racine, on le remonte : évite data/FDR26_classification/FDR26_classification/
    src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
    if DATA_DIR.exists():
        DATA_DIR.rmdir()  # vide (vérifié plus haut)
    src.replace(DATA_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    DONE.touch()


def main() -> None:
    if DONE.exists():
        print("Données déjà prêtes dans", DATA_DIR)
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        p = DATA_ROOT / name
        if not p.exists():
            download(name, p)
        print(f"Vérification MD5 de {name}...")
        if md5(p) != expected:
            p.unlink()
            sys.exit(f"MD5 invalide pour {name} : fichier supprimé, relance le script.")
        print(f"{name} : OK")
    extract()
    print("Données prêtes dans", DATA_DIR)


if __name__ == "__main__":
    main()

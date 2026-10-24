import hashlib, sys, zipfile, requests
from pathlib import Path

BASE = "https://zenodo.org/records/20605487/files/"
FILES = {
    "FDR26_classification.zip": "db90631129dc3294480a9f10a7bb744a",
    "metadata_classification_DR.xlsx": "128b725a17177d38e419993c423ad4b9",
}
DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

for name, expected in FILES.items():
    p = DATA / name
    if not p.exists():
        print(f"Téléchargement de {name}...")
        with requests.get(f"{BASE}{name}?download=1", stream=True) as r:
            r.raise_for_status()
            with open(p, "wb") as f:
                for c in r.iter_content(1 << 20):
                    f.write(c)
    if md5(p) != expected:
        sys.exit(f"MD5 invalide pour {name} : supprime-le et relance.")
    print(f"{name} : OK")

out = DATA / "FDR26_classification"

if not out.exists():
    zipfile.ZipFile(DATA / "FDR26_classification.zip").extractall(out)
print("Données prêtes dans", out)
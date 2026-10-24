"""Génère splits/splits.csv : train/val/test ~70/15/15, par PATIENT (aucun patient dans deux splits),
stratifié par classe, graine fixe. À lancer UNE SEULE FOIS par l'équipe, puis commit du CSV.

Usage : python scripts/make_splits.py [--force]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_metadata  # noqa: E402
from src.paths import SPLITS_CSV  # noqa: E402

SEED = 42
N_FOLDS = 20                      # 20 plis de 5 % : 3 -> test (15 %), 3 -> val (15 %), 14 -> train (70 %)
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="écraser un splits.csv existant")
    args = ap.parse_args()
    if SPLITS_CSV.exists() and not args.force:
        sys.exit(f"{SPLITS_CSV} existe déjà : tout le monde doit utiliser le même. --force pour le régénérer.")

    df = load_metadata()

    # Contrôle de l'hypothèse patient_id = image_id[:-2] : sexe/âge doivent être constants par patient
    incoh = df.groupby("patient_id")[["sexe", "age"]].nunique().gt(1).any(axis=1)
    if incoh.any():
        print(f"ATTENTION : {incoh.sum()} patient(s) avec sexe/âge différents entre images : {incoh[incoh].index.tolist()[:10]}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient_id"])):
        fold.iloc[idx] = k
    df["split"] = "train"
    df.loc[fold.isin(VAL_FOLDS), "split"] = "val"
    df.loc[fold.isin(TEST_FOLDS), "split"] = "test"

    # Vérifications
    leaks = df.groupby("patient_id")["split"].nunique().gt(1).sum()
    assert leaks == 0, f"{leaks} patient(s) dans plusieurs splits"

    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["image_id", "patient_id", "label", "split"]].sort_values("image_id").to_csv(SPLITS_CSV, index=False)

    print(f"Écrit {SPLITS_CSV} ({len(df)} images, {df.patient_id.nunique()} patients, seed={SEED})")
    print("Fuite patient entre splits : 0")
    print("\nImages par classe et par split :")
    print(pd.crosstab(df["label"], df["split"], margins=True)[["train", "val", "test", "All"]])
    print("\nPatients par split :", df.groupby("split")["patient_id"].nunique().to_dict())
    multi = df.groupby("patient_id")["label"].nunique().gt(1).sum()
    print(f"Patients avec des images de classes différentes : {multi}")


if __name__ == "__main__":
    main()

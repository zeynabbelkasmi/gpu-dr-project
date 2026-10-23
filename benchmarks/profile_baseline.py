"""Profilage du pipeline CPU de reference (Semaine 1).

A FAIRE : remplacer les fonctions 'TODO' par les vraies fonctions du code
fourni par l'enseignant (src/data_loader.py, preprocess_cpu.py, etc.).

Usage (depuis la racine du repo) :
    python benchmarks/profile_baseline.py            # timing par etape
    python benchmarks/profile_baseline.py --cprofile # + profil detaille cProfile
"""
import argparse, cProfile, pstats, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.stage_timer import StageTimer

# ---- TODO : importer les fonctions reelles du projet ----
# from src.data_loader import load_image
# from src.preprocess_cpu import preprocess
# from src.filters_cpu import apply_filters
# from src.train_cpu import train_one_epoch
# from src.evaluate import evaluate

def run_pipeline(timer, n_images=200):
    """Execute chaque etape SEPAREMENT pour pouvoir la chronometrer."""
    paths = []   # TODO : liste des chemins d'images (train split)

    with timer.stage("1_loading"):
        images = []  # TODO : [load_image(p) for p in paths[:n_images]]

    with timer.stage("2_preprocessing"):
        pre = []     # TODO : [preprocess(im) for im in images]

    with timer.stage("3_filtering"):
        fil = []     # TODO : [apply_filters(im) for im in pre]

    with timer.stage("4_augmentation"):
        pass         # TODO : augmentation sur le train uniquement

    with timer.stage("5_training_1_epoch"):
        pass         # TODO : train_one_epoch(...)

    with timer.stage("6_evaluation"):
        pass         # TODO : evaluate(...)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cprofile", action="store_true")
    args = ap.parse_args()

    timer = StageTimer(cuda_sync=False)   # baseline = CPU
    if args.cprofile:
        prof = cProfile.Profile()
        prof.runcall(run_pipeline, timer)
        pstats.Stats(prof).sort_stats("cumulative").print_stats(25)
    else:
        run_pipeline(timer)

    timer.summary()
    Path("results").mkdir(exist_ok=True)
    timer.save("results/timing.csv")

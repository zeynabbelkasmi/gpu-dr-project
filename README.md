# GPU Acceleration of a Diabetic Retinopathy Classification Pipeline

Projet « Programmation GPU et optimisation des performances » — classification de la rétinopathie diabétique (5 grades ALFEDIAM, 0 à 4) sur le dataset FDR26, accélération GPU d'un pipeline CPU de référence.

**Équipe** : Zeynab Belkasmi, Ouiam Ouatat, Amina Ouaayi , Meryem Ramdi

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |   Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
# PyTorch : commande adaptée à ta machine sur https://pytorch.org/get-started/locally/
```

## Données (FDR26, Zenodo record 20605487, CC BY 4.0)

Le dataset (1,7 Go) n'est **pas** dans git. Une seule commande, depuis la racine du repo :

```bash
python scripts/download_data.py
```

Le script télécharge le zip et `metadata_classification_DR.xlsx`, vérifie les MD5, extrait, et peut être relancé sans risque.

| Où sont les données | Comment |
|---|---|
| Par défaut | `./data/` (ignoré par git) |
| Ailleurs (Colab, disque externe) | variable d'environnement `FDR26_DATA` avant de lancer quoi que ce soit |

```bash
# Linux / Colab
export FDR26_DATA=/content/data
# Windows PowerShell
$env:FDR26_DATA = "D:\datasets\fdr26"
```

Arborescence attendue :

```
data/
├── FDR26_classification.zip
├── metadata_classification_DR.xlsx
└── FDR26_classification/      # images extraites (structure : voir docs/dataset_structure.txt)
```

**Règle d'équipe** : aucun chemin en dur dans le code. Tout passe par `src/paths.py` :

```python
from src.paths import DATA_DIR, METADATA, SPLITS_CSV
```

Les scripts se lancent depuis la racine du repo (`python scripts/...`, `python benchmarks/...`).

## Lancer le pipeline

Tout se lance depuis la racine du repo, avec le venv activé (Windows : `.venv\Scripts\activate`).
PyTorch CUDA : `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`.

| Tâche | Commande | Sorties |
|---|---|---|
| Environnement (Table 1) | `python scripts/record_env.py` | `results/environment.json` |
| Baseline CPU (5 époques) | `python src/train_cpu.py` | `results/baseline/` |
| Métriques baseline sur **val** (le test reste intact) | `python src/evaluate.py --config configs/baseline.yaml --run baseline --device cpu --split val` | `results/baseline/metrics_val.json` |
| Profil cProfile (1 époque, 0 worker) | `python benchmarks/profile_baseline.py` | `results/profile_stages.json` |
| Temps par étape CPU/GPU (64 images) | `python benchmarks/benchmark_stages.py --n 64` | `results/timing.csv` |
| Module A : lecture / décodage / resize | `python benchmarks/profile_io.py` | `results/week1/loading_breakdown.json` |
| line_profiler + PyTorch Profiler (CPU) | `python benchmarks/profile_line.py` | `results/week1/` |
| Démo de synchronisation CUDA (A2.1) | `python benchmarks/sync_demo.py` | `results/sync_demo.json` |
| Cache + alternatives de chargement | `python benchmarks/benchmark_loading.py` | `results/week2/loading.csv`, `cache.json` |
| Run avec cache (un seul changement) | `python src/train.py --config configs/exp_cache.yaml --device cpu --run exp_cache --monitor` | `results/exp_cache/` |
| Traitement d'image CPU vs GPU | `python benchmarks/benchmark_gpu_processing.py` | `results/week2/*.csv` |
| Nsight Systems | `nsys profile -t cuda,nvtx --stats=true -o results/week2/nsys_preprocess python benchmarks/nsys_preprocess.py` | `results/week2/nsys_stats.txt` |
| Figures des rapports | `python scripts/make_week_figures.py` | `results/figures/` |
| Tests (équivalence CPU/GPU incluse) | `python -m pytest tests -v` | |

Rapports : [`report/week1_report.md`](report/week1_report.md) (baseline et goulots) et
[`report/week2_report.md`](report/week2_report.md) (refonte et accélération GPU du traitement d'image).
Consignes de l'enseignant : [`ASSIGNMENTS.md`](ASSIGNMENTS.md), avancement : `python scripts/progress.py`.

### Problèmes connus

- Le GPU d'un portable démarre en P8 (lien PCIe en Gen1) : les benchmarks GPU appellent `gpu_warmup()` avant de mesurer.
- `cupy-cuda13x` correspond à CUDA 13 ; avec un driver CUDA 12, installer `cupy-cuda12x`. Sans CuPy, seuls les variants CuPy du filtre médian sont sautés.
- Les workers DataLoader utilisent `spawn` sous Windows : les transforms doivent être picklables (pas de lambda).
- `tests/test_data_pipeline.py::test_prefetcher_yields_same_batches_on_device` échoue tant que `CudaPrefetcher` (semaine 3, A5.2) n'est pas écrit.
- L'image `2T2SALA702` des métadonnées est absente de l'archive Zenodo : 2 999 images utilisées.

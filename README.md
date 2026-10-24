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

- Baseline CPU : *(à compléter)*
- Version optimisée GPU : *(à compléter)*
- Profilage : `python benchmarks/profile_baseline.py [--cprofile]`
- Environnement matériel/logiciel : `python environment/collect_env.py`

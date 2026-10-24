# Semaine 1 - Baseline & Profiling (notes d'equipe)

## 1. Baseline execution record
- Date / machine :
- CPU / RAM / OS : (voir environment/hardware_software.md)
- Commande utilisee pour lancer le pipeline :
- Taille des splits : train = ?  val = ?  test = ?
- Classes equilibrees ? (images par classe et par split) :
- Fuite patient verifiee ? (meme patient dans 2 splits ?) :
- Metriques baseline : Accuracy = ?  Macro-P = ?  Macro-R = ?  Macro-F1 = ?
- Temps total :
- Couts uniques (one-time) : chargement dataset, cache, etc. =
- Couts repetes (par epoque) :

## 2. Profiling
| Etape | Temps (s) | % du total | Remarque |
|---|---|---|---|
| Loading | | | |
| Preprocessing | | | |
| Filtering | | | |
| Augmentation | | | |
| Training / epoch | | | |
| Evaluation | | | |

Bottleneck #1 : ...  (preuve : ...)
Bottleneck #2 : ...
Hypotheses d'optimisation (pour la semaine 2) : ...

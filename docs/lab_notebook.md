# Lab notebook

One entry per experiment. Write the prediction BEFORE you run anything. A wrong prediction with a good explanation is worth more than a right one you cannot explain.

## Entry template

```
### <assignment id> <short title>          date: <yyyy-mm-dd>
Config / commit:      <config file and git commit hash>
Question:             <what do you want to find out?>
Prediction:           <number or ranking, and the reason, written before the run>
Command:              <exact command line>
Result:               <measured numbers: mean +/- std, n repeats, cold vs steady>
Prediction vs result: <what was different?>
Explanation:          <why? what evidence supports it (profiler output, a second measurement)?>
Decision:             <keep / drop the change, and what you do next>
```

## A1.3 Baseline configuration review

| Key | Required by the brief? | Comment |
|---|---|---|
| `seed: 42` | yes (sec. 4: fixed seeds) | seeds Python, NumPy, torch, CUDA (`set_seed`); cuDNN benchmark off for accuracy runs (A1.5) |
| `train_frac 0.70`, `val_frac 0.15` | splits are required before model development (sec. 3); fractions are the starter's choice | we use the team's `splits/splits.csv` (`data.split_csv`): same 70/15/15, patient-level **and** class-stratified (StratifiedGroupKFold, seed 42). Patient separation is enforced in `scripts/make_splits.py` (assert) and `tests/test_data_loader.py::test_no_patient_leak` |
| `image_size [224, 224]` | sec. 3 leaves resolution to the reference; Week 3 may study it | sources are 1960–3600 px wide with 3 aspect ratios: 224x224 squashes 68 % of them (Week 1 report, sec. 3) |
| `num_classes: 5` | yes | DR0–DR4 |
| `batch_size 32`, `num_workers 2` | no | kept identical in every comparison |
| `clahe/circular_crop/normalize: true`, `gamma 1.0` | sec. 4 Module B | |
| `epochs 5`, `lr 1e-3`, AdamW | no | the training budget must stay identical across comparisons |
| checkpoint rule | test never used for selection (sec. 3) | `best.pt` = best **validation** accuracy (`train.py`); test only in `evaluate.py`, once per final model |
| keys added for optimized runs | | `cache_dir`, `preprocess.gpu`, `augment`, `amp`, `channels_last`, `prefetch_gpu`; split, seed, model, epochs, lr, batch stay identical |

## Entries

The full write-ups are `report/week1_report.md` and `report/week2_report.md`; the entries below are the short records.

### A1.7 Baseline run          date: 2026-10-27
Config / commit:      configs/baseline.yaml
Command:              python src/train_cpu.py
Result:               428.3 s wall, 69.3 +/- 2.3 s train + 16.0 s val per epoch; val acc 0.3911, macro-F1 0.3499
Explanation:          underfits at 5 epochs; DR2 recall 0.011. Loader-bound (CPU 55 % mean).

### A2.3-A2.6 Profiling          date: 2026-10-27
Prediction:           forward+backward ~50 %, loading ~30 %, CLAHE third
Result:               loading 65.4 %, fwd+bwd 25.1 %, preprocessing 4.0 % (results/profile_stages.json)
Prediction vs result: loading under-estimated: 5-10 Mpx JPEGs, decode 24 ms + resize 20 ms per image
Decision:             decoded-image cache first (Amdahl bound 2.89x)

### A3.5 Cache          date: 2026-10-27
Prediction:           loading per epoch -> < 1 s; end to end ~2x (less than the 2.89x bound because loading already overlaps compute with 2 workers)
Command:              python benchmarks/benchmark_loading.py; python src/train.py --config configs/exp_cache.yaml --device cpu --run exp_cache --monitor
Result:               56.4 s -> 0.34 s per epoch of loading; 428.3 s -> 280.8 s end to end (1.53x), steady epoch 1.67x; identical metrics
Prediction vs result: end to end lower than predicted: the CPU CNN is now the bottleneck (46 img/s vs 72 img/s compute-only)
Decision:             keep; next bottleneck = CPU training (Week 3: GPU)

### A4.13 Batch sweep          date: 2026-10-27
Prediction:           GPU overtakes the CPU at B ~ 8 for CLAHE, best per-image cost at the largest batch
Result:               crossover between B = 1 and 8 (correct); best per-image cost at B = 32, not 128 (wrong)
Explanation:          CLAHE int64 index tensors exceed the 32 MB L2 above B = 32 (hypothesis, to verify with Nsight Compute)
Decision:             call GPU preprocessing on 32-image batches

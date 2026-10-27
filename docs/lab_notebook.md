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

The full write-ups are `report/week1_report.md`, `report/week2_report.md` and `report/week3_report.md`; the entries below are the short records.

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

### A5 + A6.10 GPU pipeline, one change at a time          date: 2026-10-27
Config / commit:      configs/exp_*.yaml, optimized_cnn.yaml (branch week3-meryeme); Colab T4, 1 CPU core
Question:             what does each pipeline change give once training runs on the GPU?
Command:              python scripts/week3_experiments.py
Result:               steady epoch: CPU 200.3 s -> GPU 7.83 s (25.6x) -> + GPU preprocessing 3.92 s (51x)
                      -> + GPU-resident data 3.77 s -> + precompute 3.39 s -> + AMP/channels_last/cuDNN/fused AdamW
                      1.48 s (135x) -> + CUDA graphs 0.90 s (222x, cold epoch 22.8 s). Prefetcher 3.97 s (no gain).
                      AMP, channels_last, compile ALONE on the CPU-preprocessing path: 7.88-8.20 s (no gain).
Surprise:             AMP alone gave nothing; it gave 2.3x only once combined after GPU preprocessing
                      (no prediction was written before these runs)
Explanation:          the run was input-bound (CLAHE on 1 CPU core); compute optimizations only pay once the
                      run is compute-bound (Amdahl). Prefetch: a 4.8 MB uint8 copy vs ~50 ms per step.
Decision:             keep GPU preprocessing, GPU-resident data, precompute, AMP, channels_last, cuDNN benchmark,
                      fused AdamW; compile off (break-even ~36 epochs); prefetcher kept as code, not in final config

### A5.9 / A5.10 Precompute and PCIe bytes          date: 2026-10-27
Result:               PCIe per epoch: float32 1,205 MiB -> uint8 301 MiB -> GPU-resident 0 (366 MiB once).
                      Precompute: 3.77 -> 3.39 s per epoch for 1,464 MiB float32 (732 MiB float16 under AMP)
Decision:             precompute (fits easily in 16 GB); augmentation stays random on the fly

### A6.6 / A6.9 ResNet-18 and training recipe (3 seeds)          date: 2026-10-27
Config / commit:      configs/exp_resnet18.yaml, exp_r18_*.yaml, optimized.yaml
Command:              python scripts/week3_experiments.py --seeds 42,43,44 --only optimized,exp_r18_...
Result:               ResNet-18 vs reference CNN (same pipeline, seed 42): F1 0.525 vs 0.325.
                      Val F1 (mean +/- std, n=3): no schedule 0.508+/-0.059, lr 3e-4 0.525+/-0.015,
                      augment 0.473+/-0.012, cosine 0.577+/-0.026, regul 0.480+/-0.015, all four 0.564+/-0.018,
                      all four 10 epochs 0.563+/-0.016. Final optimized (= cosine) re-run: 0.556+/-0.010.
Surprise:             augmentation and regularization HURT at 5 epochs
Explanation:          cosine brings the lr to 0 where the no-schedule run diverged (train 0.63 / val 0.45 at
                      epoch 4); augmentation/regularization make the task harder and 5 epochs is too short.
                      Same config, same seeds: 0.577 vs 0.556 -> GPU non-determinism (cudnn benchmark, atomics).
Decision:             final = fast pipeline + ResNet-18 + cosine: pooled 6 runs val F1 0.566+/-0.021, 2.41 s/epoch
                      (86x the Colab CPU, ~22x the Week 2 laptop CPU). Test set untouched until A7.8.

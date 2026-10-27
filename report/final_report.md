# GPU Acceleration of a Diabetic Retinopathy Classification Pipeline - Final Report

> Skeleton following the required outline (section 8.B of the brief). Replace every
> `<...>` with measured values. Target length 10-15 pages.

## 1. Abstract and objectives

<Problem, methods, headline results, main conclusion. Example sentence: "The
principal bottleneck was <stage> (p = <x>% of baseline runtime); the optimized
pipeline achieves <S>x end-to-end speedup at <acc> test accuracy vs <acc_base>
for the baseline.">

## 2. Problem and pipeline description

Dataset: 3,000 fundus images, 5 balanced DR classes; patient-safe splits.
Pipeline stages: loading -> preprocessing (CLAHE, circular crop, normalize) ->
filters (Gaussian, median, green channel, contrast stretch) -> augmentation ->
CNN training -> evaluation. Reference model: 3-block CNN (see `src/model.py`).

## 3. Experimental environment (Table 1)

| Item | Value |
|---|---|
| CPU | <model, cores, freq> |
| GPU | <model, VRAM> |
| RAM | <GB> |
| OS | <distro/version> |
| Python | <version> |
| PyTorch / CUDA / cuDNN | <versions> |
| Image resolution / batch size / splits | 224x224 / <B> / 70-15-15 patient-level |

## 4. Baseline profiling

<Methodology: cProfile + PyTorch Profiler + manual stage timers. Insert
Figure 1 (runtime breakdown) and the ranked bottleneck table. State the
principal bottleneck and the evidence.>

| Rank | Stage | Time (s) | % of total | Evidence |
|---|---|---|---|---|
| 1 | <stage> |  |  |  |
| 2 |  |  |  |  |

## 5. Redesign and optimization decisions

For each change: motivation (linked to section 4 measurement), implementation,
expected effect, trade-off.

| # | Change | Motivation (measured) | Expected | Trade-off |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

## 6. Benchmarking methodology

Warm-up: <n> iterations. Timing boundary: <start/end>. Synchronization:
`torch.cuda.synchronize` before/after. Repeats: <n>, reported as mean +/- std.
Transfers: included in end-to-end timing.

## 7. Results

Table 2 - Stage-level performance (fill from `results/timing.csv`):

| Pipeline stage | CPU baseline (s) | Optimized (s) | Speedup |
|---|---|---|---|
| Image loading |  |  |  |
| Preprocessing |  |  |  |
| Filtering (Gaussian/median/green/contrast) |  |  |  |
| Augmentation |  |  |  |
| Training per epoch |  |  |  |
| Evaluation |  |  |  |
| Total pipeline |  |  |  |

Table 3 - Classification quality (fill from `results/<run>/metrics.json`):

| Metric | Baseline model | Final model |
|---|---|---|
| Accuracy |  |  |
| Macro-precision |  |  |
| Macro-recall |  |  |
| Macro-F1 |  |  |
| Confusion matrix |  |  |

Figures: Fig 1 baseline breakdown; Fig 2 speedup comparison; Fig 3 accuracy vs
execution time; Fig 4 ablation; Fig 5 CPU/GPU utilization + memory.

Amdahl analysis: with p = <fraction> of runtime accelerated by s = <x>,
S_total = 1 / ((1 - p) + p/s) = <x>. <Discuss why the measured end-to-end
speedup does or does not match.>

## 8. Discussion and critical analysis

<Why did some stages benefit more? Transfer/launch overhead? Did the bottleneck
shift? Were isolated gains reflected end-to-end? Algorithmic vs hardware gains.
What changed in macro-F1 / per-class recall? Was final eval untouched by model
selection? Memory/portability trade-offs? Would this transfer to other GPUs?>

## 9. Limits and threats to validity

Dataset size (3,000 images, balanced classes), generalization, hardware
dependence, numerical differences between CPU/GPU CLAHE and precision modes,
measurement uncertainty (profiling overhead, cold vs steady state), seed
variance, single-machine reproducibility.

## 10. Conclusion and future work

<Contribution, strongest evidence, lessons learned. Future work: model
compilation/operator fusion, inference-only optimization, energy measurement,
performance model for CPU-vs-GPU dispatch.>

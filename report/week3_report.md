# Week 3 — End-to-end GPU pipeline (A5) and training acceleration (A6)

**Project:** GPU acceleration of a DR classification pipeline (ENSEEIHT, 2026).
**Machine (all Week 3 numbers):** Google Colab, Tesla T4 16 GB (sm_75, 40 SMs), 1 physical CPU core /
2 threads, 14 GB RAM, PyTorch 2.11 + CUDA 13.0 (`results/colab_t4/environment.json`). This is **not**
the Week 1–2 machine (RTX 4070 Laptop, 8-core Ryzen): cross-week comparisons are flagged as such.
**Rule followed:** every number comes from `results/week3_table.csv` (one row per run and seed) or
`results/<run>/history.json`; commands are in section 9.

---

## 0. Summary

| | Steady epoch (train + val) | vs CPU (same machine) | Val macro-F1 |
|---|---|---|---|
| CPU reference (`exp_cache`, reference CNN, CPU preprocessing) | 200.3 s | 1× | — (2 epochs, timing only) |
| Same model on the GPU (device effect only) | 7.83 s | 25.6× | 0.325 |
| Fastest pipeline, reference CNN (`optimized_cnn`) | 1.48 s | **135×** | 0.305 |
| ↳ + CUDA graphs (`exp_compile_ro`, cold epoch 22.8 s) | 0.90 s | **222×** | 0.328 |
| **Final design (`optimized`): fast pipeline + ResNet-18 + cosine schedule** | **2.34 s** | **86×** | **0.556 ± 0.010** (3 seeds) |

- The pipeline work (A5 + the GPU-resident data path) makes the reference CNN **5.3× faster than the
  same model on the same GPU** (7.83 → 1.48 s), and 8.7× with CUDA graphs.
- The model and recipe work (A6.6, A6.9) raises validation macro-F1 from **0.33 to 0.56–0.58** while an
  epoch of the larger model costs 2.34 s, 3.6× less than the same ResNet-18 on the default GPU pipeline (8.49 s).
- Main lesson: **every optimization only paid when it targeted the current bottleneck.** AMP,
  `channels_last` and `torch.compile` gave nothing while the run was input-bound, and 2.3× once it was
  compute-bound (section 4.2).

---

## 1. Starting point

End of Week 2 the pipeline had a decoded-image cache (A3) and bit-exact batched GPU preprocessing (A4),
but training still ran on the CPU and the GPU path in `src/train.py` was a TODO. Week 3 connects the
pieces and moves training to the GPU.

## 2. What was implemented

| TODO | File | Implementation |
|---|---|---|
| A5.1 | `src/data_loader.py` | `pin_memory` (CUDA only), `persistent_workers`, `prefetch_factor`, `worker_init_fn` (last three only with workers) |
| A5.2 | `src/prefetcher.py` | `CudaPrefetcher`: copy of batch i+1 on a side stream during compute of batch i; `wait_stream` before use, `record_stream` so the allocator does not recycle memory early. `test_prefetcher_yields_same_batches_on_device` passes on the T4 |
| A5.3–A5.5 | `src/train.py` | `resolve_gpu_preprocess`, `make_gpu_pre` (augmentation on train only), `non_blocking` copies, loaders wrapped in the prefetcher |
| beyond A5 | `src/gpu_resident.py` | **GPU-resident data**: the uint8 split is copied once to GPU memory, batches are a `randperm` + `index_select` on the GPU; no DataLoader, no per-epoch PCIe traffic. Optional **precompute** (A5.9) |
| A6.1–A6.3 | `src/train.py` | autocast + GradScaler; `channels_last`; optional `torch.compile` (`compile_mode: reduce-overhead` = CUDA graphs); also `cudnn_benchmark` and fused AdamW as config flags |
| A6.4 | `src/evaluate.py` | evaluation with the training precision and memory format |
| A6.5 | `src/train_gpu.py` | entry point for `configs/optimized.yaml` |
| A6.6 | `src/model.py` | ResNet-18, ImageNet weights, new head. 11.18 M parameters / 3.63 GFLOPs per image vs 0.62 M / 1.01 GFLOPs for the reference CNN (`python -m src.model`) |
| A6.8 | `configs/` | `optimized.yaml`, `optimized_cnn.yaml` and one `exp_*.yaml` per single change |
| A6.9–A6.10 | `scripts/week3_experiments.py` | runs everything in the A6.10 order, repeats over seeds, writes `results/week3_table.csv` and `week3_table_seeds.csv` |

## 3. Methodology

- **One change at a time.** Every `exp_*.yaml` differs from its parent by one key (written in its header).
  Same split (`splits/splits.csv`, patient-level), same 5-epoch budget, batch 32.
- **Timing:** `GpuTimer` synchronizes CUDA around each epoch. We report the **steady epoch** (mean of
  epochs 1–4, train + val) separately from the **cold epoch** (epoch 0: cuDNN autotuning, compilation)
  and from the **setup time** (cache build, GPU upload, precompute).
- **Accuracy:** checkpoint = best validation accuracy; we report validation macro-F1 at that epoch.
  **The test set was not used** (it is evaluated once per final model in A7.8).
- **Seeds:** the ResNet-18 recipe runs were repeated with seeds 42, 43, 44 (split unchanged); we report
  mean ± std. Pipeline runs (reference CNN) are single-seed: we use them for time, not for accuracy.

## 4. Results

### 4.1 Pipeline: one change at a time (reference CNN, seed 42)

| Run | Change vs parent | Steady epoch | vs CPU | img/s (train) | Peak GPU mem |
|---|---|---|---|---|---|
| `exp_cache_cpu` | CPU reference | 200.28 s | 1× | 11 | — |
| `exp_cache_gpu` | device only | 7.83 s | 25.6× | 331 | 868 MB |
| `exp_gpu_pre_gpu` | + GPU preprocessing (A5.3) | 3.92 s | 51.1× | 581 | 967 MB |
| `exp_prefetch_gpu` | + CUDA prefetcher (A5.5) | 3.97 s | 50.4× | 575 | 977 MB |
| `exp_amp_gpu` | `exp_cache_gpu` + AMP | 8.20 s | 24.4× | 301 | **574 MB** |
| `exp_channels_last_gpu` | `exp_cache_gpu` + channels_last | 7.88 s | 25.4× | 327 | 973 MB |
| `exp_compile_gpu` | `exp_cache_gpu` + torch.compile | 7.89 s | 25.4× | 320 | 913 MB |
| `exp_gpu_resident_gpu` | `exp_gpu_pre` + data in GPU memory | 3.77 s | 53.1× | 603 | 1339 MB |
| `exp_precompute_gpu` | + precompute (A5.9) | 3.39 s | 59.1× | 660 | 2433 MB |
| `optimized_cnn` | + AMP, channels_last, cuDNN benchmark, fused AdamW | **1.48 s** | **135×** | 1523 | 1288 MB |
| `exp_compile_ro_gpu` | + compile, `reduce-overhead` | **0.90 s** | **222×** | 2557 | 866 MB |

### 4.2 Why each step gave what it gave

1. **Device effect, 25.6×.** Same model, same CPU preprocessing in the workers. The GPU is already
   waiting for data: the loader runs CLAHE on **one physical core** (Colab).
2. **GPU preprocessing, 2.0×.** Workers now only read uint8 from the cache; CLAHE/crop/normalize run as
   batched kernels. This removes the CPU from the critical path.
3. **Prefetcher: no gain (3.92 → 3.97 s, within noise).** With GPU preprocessing the copy is a 4.8 MB
   uint8 batch (Week 2, A4.14, measured 1.5 ms for a 128-image uint8 batch, i.e. well under 1 ms for 32),
   against ~50 ms per step (3.92 s / ~80 steps): overlapping it cannot save more than ~1–2 %. The test proves it is correct; the profiler (A7) will show the overlap.
4. **AMP, channels_last, compile alone: no gain** (7.88–8.20 s vs 7.83 s), because the run is
   **input-bound**: making the compute faster only lengthens the time the GPU waits for the loader
   (Amdahl). AMP still cuts peak memory by 34 % (868 → 574 MB). `torch.compile` adds ~20 s of
   compilation to the cold epoch (29.5 s vs 9.9 s) for no steady-state gain.
5. **GPU-resident data: 1.04×.** Removing the DataLoader entirely saved only 0.15 s per epoch: after
   step 2 the loader was no longer the bottleneck; GPU preprocessing and fp32 compute were. Copying the
   two splits once costs 366 MiB of GPU memory and ~1.5 s of setup (1.74 s vs 0.27 s).
6. **Precompute: 1.11×.** CLAHE is no longer recomputed every epoch, but it was already cheap on the GPU.
   Price: the float32 result of both splits, 1.46 GiB (peak 2.4 GB) — the A5.9 trade-off (section 5).
7. **AMP + channels_last + cuDNN benchmark + fused AdamW together: 2.3× (3.39 → 1.48 s).** The same
   options that gave nothing in step 4 now pay, because the run has become **compute-bound**: AMP uses
   the T4 Tensor Cores (float16), channels_last is the layout they prefer, cuDNN benchmark picks the
   fastest convolution for our fixed shapes, the fused optimizer replaces per-tensor kernels by one.
   Precomputed images stored in float16 under AMP bring memory back to 1.29 GB.
8. **CUDA graphs: 1.64× more (1.48 → 0.90 s), but a 22.8 s cold epoch.** The reference CNN is small, so
   a step is many short kernels and the time goes into **launching** them from Python. A CUDA graph
   replays the whole step with one launch. Break-even: (22.76 − 2.00) s extra once / (1.48 − 0.90) s saved
   per epoch ≈ **36 epochs**. Not worth it for our 5-epoch runs, worth it for a long training. Kept off in
   `optimized.yaml`.

### 4.3 Model and training recipe (ResNet-18, 3 seeds, one change at a time on `optimized.yaml`)

`exp_resnet18_gpu` (default GPU pipeline, fp32, seed 42): 8.49 s per epoch, val acc 0.516, F1 0.525,
against 0.382 / 0.325 for the reference CNN on the same pipeline: **+20 F1 points** for 3.6× the FLOPs.
On the fast pipeline (AMP etc.), ResNet-18 trains in 2.3 s per epoch (3.6× faster), and overfits:
seed 42 reached 0.63 train accuracy at epoch 4 while validation dropped from 0.54 to 0.45.

| Run (mean ± std, n = 3) | Change | Val acc | Val macro-F1 | Steady epoch |
|---|---|---|---|---|
| fast pipeline + ResNet-18, no schedule (first `optimized`) | — | 0.535 ± 0.035 | 0.508 ± 0.059 | 2.29 s |
| `exp_r18_lr3e4` | lr 1e-3 → 3e-4 | 0.533 ± 0.007 | 0.525 ± 0.015 | 2.35 s |
| `exp_r18_augment` | GPU flip + rotation | 0.519 ± 0.013 | 0.473 ± 0.012 | 2.63 s |
| **`exp_r18_cosine`** | **cosine LR schedule** | **0.579 ± 0.022** | **0.577 ± 0.026** | 2.48 s |
| `exp_r18_regul` | weight decay 1e-4 + label smoothing 0.1 | 0.501 ± 0.009 | 0.480 ± 0.015 | 2.48 s |
| `exp_r18_recipe` | all four | 0.580 ± 0.015 | 0.564 ± 0.018 | 2.63 s |
| `exp_r18_recipe10` | all four, 10 epochs | 0.567 ± 0.015 | 0.563 ± 0.016 | 2.64 s |
| **`optimized` (final = fast pipeline + ResNet-18 + cosine)** | re-run of `exp_r18_cosine` | 0.563 ± 0.008 | 0.556 ± 0.010 | 2.34 s |

The first row was overwritten in `results/optimized*` when the final config was re-run; its numbers are
the ones printed by `week3_experiments.py` before the change (same table format).

- **Cosine is the only change that helps** (+0.07 F1 over the no-schedule run). The learning rate goes
  to zero over the last epochs, exactly where the no-schedule run diverged.
- **Augmentation and regularization hurt at 5 epochs**: they make the training task harder and the
  budget is too short to benefit. Doubling the budget with them (`recipe10`) did not catch up.
- Combining the four changes is not better than cosine alone (difference smaller than one std), so we
  keep the simplest configuration.
- **Run-to-run noise is real even at a fixed seed.** `optimized` and `exp_r18_cosine` have the *same*
  configuration and seeds, yet give 0.556 ± 0.010 and 0.577 ± 0.026: `cudnn_benchmark: true` and atomic
  additions in the backward pass make GPU training non-deterministic. Pooling the 6 runs:
  **val F1 0.566 ± 0.021, val acc 0.571 ± 0.017, 2.41 ± 0.08 s per epoch.** With n = 3 the gain of cosine
  is consistent in direction but not statistically established; we state it as such.

## 5. A5.9 and A5.10: precompute or not, and bytes over PCIe

**Bytes per training epoch (2,099 images, 224×224×3):**

| Data path | Bytes over PCIe per epoch |
|---|---|
| CPU preprocessing, float32 batches | 1,205 MiB |
| GPU preprocessing, uint8 batches | 301 MiB (4× less) |
| GPU-resident (uint8 uploaded once: train + val 366 MiB at setup) | **0** |

**Precompute vs on-the-fly (A5.9):**

| | On-the-fly (GPU-resident uint8) | Precomputed |
|---|---|---|
| GPU memory for train + val | 366 MiB | 1,464 MiB (float32) / 732 MiB (float16 under AMP) |
| Steady epoch (reference CNN, fp32) | 3.77 s | 3.39 s (−10 %) |
| Augmentation | before CLAHE, as the CPU reference | after crop/normalize (order changes, see below) |

Decision: **precompute**, because the memory fits easily on a 16 GB GPU and every epoch is 10 % faster;
the decision would flip for a dataset that does not fit in GPU memory (300,000 images ≈ 45 GB uint8),
where the uint8 cache + on-the-fly GPU preprocessing is the only option. For augmentation, **random on
the fly** is the right choice (a fixed set of augmented copies multiplies storage and gives the network
the same "random" image every epoch); in our 5-epoch budget augmentation was not kept anyway. With
precompute, flip and rotation are applied after the deterministic preprocessing: they commute with the
centred circular crop, but CLAHE tiles are computed on the un-rotated image — a small, documented
difference from the CPU order.

## 6. Checkpoint answers

A5.11 and A6.11 are answered in `docs/week3_notes.md`.

## 7. Limitations and threats to validity

- **The CPU reference is weak.** Colab gives one physical core, so 200 s per epoch on Colab vs 51.6 s
  steady for the same `exp_cache` run on the 8-core Week 2 laptop. Against that laptop CPU the final
  design is ~22× faster per epoch and `optimized_cnn` ~35×; both machines must be named whenever a
  speedup is quoted.
- **GPU differs from Weeks 1–2** (T4 here, RTX 4070 Laptop before).
- Pipeline runs are single-seed; their time is reliable (±0.1 s between seeds in section 4.3), their
  accuracy is not used.
- Not done yet: the A5.8 `num_workers` sweep and the A6.7 batch-size sweep
  (`benchmarks/sweep_loader.py`; workers are irrelevant once the data is GPU-resident, batch size is
  not), the profiler trace confirming the prefetcher overlap and the moved bottleneck (A7).
- The CPU run used 2 epochs (one cold + one steady) to keep it to ~9 minutes (including a 108 s cache build).

## 8. Next steps (Week 4)

A6.7 batch-size sweep on the final config; A7 GPU profiling of `optimized` (where does the 2.3 s go
now?); ablation A7.6 (switch off one optimization at a time from `optimized`); test-set evaluation
once for the baseline and once for `optimized` (A7.8).

## 9. Reproduce (Colab, T4)

```
python -m pytest tests/test_data_pipeline.py -q
python scripts/week3_experiments.py                                   # pipeline runs, seed 42
python scripts/week3_experiments.py --seeds 42,43,44 --only optimized,exp_r18_lr3e4_gpu,exp_r18_augment_gpu,exp_r18_cosine_gpu,exp_r18_regul_gpu,exp_r18_recipe_gpu,exp_r18_recipe10_gpu
python -m src.model                                                   # parameters and FLOPs
```

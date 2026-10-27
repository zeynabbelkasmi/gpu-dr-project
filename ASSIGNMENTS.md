# Assignments

This file lists everything you have to do, in order. It follows the plan in
`Project_GPU_Acceleration_2026_V1.pdf` (the brief), compressed into **two weeks**: the end-to-end
run and the final report are included, and nothing is due after day 14. Read the brief first; this file does not
replace it.

The rule of the project: **profile, hypothesize, optimize, validate, re-profile.** Every
optimization needs evidence from a measurement, and every claim in your report needs a number
from a file in `results/`.

## How to work

- Every step below carries a tag `TODO(A<n>.<m>)`. Steps that edit a file have the tag in that
  file. Steps with no file of their own (setup, runs, measurements, answers) are listed in
  `docs/checklist.md`. Run `python scripts/progress.py` to list the tags that remain and delete a
  tag when the step is done and its evidence is in the notebook or in `results/`.
- `python -m pytest tests -v` is your automatic check. Tests that need a GPU are skipped on
  machines without CUDA, so on a CPU-only machine only the A3 test fails at the start. Run the
  GPU tests on a CUDA machine before you call a step done.
- Before every experiment, write a prediction in `docs/lab_notebook.md` (template inside).
  Run, compare, explain. Wrong predictions are fine if you explain them.
- Change one factor at a time. Keep a config file per experiment: `configs/exp_<name>.yaml`.
- Commit after each assignment. Your history is part of the evidence that the work is yours.
- Never use the test set to choose anything. `src/evaluate.py` runs once per final model.

Checkpoints: by day 7, A1-A4 are done and the baseline profile is in `results/`. Run the final
end-to-end experiments (A7) by day 12 so the report is written from finished results. Code and
report are due on day 14.

The brief spreads this work over four weeks; here it is compressed into two, so budget your
time: A4 is the largest block, and A4.11 and A6.3 are optional. Without an NVIDIA GPU you can
complete A1 to A3 only; ask your instructor for a lab machine or a cloud GPU for A4 to A7.

Work in teams of 2-3. Agree who implements what, but every member must be able to explain every
file in the defense.

## Map: brief, schedule, rubric

| Assignment | Brief section (with its week label) | Days | Rubric criterion |
|---|---|---|---|
| A1 Baseline execution and verification | 5 (Week 1), Modules A-E | 1-2 | Baseline execution and understanding (10%) |
| A2 Profiling and bottleneck identification | 5 (Week 1), 7 | 2-3 | Profiling methodology (20%) |
| A3 Loading redesign: decoded-image cache | 5 (Week 2), 6 | 3-4 | Algorithmic redesign (20%) |
| A4 GPU image processing | 5 (Week 2) | 4-6 | Algorithmic redesign and GPU implementation (20%) |
| A5 End-to-end input pipeline | 5 (Week 3) | 7-8 | Experimental rigor (15%) |
| A6 Training acceleration and model redesign | 5 (Week 3), 6 | 8-10 | Classification evaluation (10%) |
| A7 Final benchmarking and GPU profiling | 5 (Week 4), 7 | 11-12 | Performance analysis (15%) |
| A8 Figures, tables, report and defense | 8, 9, 10 | 12-14 | Code quality, report, defense (10%) |

## Files you will edit

| File | What you do | Assignment |
|---|---|---|
| `configs/baseline.yaml` | read and review (do not change values) | A1 |
| `src/utils.py` | synchronized timer, seeding decision | A1, A2 |
| `src/data_loader.py` | decoded-image cache, DataLoader options | A3, A5 |
| `src/preprocess_gpu.py` | batched GPU preprocessing and augmentation | A4 |
| `src/filters_gpu.py` | four GPU filters | A4 |
| `src/prefetcher.py` | overlap copies with compute | A5 |
| `src/train.py` | GPU preprocessing hook, AMP, channels_last, compile, prefetch | A5, A6 |
| `src/evaluate.py` | evaluation with the same precision as training | A6 |
| `src/model.py` | justified alternative model | A6 |
| `src/train_gpu.py` | entry point for your optimized run | A6 |
| `configs/optimized.yaml` and `configs/exp_*.yaml` | experiments | A5, A6, A7 |
| `benchmarks/benchmark_stages.py` | GPU rows and speedups | A5 |
| `benchmarks/benchmark_end_to_end.py` | aggregation | A7 |
| `benchmarks/profile_gpu.py` | PyTorch Profiler analysis | A7 |
| `scripts/make_figures.py` | the five figures and the tables | A8 |
| `docs/`, `report/`, `environment/` | notebook, design proposal, report, hardware table | all |

---

## A1. Baseline execution and verification (days 1-2)

**Goal:** understand the CPU pipeline and produce numbers you trust, before touching anything.

1. **TODO(A1.1)** Create the environment (see `README.md`) and download the data:
   `python scripts/prepare_fdr26.py --download`. Count the images per class folder.
2. **TODO(A1.2)** Run `python scripts/record_env.py` and fill `environment/hardware_software.md`
   (Table 1 of the brief).
3. **TODO(A1.3)** Compare `configs/baseline.yaml` with sections 3, 4 and 7 of the brief, line by
   line, and answer the questions at the top of the config in `docs/lab_notebook.md`.
4. **TODO(A1.4)** Verify the data split. Run this and record the output in your notebook:
   ```python
   import yaml
   from src.data_loader import find_images, split_by_patient, patient_id
   cfg = yaml.safe_load(open("configs/baseline.yaml"))
   s = split_by_patient(find_images(cfg["data"]["root"]), cfg["data"]["train_frac"],
                        cfg["data"]["val_frac"], cfg["seed"])
   for k, v in s.items():
       print(k, len(v), {c: sum(l == c for _, l in v) for c in range(5)}, len({patient_id(p) for p, _ in v}))
   ```
   Questions: are the classes still balanced inside each split? Do any patient ids appear in two
   splits? Why does the brief insist on patient-level splitting, and what would happen to your
   accuracy if it were image-level?
5. **TODO(A1.5)** Decide how `torch.backends.cudnn.benchmark` should be set for the baseline
   and for benchmarks, and justify it.
6. **TODO(A1.6)** Read each source file once, top to bottom: `data_loader.py`, `preprocess_cpu.py`,
   `filters_cpu.py`, `model.py`, `train.py`, `evaluate.py`. For each of the five modules A-E of
   the brief, write down which function implements it.
7. **TODO(A1.7)** Run the baseline. It is slow on a CPU, so first check that it works with
   `python src/train.py --config configs/baseline.yaml --device cpu --run baseline_smoke --epochs 1`,
   then run the full baseline once (overnight or on a lab machine) with `python src/train_cpu.py`.
   Record the validation accuracy from `results/baseline/history.json`. Do NOT run
   `src/evaluate.py` yet: the test set is used once per final model, and the baseline's test run
   happens in A7.
8. **TODO(A1.8)** Separate **one-time costs** (`setup_time_s`, the first "cold" epoch) from **repeated costs**
   (steady-state epoch) in `results/baseline/history.json`.

**Deliverable:** baseline execution record (`results/baseline/*.json` plus a short paragraph in
the notebook) and a pipeline diagram in `docs/pipeline.md` (**TODO(A1.9)**, template inside; boxes for each stage with tensor
shape and dtype between boxes, and where data crosses the CPU/GPU boundary in the optimized design).

**TODO(A1.10)** **Checkpoint questions** (answer in the notebook)
- Which stages run once, and which run every epoch?
- How many bytes does one batch occupy as uint8 and as float32 at 224x224?
- What does `drop_last=True` on the train loader change in your throughput numbers?

---

## A2. Profiling and bottleneck identification (days 2-3)

**Goal:** know where the time goes, with evidence, before deciding what to optimize.

1. **TODO(A2.1)** Make `GpuTimer` in `src/utils.py` synchronize CUDA before and after the timed
   region. Write a 5-line experiment that times a large matrix multiplication on the GPU with
   and without synchronization and explain the difference. This is a frequent mistake in
   GPU benchmarks.
2. **TODO(A2.2)** Read `run_epoch` in `src/train.py` and answer the question in its docstring.
3. **TODO(A2.3)** **Predict first.** In the notebook, rank these stages by share of baseline epoch time:
   loading, preprocessing (CLAHE, crop, normalize), forward, backward, optimizer step.
4. **TODO(A2.4)** Run `python benchmarks/profile_baseline.py`. It profiles one CPU epoch with `cProfile`
   (workers forced to 0 so that the profile sees decoding and preprocessing) and writes
   `results/profile_stages.json` and `results/profile_baseline.txt`.
5. **TODO(A2.5)** Run `python benchmarks/benchmark_stages.py --n 64` to get per-stage CPU times on the same
   64 images.
6. **TODO(A2.6)** Rank the bottlenecks. For the top two, compute the Amdahl bound: if stage fraction is `p`
   and it became infinitely fast, the best end-to-end speedup is `1 / (1 - p)`.
7. **TODO(A2.7)** Use one more tool on one function of your choice: `line_profiler`, or `torch.profiler` on
   the CPU training step. Show one thing it revealed that `cProfile` did not.
8. **TODO(A2.8)** Read the CPU utilization and memory log of the baseline run of A1.7
   (`train_cpu.py` writes `results/baseline/resources.csv`; add `--monitor` to any other run).
   Is the CPU saturated? No extra run is needed.

**Deliverable (TODO(A2.9)):** profiling report (a section of your notebook now, section 4 of the final report
later): timing table, a bar chart, ranked priorities, each backed by a file in `results/`.

**TODO(A2.10)** **Checkpoint questions**
- Your prediction vs the measurement: where were you wrong and why?
- Why is `num_workers=0` right for `cProfile` but wrong for a benchmark of the real pipeline?
- A stage takes 18% of the baseline time. What is the largest possible end-to-end speedup from
  accelerating only that stage?

---

## A3. Loading redesign: decoded-image cache (days 3-4)

**Hypothesis to test:** decoding and resizing the same 3,000 images every epoch is redundant work.
Decoding once and reading raw uint8 pixels from a memory-mapped file removes it.

1. **TODO(A3.1)** Implement `build_cache` in `src/data_loader.py`.
2. **TODO(A3.2)** Use the cache inside `DRDataset.load`.
3. **TODO(A3.3)** Build the cache in `make_loaders` when `data.cache_dir` is set.
4. **TODO(A3.4)** `python -m pytest tests/test_data_pipeline.py -v` must pass.
5. **TODO(A3.5)** **Predict, then measure** the loading stage with and without the cache (use
   `benchmarks/benchmark_stages.py`). Separate three situations: first run (cache is built),
   warm cache, and cold operating-system file cache. Report the cache size in MB and compare it
   with the dataset size on disk. If you cannot clear the operating-system file cache (for
   example on Windows), say so in the notebook and report the limitation.
6. **TODO(A3.6)** Write `docs/design_proposal.md` (template inside): for every optimization you
   plan for A4-A6, name the measured bottleneck that motivates it, your hypothesis, the expected
   gain, the cost (memory, disk, complexity), and how you will validate correctness.

**Deliverable:** design proposal (revised architecture, hypotheses, validation plan).

**TODO(A3.7)** **Checkpoint questions**
- When does a cache lose? (Think: 300,000 images, small SSD, augmentation applied before resize.)
- Why must the cache file name depend on the image list and the size?
- Is the cache a CPU optimization, a GPU optimization, or both? Does it still pay off when
  the GPU does the preprocessing?

<details><summary>Hint: build_cache</summary>

`np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(N, H, W, 3))` creates the
file. Fill row `i` from a `ThreadPoolExecutor`, call `flush()`, then `del` the memmap (on Windows
a file that is still mapped cannot be renamed), `os.replace` the temporary
file to its final name and reopen with `np.load(path, mmap_mode="r")`.
</details>

---

## A4. GPU image processing (days 4-6)

**Goal:** reimplement preprocessing, filters and augmentation as batched GPU operations that
match the CPU reference.

1. **TODO(A4.1 to A4.6)** `src/preprocess_gpu.py`: circular crop, normalization, CLAHE,
   augmentation, uint8-to-float conversion, and the full batch pipeline.
2. **TODO(A4.7 to A4.10)** `src/filters_gpu.py`: Gaussian blur, median filter, green channel,
   contrast stretch.
3. **TODO(A4.11)** (optional) For one filter of your choice implement a second GPU version with a different
   tool (for example CuPy `cupyx.scipy.ndimage`, a Numba CUDA kernel, or separable convolutions in
   torch) and compare both against the CPU reference in speed and in numerical difference.
4. **TODO(A4.12)** `python -m pytest tests/test_cpu_gpu_equivalence.py -v` is the specification. Read the
   tolerances in the tests before you write code, and decide whether you agree with them. If not,
   say why in the notebook.
5. **TODO(A4.13)** **Predict, then measure** (write the sweep yourself; batch sizes 1, 8, 32, 64, 128): GPU time per image for CLAHE and
   for the median filter. At which batch size does the GPU overtake the CPU, if ever?
6. **TODO(A4.14)** Measure the host-to-device copy of one uint8 batch and of one float32 batch. How much of the
   GPU preprocessing time is the copy?
7. **TODO(A4.15)** Look at your results: which function got the largest speedup, which the smallest, and why?

**Deliverable:** CPU/GPU comparison of the processing stages with correctness tests.

**TODO(A4.16)** **Checkpoint questions**
- Why can a GPU kernel be slower than OpenCV on a single 224x224 image?
- The median filter with `unfold` uses `ksize^2` times the input memory. What does that limit?
- Your CLAHE is not bit-identical to OpenCV (different color space). Does the difference matter
  for classification? How would you test that, using A6?

<details><summary>Hint: where batching helps</summary>

Everything that is independent per pixel or per image (normalize, crop, flip) becomes one kernel
launch for the whole batch. CLAHE needs one histogram per tile per image: think of all
`N * grid * grid` tiles as one big 2D tensor and use one `scatter_add_`.
</details>

<details><summary>Hint: rotation on the GPU</summary>

`F.affine_grid` takes a (N,2,3) matrix per image. The rotation by angle a is
`[[cos a, -sin a, 0], [sin a, cos a, 0]]`. `grid_sample` with `padding_mode="reflection"`
matches the reflect border of the OpenCV reference.
</details>

---

## A5. End-to-end input pipeline (days 7-8)

**Goal:** a GPU that is not waiting for data. Optimize the whole pipeline, not isolated kernels.

1. **TODO(A5.1)** Add the DataLoader options in `make_loaders`: `pin_memory`,
   `persistent_workers`, `prefetch_factor`, `worker_init_fn`.
2. **TODO(A5.2)** Implement `CudaPrefetcher` in `src/prefetcher.py`. The test
   `test_prefetcher_yields_same_batches_on_device` checks correctness; the profiler (A7) shows
   whether it overlaps.
3. **TODO(A5.3)** `resolve_gpu_preprocess` and `make_gpu_pre` in `src/train.py`, and the
   `gpu_pre` argument of `run_epoch`.
4. **TODO(A5.4)** Asynchronous copies (`non_blocking=True`) in `run_epoch`.
5. **TODO(A5.5)** Wrap the loaders in `CudaPrefetcher` in `main` when `data.prefetch_gpu` is true.
6. **TODO(A5.6, A5.7)** Add the GPU rows and the speedup column to `benchmarks/benchmark_stages.py`.
7. **TODO(A5.8)** Write a small loop (or script) and tune `num_workers` and `batch_size` on your machine: sweep 0, 1, 2, 4, 8
   workers, 3 repeats each, and plot images per second. Where does adding workers stop helping,
   and what limits you (CPU cores, disk, RAM, the GPU)?
8. **TODO(A5.9)** Compare **precomputed** preprocessing (store the final float tensors) with **on-the-fly GPU**
   preprocessing. Count storage and memory for both and decide, with numbers. Ask the same question for
   augmentation (Module D): fixed augmented copies on disk, or random augmentation on the fly?
9. **TODO(A5.10)** Reduce transfers: how many bytes cross the PCIe bus per epoch with CPU preprocessing (float32
   after preprocessing) and with GPU preprocessing (uint8 before)?

**Deliverable:** optimized input pipeline and end-to-end throughput measurements.

**TODO(A5.11)** **Checkpoint questions**
- Why does `non_blocking=True` do nothing useful for a tensor in pageable memory?
- You doubled `num_workers` and nothing changed. Name three possible reasons and one measurement
  for each.
- Why is Windows different from Linux for DataLoader workers (spawn vs fork), and what does that
  change in your code?

---

## A6. Training acceleration and model redesign (days 8-10)

**Goal:** make training fast without losing quality, and show which gain comes from where.

1. **TODO(A6.1)** Mixed precision: `torch.autocast` and `GradScaler` in `run_epoch` and `main`.
2. **TODO(A6.2)** `channels_last` memory format. Measure it; do not assume it helps.
3. **TODO(A6.3)** (optional) `torch.compile` (Linux, needs Triton; on Windows skip it and say so). Report the cold-epoch cost separately.
4. **TODO(A6.4)** Evaluation with the same precision as training.
5. **TODO(A6.5)** `src/train_gpu.py` entry point.
6. **TODO(A6.6)** Add a justified alternative model in `src/model.py` (the brief suggests a
   lightweight residual network or a pretrained model).
7. **TODO(A6.7)** Tune the batch size: throughput, GPU memory (`gpu_peak_mem_mb` in
   `history.json`) and validation accuracy.
8. **TODO(A6.8)** Complete `configs/optimized.yaml` and the single-change variants `configs/exp_*.yaml`.

**TODO(A6.9)** **The required redesign challenge (brief, section 6).** Investigate and report on:
(1) model architecture, (2) input representation (resolution, color, normalization, contrast
enhancement), (3) training strategy (augmentation, regularization, learning-rate schedule),
(4) computational cost of the final design, (5) controlled comparison with identical splits.
Isolate each change. A table with one row per experiment and columns for model, device,
precision, preprocessing, epoch time, memory, validation macro-F1 (`val_f1` in `history.json`) is
the minimum. Keep `data.split_seed` fixed in every config and change only `seed` when you repeat
a run, so that the splits stay identical.

**TODO(A6.10)** **To separate GPU speedup from redesign speedup**, run the same model on CPU and on GPU first
(device effect), then change the model on GPU (model effect), then change the training recipe.

**Deliverable:** optimized training implementation and a controlled accuracy/performance comparison.

**TODO(A6.11)** **Checkpoint questions**
- With AMP, which operations stay in float32 and why?
- Why does a bigger batch size sometimes lower validation accuracy at the same learning rate?
- Is the accuracy difference between two runs larger than the difference between two seeds of the
  same run? How many seeds do you need to say?

---

## A7. Final benchmarking and GPU profiling (days 11-12)

1. **TODO(A7.1)** Aggregate the end-to-end results in `benchmarks/benchmark_end_to_end.py`.
2. **TODO(A7.2 to A7.4)** Implement `benchmarks/profile_gpu.py` and read the trace. Nsight Systems is an optional second view if it is installed. Is the GPU
   idle between steps? How much of the step is the host-to-device copy?
3. **TODO(A7.5)** Run the benchmark protocol: warm-up, synchronization, at least 3 repeats, mean and standard
   deviation, cold vs steady-state, transfers included.
4. **TODO(A7.6)** Ablation: starting from your final configuration, switch off one optimization at a time
   (`configs/exp_<name>_off.yaml`; at least three optimizations) and measure steady-state epoch
   time and validation macro-F1.
5. **TODO(A7.7)** Re-profile the final version. Which stage dominates now? Did the bottleneck move?
6. **TODO(A7.8)** Run `python src/evaluate.py` once for the baseline and once for your final model on the test set.

**Deliverable:** measured end-to-end speedup, ablation table, resource utilization, final metrics.

**TODO(A7.9)** **Checkpoint questions**
- Your stage-level speedup was 30x and the end-to-end speedup was 3x. Use Amdahl's law to explain it.
- Which optimization helped least relative to your expectation, and what does the profiler say?

---

## A8. Figures, tables, report and defense (days 12-14)

1. **TODO(A8.1 to A8.6)** Implement `scripts/make_figures.py`: Figures 1-5 and Tables 2-3 of the
   brief, generated only from files in `results/`.
2. **TODO(A8.7, A8.8)** Write the README sections the brief requires (the tags in `README.md`): objective and team members, environment setup,
   dataset structure, how to run the baseline, the optimized version, profiling, benchmarking and
   evaluation, known issues.
3. **TODO(A8.9)** Write `report/final_report.md` following the required outline (brief, section 8B). Render it to
   PDF with any tool you like (`report/final_report.pdf`, 10-15 pages without appendices).
4. **TODO(A8.10)** In the report, answer every question of section 9 of the brief (performance, classification
   quality, engineering trade-offs) and write the limits and threats to validity of section 10.
5. **TODO(A8.11)** Complete `Table 1` (hardware/software) in the report from `environment/hardware_software.md`.
6. **TODO(A8.12)** Prepare a 15-20 minute defense that answers the six questions of section 8E of the brief.

**Submission checklist**
- [ ] `python -m pytest tests -v` passes (GPU tests pass on a CUDA machine)
- [ ] `python scripts/progress.py` shows no TODO left
- [ ] every number in the report comes from a file in `results/` and the command is in the README
- [ ] same splits, same budgets for every comparison
- [ ] unsuccessful experiments are documented too
- [ ] the test set was used once per final model

---

## Optional advanced challenges

From section 12 of the brief: inference optimization separate from training; resolution vs
accuracy-latency trade-off; batch size vs throughput, latency and GPU memory; a simple performance
model predicting when the CPU beats the GPU; NVIDIA DALI for GPU decoding; model compilation and
operator fusion; CPU vs GPU energy (only if you have a reliable power measurement).

## References

Read these before you decide what to optimize.

- PyTorch, [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- PyTorch, [A guide on good usage of `non_blocking` and `pin_memory()`](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- PyTorch, [Data Loading Optimization](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
- PyTorch, [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- PyTorch, [Profiling to understand `torch.compile` performance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
- PyTorch, [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) (streams, asynchronous execution)
- PyTorch, [Automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- NVIDIA, [DALI user guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) (GPU image decoding pipelines)
- [CuPy documentation](https://docs.cupy.dev/) and [Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html)

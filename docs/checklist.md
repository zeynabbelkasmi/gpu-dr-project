# Checklist for steps without a file of their own

Steps that edit a file carry their `TODO(A<n>.<m>)` tag in that file. The steps below are setup,
runs, measurements and written answers, so they are tracked here. When a step is done and its
evidence exists (a notebook entry or a file in `results/`), delete the tag from the line.
`python scripts/progress.py` counts the tags that remain. The full instructions are in
`ASSIGNMENTS.md`.

Evidence for every [x] step: `report/week1_report.md` (A1, A2) and `report/week2_report.md` (A3, A4).

## A1. Baseline execution and verification

- [x] A1.1 Create the environment, download FDR26, count the images per class folder.
- [x] A1.4 Run the split check from `ASSIGNMENTS.md`, paste the output in the notebook, answer its questions.
- [x] A1.6 Read the six source files and write which function implements each of modules A to E.
- [x] A1.7 Smoke-run the baseline for one epoch, then run it fully once. Note the validation accuracy.
- [x] A1.8 Separate one-time costs from repeated costs in `results/baseline/history.json`.
- [x] A1.10 Answer the A1 checkpoint questions in the notebook.

## A2. Profiling and bottleneck identification

- [x] A2.3 Write your ranking of stage shares before you measure anything.
- [x] A2.4 Run `benchmarks/profile_baseline.py`.
- [x] A2.5 Run `benchmarks/benchmark_stages.py --n 64`.
- [x] A2.6 Rank the bottlenecks and compute the Amdahl bound for the top two.
- [x] A2.7 Use a second profiler on one function and report what it showed that `cProfile` did not.
- [x] A2.8 Read `results/baseline/resources.csv` and decide whether the CPU is saturated.
- [x] A2.9 Write the profiling report (timing table, bar chart, ranked priorities).
- [x] A2.10 Answer the A2 checkpoint questions.

## A3. Loading redesign: decoded-image cache

- [x] A3.4 Make `tests/test_data_pipeline.py` pass.
- [x] A3.5 Predict, then measure loading without cache, with a first-run cache build, and with a warm cache. Report the cache size in MB.
- [x] A3.7 Answer the A3 checkpoint questions.

## A4. GPU image processing

- [x] A4.12 Read the tolerances in `tests/test_cpu_gpu_equivalence.py`, run it, and say in the notebook whether you agree with them.
- [x] A4.13 Predict, then sweep batch sizes 1, 8, 32, 64, 128 for CLAHE and the median filter on CPU and GPU.
- [x] A4.14 Measure the host-to-device copy of one uint8 batch and one float32 batch.
- [x] A4.15 Say which function gained most and which least, and why.
- [x] A4.16 Answer the A4 checkpoint questions.

## A5. End-to-end input pipeline

- TODO(A5.9) Compare precomputed preprocessing with on-the-fly GPU preprocessing in storage and memory, and decide with numbers.
- TODO(A5.10) Compute the bytes per epoch that cross PCIe with CPU preprocessing and with GPU preprocessing.
- TODO(A5.11) Answer the A5 checkpoint questions.

## A6. Training acceleration and model redesign

- TODO(A6.8) Complete `configs/optimized.yaml` and write the single-change `configs/exp_*.yaml` files.
- TODO(A6.9) Work through the five parts of the redesign challenge and fill the experiment table (model, device, precision, preprocessing, epoch time, memory, validation macro-F1).
- TODO(A6.10) Separate the device effect, the model effect and the training-recipe effect with controlled runs.
- TODO(A6.11) Answer the A6 checkpoint questions.

## A7. Final benchmarking and GPU profiling

- TODO(A7.5) Run the benchmark protocol: warm-up, synchronization, 3 or more repeats, mean and standard deviation, cold and steady epochs.
- TODO(A7.6) Run the ablation, one optimization off at a time (at least three), with `configs/exp_<name>_off.yaml`.
- TODO(A7.7) Re-profile the final version and say whether the bottleneck moved.
- TODO(A7.8) Run `src/evaluate.py` once for the baseline and once for the final model.
- TODO(A7.9) Answer the A7 checkpoint questions.

## A8. Figures, tables, report and defense

- TODO(A8.9) Write `report/final_report.md` and render `report/final_report.pdf`.
- TODO(A8.10) Answer the questions of section 9 of the brief and write the limits of section 10.
- TODO(A8.11) Complete Table 1 in the report.
- TODO(A8.12) Prepare the defense and answer the six questions of section 8E of the brief.

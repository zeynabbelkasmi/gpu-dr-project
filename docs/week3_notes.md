# Week 3 notes (A5 + A6): what was implemented, how to run it, checkpoint answers

## Implemented

| TODO | File | What |
|---|---|---|
| A5.1 | `src/data_loader.py` | `pin_memory` (CUDA only), `persistent_workers`, `prefetch_factor`, `worker_init_fn` (last three only when `num_workers > 0`) |
| A5.2 | `src/prefetcher.py` | `CudaPrefetcher`: copy of batch i+1 on a side stream while batch i is computed; `wait_stream` + `record_stream` |
| A5.3 | `src/train.py` | `resolve_gpu_preprocess`, `make_gpu_pre`, `gpu_pre` argument of `run_epoch` (augment on train only) |
| A5.4 | `src/train.py` | `non_blocking=True` copies of `x` and `y` |
| A5.5 | `src/train.py` | loaders wrapped in `CudaPrefetcher` when `data.prefetch_gpu` is true on CUDA |
| A5.6, A5.7 | `benchmarks/benchmark_stages.py` | already done in Weeks 1-2 |
| A5.8, A5.10 | `benchmarks/sweep_loader.py` | workers x batch-size sweep (3 repeats, plot) and PCIe bytes per epoch |
| A6.1 | `src/train.py` | `torch.autocast` + `GradScaler` |
| A6.2 | `src/train.py` | `channels_last` for model and input batch |
| A6.3 | `src/train.py` | optional `torch.compile`; the checkpoint saves the original model's `state_dict` |
| A6.4 | `src/evaluate.py` | same precision and memory format as training |
| A6.5 | `src/train_gpu.py` | entry point (`configs/optimized.yaml`, CUDA, run `optimized`) |
| A6.6 | `src/model.py` | `resnet18` (ImageNet weights); `python -m src.model` prints parameters and FLOPs |
| A6.8 | `configs/optimized.yaml`, `configs/exp_*.yaml` | final config and one single-change config per optimization |
| A6.9, A6.10 | `scripts/week3_experiments.py` | runs CPU -> GPU -> one change at a time -> model -> all, writes `results/week3_table.csv` |

Parameters and FLOPs (forward, one 224x224 image): reference CNN 0.62 M / 1.01 GFLOPs,
ResNet-18 11.18 M / 3.63 GFLOPs.

Still to do with measurements (needs the GPU): A5.8 sweep, A5.9 decision, A6.7 batch-size
tuning, the A6.9 table and its analysis, and the final choice of `channels_last` and of the
model in `optimized.yaml`.

## How to run (Colab, T4)

```
python -m pytest tests/test_data_pipeline.py -v          # prefetcher test must pass on CUDA
python benchmarks/sweep_loader.py --config configs/optimized.yaml --workers 0,1,2,4 --batch-sizes 32,64,128
python scripts/week3_experiments.py                      # all runs + results/week3_table.csv
python src/train_gpu.py                                  # final optimized run
```

## A5.11 checkpoint answers

- **Why does `non_blocking=True` do nothing useful for pageable memory?** The GPU DMA engine
  can only read page-locked memory. For a pageable tensor the driver first copies it into a
  pinned staging buffer, and that copy is done by the CPU, synchronously, so the call returns
  only when the data has been staged: nothing overlaps. With `pin_memory=True` the batch is
  already pinned and the copy is queued on the stream while the CPU continues.
- **Doubling `num_workers` changed nothing: three reasons and a measurement for each.**
  (1) The GPU (or the training step) is the bottleneck, not the loader: measure the loader alone
  with `sweep_loader.py` and compare with `train_imgs_per_s`. (2) Not enough CPU cores
  (Colab has 2): check `os.cpu_count()` and CPU utilization in `resources.csv`. (3) Disk/I-O or
  memory bound (cache read, page cache): compare warm vs cold cache throughput, or watch RAM.
- **Windows vs Linux workers.** Linux forks workers (they inherit the parent memory, start fast);
  Windows spawns them (a new Python process that re-imports modules and receives a pickled
  dataset). Consequences in our code: the dataset must be picklable (the memmap is sent by file
  name, `DRDataset.__getstate__`), worker functions must be top-level (`_worker_init`, not a
  lambda), the entry point needs `if __name__ == "__main__"`, and `persistent_workers=True`
  avoids paying the spawn cost every epoch.

## A6.11 checkpoint answers

- **With AMP, which operations stay in float32 and why?** Autocast runs matmuls and convolutions
  in float16 but keeps numerically sensitive ops in float32: reductions and softmax/log-softmax,
  hence the cross-entropy loss, normalizations, exp/log, and the weights and optimizer state
  (master weights stay float32; AdamW updates them in float32). The GradScaler multiplies the
  loss so that small gradients do not underflow in float16, and skips steps with inf/NaN.
- **Why can a bigger batch lower validation accuracy at the same learning rate?** Fewer
  optimizer steps per epoch for the same epoch budget, and less gradient noise (less implicit
  regularization, sharper minima). Usual remedy: scale the learning rate with the batch size and
  use warm-up.
- **Is the accuracy difference between two runs larger than between two seeds?** Only repeating
  the same configuration with several `seed` values (split unchanged) answers it: report mean
  +/- std over at least 3 seeds and call a difference real only if it is clearly larger than
  that spread.

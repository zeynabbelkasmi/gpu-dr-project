"""Optimized GPU pipeline entry point: configs/optimized.yaml on CUDA -> results/optimized/.

TODO(A6.5): once configs/optimized.yaml and the GPU path in src/train.py work, call
src.train.main with device "cuda", run="optimized" and monitor=True, mirroring
src/train_cpu.py. Keep the file as small as train_cpu.py: the logic lives in train.py.
"""

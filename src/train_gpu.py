"""Optimized GPU pipeline entry point: configs/optimized.yaml on CUDA -> results/optimized/."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import main

if __name__ == "__main__":
    main("configs/optimized.yaml", "cuda", run="optimized", monitor=True)

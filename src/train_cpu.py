"""Reference CPU baseline: configs/baseline.yaml on CPU -> results/baseline/."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import main

if __name__ == "__main__":
    main("configs/baseline.yaml", "cpu", run="baseline", monitor=True)

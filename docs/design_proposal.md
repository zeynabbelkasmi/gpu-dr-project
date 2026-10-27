# Design proposal (A3.6)

The full proposal, with every row tied to a measurement in `results/`, is section 2 of
[`report/week2_report.md`](../report/week2_report.md). Summary:

| # | Optimization | Measured bottleneck | Expected | Measured |
|---|---|---|---|---|
| 1 | uint8 decoded-image cache | loading p = 0.654 (`results/profile_stages.json`) | loading -> ~0, end to end <= 2.89x | 164x on loading, **1.53x end to end**, identical metrics |
| 2 | batched GPU CLAHE (LUT LAB, exact) | LAB conversion = 77 % of CLAHE (`results/week1/line_profile.txt`) | >= 10x | 10.5x (B=128), 14.6x (B=32), bit-exact |
| 3 | fused crop + normalize, cached constants | boolean indexing 87 % of crop | > 50x | 148x |
| 4 | uint8 host-to-device copies | float32 batches 4 B/value | ~4x on the copy | 3.9x |
| 5 | custom CuPy median kernel | median 1.0 ms/img CPU, `unfold` 25x memory | > library | 22.4x (vs 18.1x cupyx, 3.0x torch) |
| 6 | batched affine augmentation | 0.48 ms/img CPU | > 5x | 5.3x |

## Revised architecture

See `docs/pipeline.md` (second diagram): decode once into the cache, uint8 batches cross PCIe,
everything after the copy stays on the GPU.

## Order of experiments

1 alone and end to end first (largest Amdahl bound), then 2-6 as stage benchmarks on identical
images; their end-to-end effect is measured once training runs on the GPU (Week 3).

## Risks

Launch overhead at small batch (crossover between B = 1 and 8), laptop GPU power states (measured,
mitigated by `gpu_warmup`), 96 MiB of VRAM for the LUTs, stale cache (hashed file name).

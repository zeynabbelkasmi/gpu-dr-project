# Pipeline diagram

Shapes are per batch (B = 32, 224x224). "once" = runs once per run, "epoch" = every epoch.

## CPU baseline (`configs/baseline.yaml`, `src/train_cpu.py`)

```
 splits/splits.csv + metadata .xlsx      [once]  get_splits -> 2099 / 450 / 450 (path, label)
              |
              v
 [A] DRDataset.load  (DataLoader worker x2)  [epoch, per image]
     disk read      0.65 MB JPEG
     PIL decode     2576x1958x3 uint8  (15 MB intermediate frame)
     PIL resize     224x224x3 uint8   (BILINEAR, antialiased)
              |   224x224x3 uint8  ->  astype(float32)/255   224x224x3 float32 (0.6 MB)
              v
 [B] CpuTransform -> preprocess_image   [epoch, per image]
     apply_clahe:  float->uint8, RGB->LAB, CLAHE(L), LAB->RGB, uint8->float
     circular_crop: mask rebuilt per image, boolean assignment
     normalize:     (x - mean) / std
     transpose      -> 3x224x224 float32
              |   default_collate:  B x 3 x 224 x 224 float32 (19.3 MB)
              v
 [D] augmentation  (off in the baseline: preprocess.augment absent)
 [C] filters       (not part of the training path; benchmarked as a standalone stage)
              |
              v
 [E] ReferenceCNN forward -> CrossEntropy -> backward -> AdamW.step   [epoch, per batch]
     logits B x 5 float32
              |
              v
     validation pass (same transform, no grad) -> val acc / macro-F1 -> best.pt  [epoch]
     src/evaluate.py on test                                                     [once per final model]
```

## Week 2 design (GPU image processing) — where data crosses the CPU/GPU boundary

```
 [once]  build_cache: decode + resize every image ONCE -> data/cache/u8_224x224_<N>_<md5>.npy
         (N x 224 x 224 x 3 uint8, memory-mapped, bit-exact with DRDataset.load)
              |
 [epoch] DataLoader workers: memmap row -> 3x224x224 uint8 (raw_uint8=True), no decode
              |   B x 3 x 224 x 224 uint8, pinned (4.8 MB per batch)
 ======== PCIe: one host-to-device copy per batch, 1 byte per value ========
              v
 GPU     to_float_bchw (uint8 -> float32 on device, true /255)
         [augment_gpu: flip folded into one affine grid_sample, train only]
         clahe_gpu: LUT RGB->LAB gather, batched CLAHE (1 scatter_add_ for all B*64 tiles), LUT LAB->RGB
         crop_normalize_fused: one addcmul with cached (mask/std, -mean/std)
              |   B x 3 x 224 x 224 float32, already on the device
              v
         ReferenceCNN (Week 3)
```

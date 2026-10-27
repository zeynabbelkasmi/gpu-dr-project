import os
import numpy as np
import pytest
import torch
import yaml

torch.manual_seed(0)

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def sample_batch(n=4, h=224, w=224):
    rng = np.random.default_rng(1)
    imgs = rng.uniform(0.2, 0.9, (n, h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(h, w) // 2 - 2) ** 2
    imgs *= mask[..., None]
    return imgs


@cuda
def test_circular_crop_equivalence():
    from src import preprocess_cpu, preprocess_gpu
    imgs = sample_batch(2)
    cpu = np.stack([preprocess_cpu.circular_crop(i) for i in imgs])
    gpu = preprocess_gpu.circular_crop_gpu(
        torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()).permute(0, 2, 3, 1).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_normalize_equivalence():
    from src import preprocess_cpu, preprocess_gpu
    imgs = sample_batch(2)
    cpu = np.stack([preprocess_cpu.normalize(i) for i in imgs])
    gpu = preprocess_gpu.normalize_gpu(
        torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()).permute(0, 2, 3, 1).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_clahe_equivalence_within_tolerance():
    """GPU CLAHE is a reimplementation; require structural equivalence.

    Tolerance is deliberately loose (correlation on luminance) because the
    CPU (OpenCV) and GPU (torch) histogram pipelines differ in binning detail.
    """
    from src import preprocess_cpu, preprocess_gpu
    imgs = sample_batch(2)
    for i in range(2):
        cpu = preprocess_cpu.apply_clahe(imgs[i]).mean(-1)
        gpu = preprocess_gpu.clahe_gpu(
            torch.from_numpy(imgs[i]).permute(2, 0, 1)[None].cuda()
        )[0].permute(1, 2, 0).cpu().numpy().mean(-1)
        assert np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1] > 0.95


@cuda
def test_gaussian_filter_equivalence():
    from src import filters_cpu, filters_gpu
    img = sample_batch(1)[0]
    cpu = filters_cpu.gaussian_blur(img, ksize=5, sigma=1.0)
    gpu = filters_gpu.gaussian_blur_gpu(
        torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    )[0].permute(1, 2, 0).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=2e-2)


@cuda
def test_median_filter_equivalence():
    from src import filters_cpu, filters_gpu
    img = sample_batch(1)[0]
    cpu = filters_cpu.median_filter(img, ksize=5)
    gpu = filters_gpu.median_filter_gpu(
        torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    )[0].permute(1, 2, 0).cpu().numpy()
    frac_diff = (np.abs(cpu - gpu) > (1.0 / 255)).mean()
    assert frac_diff < 0.02               # <=2% pixels may differ by one gray level


@cuda
def test_clahe_batched_matches_single_image():
    """Batched CLAHE must give each image the same result as processing it alone."""
    from src import preprocess_gpu
    t = torch.from_numpy(sample_batch(3)).permute(0, 3, 1, 2).cuda()
    batched = preprocess_gpu.clahe_gpu(t)
    for i in range(3):
        single = preprocess_gpu.clahe_gpu(t[i:i + 1])
        assert torch.allclose(batched[i:i + 1], single, atol=1e-5)


@cuda
def test_contrast_stretch_equivalence():
    from src import filters_cpu, filters_gpu
    img = sample_batch(1)[0]
    cpu = filters_cpu.contrast_stretch(img)
    gpu = filters_gpu.contrast_stretch_gpu(
        torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    )[0].permute(1, 2, 0).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-3)


@cuda
def test_uint8_batch_pipeline_matches_cpu_without_clahe():
    """Full GPU batch path from uint8 (as the loader delivers it) equals the CPU
    path exactly when only the deterministic steps (crop, normalize) are on."""
    from src import preprocess_cpu, preprocess_gpu
    cfg = {"preprocess": {"clahe": False, "circular_crop": True, "normalize": True, "gamma": 1.0}}
    u8 = (sample_batch(2) * 255).astype(np.uint8)
    cpu = np.stack([preprocess_cpu.preprocess_image(i.astype(np.float32) / 255.0, cfg) for i in u8])
    gpu = preprocess_gpu.preprocess_batch_gpu(
        torch.from_numpy(u8).permute(0, 3, 1, 2).contiguous(), cfg).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_augment_gpu_shape_range_and_per_image_randomness():
    from src import preprocess_gpu
    torch.manual_seed(0)
    t = torch.from_numpy(sample_batch(8)).permute(0, 3, 1, 2).cuda()
    out = preprocess_gpu.augment_gpu(t)
    assert out.shape == t.shape
    assert out.min() >= 0 and out.max() <= 1 + 1e-5
    assert not torch.allclose(out, t)
    # different images must not all receive the same transform
    assert not torch.allclose(out[0] - t[0], out[1] - t[1])


# ---------------------------------------------------------------------------------------------
# Stricter checks added in Week 2: the implementation is exact where the CPU reference is
# integer/uint8 based, so the tests demand bit-exactness there instead of a loose tolerance.
# ---------------------------------------------------------------------------------------------
try:
    import cupy  # noqa: F401
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
cupy_only = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")


def _nchw(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).cuda()


def _nhwc(t):
    return t.permute(0, 2, 3, 1).cpu().numpy()


@cuda
@pytest.mark.parametrize("hw", [(224, 224), (256, 256), (299, 299), (224, 320), (100, 100), (31, 47)])
def test_circle_mask_matches_cv2_raster(hw):
    import cv2
    from src import preprocess_gpu
    h, w = hw
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), max(min(h, w) // 2 - 2, 1), 255, -1)
    gpu = preprocess_gpu.circle_mask(h, w, "cuda")[0, 0].cpu().numpy()
    assert np.array_equal(m > 0, gpu > 0)


@cuda
def test_div255_is_true_division():
    from src import preprocess_gpu
    u8 = np.arange(256, dtype=np.uint8)
    cpu = u8.astype(np.float32) / 255.0
    gpu = preprocess_gpu.div255_(torch.from_numpy(u8).cuda().float()).cpu().numpy()
    assert np.array_equal(cpu, gpu)


def _fundus_like(n, h, w, seed=3):
    """Smooth images with structure (CLAHE on pure noise barely exercises clipping)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    out = []
    for _ in range(n):
        base = 0.5 + 0.3 * np.sin(rng.uniform(3, 9) * xx + rng.uniform(0, 6)) * np.cos(rng.uniform(3, 9) * yy)
        img = np.stack([base * c for c in rng.uniform(0.5, 1.0, 3)], -1) + rng.normal(0, 0.03, (h, w, 3))
        out.append(np.clip(img, 0, 1).astype(np.float32))
    return np.stack(out)


@cuda
@pytest.mark.parametrize("hw", [(224, 224), (230, 250)])   # 230x250: OpenCV's reflect-101 padding path
def test_clahe_lab_is_bit_exact(hw):
    from src import preprocess_cpu, preprocess_gpu
    imgs = _fundus_like(3, *hw)
    cpu = np.stack([preprocess_cpu.apply_clahe(i) for i in imgs])
    gpu = _nhwc(preprocess_gpu.clahe_gpu(_nchw(imgs)))
    assert np.array_equal(cpu, gpu)


@cuda
def test_median_filter_is_bit_exact():
    from src import filters_cpu, filters_gpu
    imgs = _fundus_like(2, 224, 224)
    cpu = np.stack([filters_cpu.median_filter(i) for i in imgs])
    assert np.array_equal(cpu, _nhwc(filters_gpu.median_filter_gpu(_nchw(imgs))))


@cuda
@cupy_only
@pytest.mark.parametrize("fn", ["median_filter_cupy_kernel", "median_filter_cupy_ndimage"])
def test_median_filter_cupy_variants_are_bit_exact(fn):
    from src import filters_cpu, filters_gpu
    imgs = _fundus_like(2, 224, 224)
    cpu = np.stack([filters_cpu.median_filter(i) for i in imgs])
    assert np.array_equal(cpu, _nhwc(getattr(filters_gpu, fn)(_nchw(imgs))))


@cuda
@pytest.mark.parametrize("fuse,atol", [(False, 0.0), (True, 1e-6)])
def test_full_pipeline_with_clahe_matches_cpu(fuse, atol):
    """The GPU pipeline as the training loop will call it: uint8 NCHW in, normalized out."""
    from src import preprocess_cpu, preprocess_gpu
    u8 = (_fundus_like(4, 224, 224) * 255).astype(np.uint8)
    cpu = preprocess_cpu.batch_preprocess_cpu(u8.astype(np.float32) / 255.0, CFG)
    gpu = preprocess_gpu.preprocess_batch_gpu(
        torch.from_numpy(u8).permute(0, 3, 1, 2).contiguous(), CFG, fuse=fuse).cpu().numpy()
    assert np.abs(cpu - gpu).max() <= atol


@cuda
def test_augment_gpu_matches_cv2_for_fixed_parameters():
    import cv2
    from src import preprocess_gpu
    imgs = _fundus_like(4, 224, 224)
    flips, angles = [False, True, False, True], [7.0, -9.5, 0.0, 3.3]
    cpu = []
    for img, fl, a in zip(imgs, flips, angles):
        img = img[:, ::-1] if fl else img
        m = cv2.getRotationMatrix2D((112, 112), a, 1.0)
        cpu.append(cv2.warpAffine(np.ascontiguousarray(img), m, (224, 224),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    gpu = _nhwc(preprocess_gpu.augment_gpu(_nchw(imgs), flip=torch.tensor(flips, device="cuda"),
                                           angle=torch.tensor(angles, device="cuda")))
    # OpenCV quantises bilinear weights to 1/32 pixel; geometry errors would be >> 1e-2
    assert np.abs(np.stack(cpu) - gpu).max() < 1e-2


@cuda
def test_contrast_stretch_beyond_quantile_size_limit():
    """129 images x 3 x 224 x 224 > 2^24 elements: torch.quantile would refuse this batch."""
    from src import filters_cpu, filters_gpu
    t = torch.rand(129, 3, 224, 224, device="cuda")
    out = filters_gpu.contrast_stretch_gpu(t)
    ref = filters_cpu.contrast_stretch(t[7].permute(1, 2, 0).cpu().numpy())
    assert np.allclose(_nhwc(out[7:8])[0], ref, atol=1e-3)

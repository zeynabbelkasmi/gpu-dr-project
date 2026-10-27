import os
import numpy as np
import yaml
import cv2

from src import preprocess_cpu

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")))


def sample_image(h=224, w=224):
    rng = np.random.default_rng(0)
    img = rng.uniform(0.2, 0.9, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(h, w) // 2 - 2) ** 2
    img *= mask[..., None]
    return img


def test_circular_crop_zeroes_outside():
    out = preprocess_cpu.circular_crop(sample_image())
    h, w = out.shape[:2]
    assert out[2, 2].sum() == 0            # corner outside circle
    assert out[h // 2, w // 2].sum() > 0   # centre kept


def test_clahe_preserves_range_and_shape():
    img = sample_image()
    out = preprocess_cpu.apply_clahe(img)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1


def test_preprocess_image_output_format():
    out = preprocess_cpu.preprocess_image(sample_image(), CFG)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_gaussian_blur_is_lowpass():
    from src import filters_cpu
    img = sample_image()
    out = filters_cpu.gaussian_blur(img, ksize=5, sigma=1.0)
    assert out.shape == img.shape
    # sample_image is i.i.d. noise, so test low-pass behaviour directly:
    # blurring must shrink pixel-to-pixel variation inside the fundus
    inner = (slice(20, -20), slice(20, -20))
    d_in = np.abs(np.diff(img[inner], axis=1)).mean()
    d_out = np.abs(np.diff(out[inner], axis=1)).mean()
    assert d_out < 0.5 * d_in
    assert out[inner].std() < img[inner].std()


def test_median_filter_removes_salt_noise():
    from src import filters_cpu
    img = sample_image()
    noisy = img.copy()
    noisy[::17, ::13] = 1.0                  # salt noise
    out = filters_cpu.median_filter(noisy, ksize=5)
    assert out.shape == img.shape
    assert (out <= 0.999).all()              # impulses removed


def test_cpu_transform_is_picklable():
    """DataLoader workers use spawn on Windows/macOS: the transform must pickle."""
    import pickle
    from src.data_loader import get_preprocess_transform
    tf = pickle.loads(pickle.dumps(get_preprocess_transform(CFG)))
    assert tf(sample_image()).shape == (3, 224, 224)


def test_cpu_augment_shape_and_range():
    out = preprocess_cpu.augment_image(sample_image(), np.random.default_rng(0))
    assert out.shape == (224, 224, 3)
    assert out.min() >= 0 and out.max() <= 1

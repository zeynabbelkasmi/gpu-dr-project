"""CPU reference preprocessing: CLAHE, circular crop, gamma, normalize."""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def resize_image(img, size):
    return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)


def circular_crop(img):
    """Zero out everything outside the fundus circle."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), max(min(h, w) // 2 - 2, 1), 255, -1)
    out = img.copy()
    out[mask == 0] = 0
    return out


def apply_clahe(img, clip=2.0, grid=8):
    """CLAHE on the L channel of LAB. Input RGB float [0,1], same range out."""
    u8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


def apply_gamma(img, gamma=1.0):
    if abs(gamma - 1.0) < 1e-6:
        return img
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT((img * 255).astype(np.uint8), lut).astype(np.float32) / 255.0


def normalize(img):
    return (img - MEAN) / STD


def preprocess_image(img, cfg):
    """Full CPU preprocessing for one RGB float [0,1] image -> CHW array."""
    p = cfg["preprocess"]
    x = img
    if p.get("gamma", 1.0) != 1.0:
        x = apply_gamma(x, p["gamma"])
    if p.get("clahe"):
        x = apply_clahe(x)
    if p.get("circular_crop"):
        x = circular_crop(x)
    if p.get("normalize"):
        x = normalize(x)
    return x.transpose(2, 0, 1).astype(np.float32)


def batch_preprocess_cpu(images, cfg):
    return np.stack([preprocess_image(i, cfg) for i in images])


def augment_image(img, rng):
    """CPU reference augmentation (same ops as preprocess_gpu.augment_gpu):
    random horizontal flip + rotation in [-10, 10] degrees, reflect padding."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), 1.0)
    return cv2.warpAffine(np.ascontiguousarray(img), m, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

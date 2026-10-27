"""CPU reference filters: Gaussian, median, green channel, contrast stretch."""
import cv2
import numpy as np


def gaussian_blur(img, ksize=5, sigma=1.0):
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, ksize=5):
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur((img * 255).astype(np.uint8), k).astype(np.float32) / 255.0


def green_channel(img):
    return img[..., 1]


def contrast_stretch(img):
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

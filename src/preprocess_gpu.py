"""GPU (PyTorch/CUDA) preprocessing: batched tensor ops, no CPU round-trips.

Assignment A4. Every function works on a whole batch (N,3,H,W) at once and matches
the CPU reference in src/preprocess_cpu.py (tests/test_cpu_gpu_equivalence.py,
benchmarks/benchmark_gpu_processing.py). No Python loop over images or tiles.

Design rules (Week 2 report, section 3):
  - constants (circle mask, mean/std, CLAHE index maps, colour LUTs) are built once per
    (shape, device) and cached in _CONST, so the steady state allocates only activations;
  - the pipeline owns the tensor produced by to_float_bchw, so the later steps run in place
    (crop + normalize are fused into a single addcmul kernel);
  - the uint8 -> float conversion happens on the device, after a 1-byte-per-value copy.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocess_cpu import MEAN, STD

_CONST = {}   # (name, *shape, device, dtype) -> tensor, built once per process


def _const(key, build):
    t = _CONST.get(key)
    if t is None:
        t = _CONST[key] = build()
    return t


def div255_(x):
    """In-place x / 255 with true IEEE division, bit-identical to numpy's `u8 / 255.0`.

    `x.div_(255.0)` with a Python scalar is lowered by PyTorch to a multiplication by the
    reciprocal (1/255 rounded), which is 1 ulp off for ~25% of the 256 values. Dividing by a
    0-dim device tensor keeps the real division.
    """
    return x.div_(_const(("255", str(x.device), x.dtype),
                         lambda: torch.tensor(255.0, device=x.device, dtype=x.dtype)))


# ---------------------------------------------------------------------------------------------
# Per-pixel steps: crop, normalize, gamma
# ---------------------------------------------------------------------------------------------

def circle_mask(h, w, device, dtype=torch.float32):
    """(1,1,H,W) fundus mask: centre (W//2, H//2), radius max(min(H,W)//2 - 2, 1), built
    once per (H, W, device) with arange + meshgrid and cached.

    Same pixels as cv2.circle(..., thickness=-1) in preprocess_cpu.circular_crop: checked at
    7 sizes in tests/test_cpu_gpu_equivalence.py::test_circle_mask_matches_cv2_raster.
    """
    def build():
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                indexing="ij")
        r = max(min(h, w) // 2 - 2, 1)
        return (((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= r * r).to(dtype)[None, None]
    return _const(("circle", h, w, str(device), dtype), build)


def circular_crop_gpu(t, inplace=False):
    """Zero everything outside the fundus circle. t: (N,3,H,W) float."""
    m = circle_mask(t.shape[-2], t.shape[-1], t.device, t.dtype)
    return t.mul_(m) if inplace else t * m


def _mean_std(device, dtype):
    return (_const(("mean", str(device), dtype),
                   lambda: torch.as_tensor(MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)),
            _const(("std", str(device), dtype),
                   lambda: torch.as_tensor(STD, device=device, dtype=dtype).view(1, 3, 1, 1)))


def normalize_gpu(t, inplace=False):
    """(t - MEAN) / STD with cached (1,3,1,1) constants: same float32 ops as the CPU."""
    mean, std = _mean_std(t.device, t.dtype)
    return t.sub_(mean).div_(std) if inplace else (t - mean) / std


def crop_normalize_fused(t, out=None):
    """circular_crop then normalize in ONE kernel: (t*m - mean)/std = t*(m/std) + (-mean/std).

    Reads and writes the batch once instead of three times (crop, sub, div). Not bit-exact
    (different rounding order): max error ~1e-6, below the 1e-5 tolerance of the tests.
    """
    h, w = t.shape[-2:]
    mean, std = _mean_std(t.device, t.dtype)
    scale = _const(("cn_scale", h, w, str(t.device), t.dtype),
                   lambda: circle_mask(h, w, t.device, t.dtype) / std)
    bias = _const(("cn_bias", str(t.device), t.dtype), lambda: -mean / std)
    return torch.addcmul(bias, t, scale, out=out)


def gamma_gpu(t, gamma=1.0):
    """Same 8-bit LUT as preprocess_cpu.apply_gamma, applied with one gather."""
    if abs(gamma - 1.0) < 1e-6:
        return t
    lut = _const(("gamma", float(gamma), str(t.device), t.dtype), lambda: (
        torch.from_numpy((np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8))
        .to(t.device).to(t.dtype) / 255.0))           # built once on the host side, exactness irrelevant
    return lut[(t * 255).to(torch.uint8).long()]


# ---------------------------------------------------------------------------------------------
# CLAHE (OpenCV semantics, batched)
# ---------------------------------------------------------------------------------------------

def _lab_luts(device):
    """RGB->LAB and LAB->RGB tables for all 256^3 uint8 triples, generated ONCE with OpenCV.

    OpenCV's 8-bit Lab conversions are fixed-point code with internal tables; re-deriving
    them in float gives +/-1 level differences. A 16.7M-entry table (48 MiB each, 96 MiB of
    VRAM in total) turns each conversion into one gather and makes it bit-exact.
    """
    def build():
        v = np.arange(256, dtype=np.uint8)
        cube = np.stack(np.meshgrid(v, v, v, indexing="ij"), -1).reshape(4096, 4096, 3)
        rgb2lab = cv2.cvtColor(cube, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        lab2rgb = cv2.cvtColor(cube, cv2.COLOR_LAB2RGB).reshape(-1, 3)
        return (torch.from_numpy(rgb2lab).to(device), torch.from_numpy(lab2rgb).to(device))
    return _const(("lab_luts", str(device)), build)


def _ones(n, device):
    """A stride-0 view of a cached int64 one: scatter_add_ source without allocating n values."""
    return _const(("one", str(device)), lambda: torch.ones(1, dtype=torch.long, device=device)).expand(n)


def _clahe_geometry(h, w, grid, device):
    """Index maps and weights shared by every batch of size (h, w) (OpenCV clahe.cpp)."""
    def build():
        # OpenCV pads bottom/right with BORDER_REFLECT_101 when h or w is not a multiple of grid
        ph, pw = (grid - h % grid, grid - w % grid) if (h % grid or w % grid) else (0, 0)
        th, tw = (h + ph) // grid, (w + pw) // grid

        def reflect101(n, size):
            i = np.arange(n)
            return np.where(i < size, i, 2 * (size - 1) - i)
        rows, cols = reflect101(h + ph, h), reflect101(w + pw, w)
        tile_of = (np.arange(h + ph) // th)[:, None] * grid + (np.arange(w + pw) // tw)[None, :]

        # bilinear interpolation between the 4 nearest tile centres, float32 as in OpenCV
        def axis(n, t):
            f = np.arange(n, dtype=np.float32) * np.float32(1.0 / np.float32(t)) - np.float32(0.5)
            i1 = np.floor(f).astype(np.int64)
            a = (f - i1.astype(np.float32)).astype(np.float32)
            return np.maximum(i1, 0), np.minimum(i1 + 1, grid - 1), a, (np.float32(1) - a)
        ty1, ty2, ya, ya1 = axis(h, th)
        tx1, tx2, xa, xa1 = axis(w, tw)
        base = lambda ty, tx: ((ty[:, None] * grid + tx[None, :]) * 256).reshape(-1)
        g = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in dict(
            rows=rows, cols=cols, tile_of=tile_of.reshape(-1),
            b11=base(ty1, tx1), b12=base(ty1, tx2), b21=base(ty2, tx1), b22=base(ty2, tx2),
            xa=np.tile(xa, h), xa1=np.tile(xa1, h), ya=np.repeat(ya, w), ya1=np.repeat(ya1, w),
        ).items()}
        g["padded"] = bool(ph or pw)
        g["tile_area"] = th * tw
        return g
    return _const(("clahe_geom", h, w, grid, str(device)), build)


def clahe_u8_gpu(L, clip=2.0, grid=8):
    """Batched CLAHE of a (N,H,W) uint8 plane with cv2.createCLAHE semantics -> (N,H,W) uint8.

    Every tile of every image is processed by the same kernels: one scatter_add_ builds all
    N*grid*grid histograms, clipping/redistribution and the cumulative LUTs are row-wise
    tensor ops, and the bilinear blend is four gathers.
    """
    n, h, w = L.shape
    g = _clahe_geometry(h, w, grid, L.device)
    T, area = grid * grid, g["tile_area"]
    v = L.long()
    vp = v[:, g["rows"]][:, :, g["cols"]] if g["padded"] else v

    # 1. histograms: bin id = (image * T + tile) * 256 + value, all tiles in one scatter_add_.
    #    (torch.bincount would be shorter but reads max(ids) back to the host: one blocking
    #    device->host sync per call, 75% of the host time in torch_profiler_gpu_preprocess_v1.txt)
    img_off = (torch.arange(n, device=L.device) * (T * 256)).view(n, 1)
    ids = (img_off + g["tile_of"] * 256 + vp.reshape(n, -1)).reshape(-1)
    hist = torch.zeros(n * T * 256, dtype=torch.long, device=L.device)
    hist.scatter_add_(0, ids, _ones(ids.numel(), L.device))
    hist = hist.view(n * T, 256)

    # 2. clip at int(clip * area / 256) and redistribute the excess (OpenCV CLAHE_CalcLut_Body)
    limit = max(int(clip * area / 256), 1)
    excess = (hist - limit).clamp_(min=0).sum(1, keepdim=True)
    hist = hist.clamp_(max=limit)
    redist = excess // 256
    residual = excess - redist * 256
    hist += redist
    bins = torch.arange(256, device=L.device)
    step = (256 // residual.clamp(min=1)).clamp(min=1)
    hist += ((bins % step == 0) & (bins // step < residual)).long()

    # 3. per-tile LUT = round(cdf * 255 / area), float32 multiply + round-half-even like cvRound
    #    (the scale is cached on the device: building it on the host and calling .to(device)
    #    would be a synchronous pageable copy on every call, seen in Nsight Systems)
    scale = _const(("clahe_scale", area, str(L.device)), lambda: (
        torch.tensor(255.0, dtype=torch.float32) / torch.tensor(float(area), dtype=torch.float32)).to(L.device))
    lut = torch.round(hist.cumsum(1).float() * scale).clamp_(0, 255).view(n, T * 256)

    # 4. bilinear blend of the four neighbouring tile LUTs, same evaluation order as OpenCV
    vf = v.reshape(n, -1)
    l11, l12 = lut.gather(1, g["b11"] + vf), lut.gather(1, g["b12"] + vf)
    l21, l22 = lut.gather(1, g["b21"] + vf), lut.gather(1, g["b22"] + vf)
    res = (l11 * g["xa1"] + l12 * g["xa"]) * g["ya1"] + (l21 * g["xa1"] + l22 * g["xa"]) * g["ya"]
    return torch.round(res).clamp_(0, 255).to(torch.uint8).view(n, h, w)


def clahe_gpu(t, clip=2.0, grid=8, mode="lab"):
    """Batched tiled CLAHE. t: (N,3,H,W) float in [0,1] -> same shape and range.

    mode="lab"  (default): the CPU reference algorithm (8-bit RGB -> LAB, CLAHE on L,
                LAB -> RGB) with LUT colour conversions; bit-exact with apply_clahe.
    mode="luma": the lighter variant suggested in the starter code: CLAHE on the channel
                mean, RGB rescaled by (eq + 1) / (lum + 1). No 96 MiB tables, not exact.
    Leftover pixels when H or W is not a multiple of grid are handled like OpenCV
    (reflect-101 padding for the histograms only).
    """
    n, _, h, w = t.shape
    u8 = (t * 255).to(torch.uint8)                      # truncation, as (img*255).astype(uint8)
    if mode == "luma":
        lum = (t.mean(1) * 255).round_().to(torch.uint8)
        eq = clahe_u8_gpu(lum, clip, grid)
        ratio = (eq.float() + 1) / (lum.float() + 1)
        return (div255_(u8.float()) * ratio[:, None]).clamp_(0, 1)
    rgb2lab, lab2rgb = _lab_luts(t.device)
    w3 = _const(("lut_w", str(t.device)),
                lambda: torch.tensor([65536, 256, 1], device=t.device).view(1, 3, 1, 1))
    idx = (u8.long() * w3).sum(1)                                  # r*65536 + g*256 + b, (N,H,W)
    lab = rgb2lab[idx.view(-1)].view(n, h, w, 3)
    L2 = clahe_u8_gpu(lab[..., 0], clip, grid)
    idx = L2.long() * 65536 + (lab[..., 1].long() * 256 + lab[..., 2])
    rgb = lab2rgb[idx.view(-1)].view(n, h, w, 3)
    return div255_(rgb.permute(0, 3, 1, 2).float().contiguous())


# ---------------------------------------------------------------------------------------------
# Augmentation, conversion, full pipeline
# ---------------------------------------------------------------------------------------------

def augment_gpu(t, flip=None, angle=None):
    """Batched random horizontal flip + rotation in [-10, 10] degrees, reflect border.

    One flip decision and one angle PER IMAGE, drawn on the GPU (or given, for tests).
    The flip is folded into the affine matrix, so flip + rotation cost one grid_sample.
    The matrix reproduces cv2.getRotationMatrix2D((W/2, H/2), angle, 1) + warpAffine
    (pixel-centre conventions included); padding_mode="reflection" == BORDER_REFLECT.
    """
    n, _, h, w = t.shape
    dev = t.device
    if flip is None:
        flip = torch.rand(n, device=dev) < 0.5
    if angle is None:
        angle = torch.empty(n, device=dev).uniform_(-10, 10)
    a = torch.deg2rad(torch.as_tensor(angle, device=dev, dtype=torch.float32))
    cos, sin = a.cos(), a.sin()
    # output -> input map in normalised coords (align_corners=False):
    #   src_n = S^-1 A^-1 S dst_n + S^-1 (A^-1 d - d),  d = pixel offset of the cv2 centre
    #   from the grid centre = (-0.5, -0.5), S = diag(W/2, H/2), A^-1 = [[c, -s], [s, c]]
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin * h / w
    theta[:, 1, 0], theta[:, 1, 1] = sin * w / h, cos
    dx, dy = -0.5, -0.5
    theta[:, 0, 2] = ((cos * dx - sin * dy) - dx) * 2 / w
    theta[:, 1, 2] = ((sin * dx + cos * dy) - dy) * 2 / h
    sign = torch.where(torch.as_tensor(flip, device=dev), -1.0, 1.0)
    theta[:, 0] *= sign[:, None]                     # flip first == negate the source x
    grid = F.affine_grid(theta, (n, t.shape[1], h, w), align_corners=False)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def to_float_bchw(x, device="cuda"):
    """uint8/float NCHW tensor, or uint8/float NHWC numpy batch -> float [0,1] NCHW on device.

    The data moves to the device FIRST (non_blocking: asynchronous when the source is
    pinned) and is converted/scaled there, so a uint8 batch crosses PCIe at 1 byte per value.
    """
    nhwc = isinstance(x, np.ndarray)
    if nhwc:
        x = torch.from_numpy(np.ascontiguousarray(x))
    x = x.to(device, non_blocking=True)
    if nhwc:
        x = x.permute(0, 3, 1, 2)
    if x.dtype == torch.float32 and x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    out.copy_(x)                                      # layout + dtype conversion in one kernel
    if x.dtype == torch.uint8:
        div255_(out)                                  # same float32 op as u8.astype(np.float32) / 255.0
    return out


def preprocess_batch_gpu(images, cfg, augment=False, device="cuda", fuse=True):
    """The whole GPU preprocessing pipeline for one batch, same order as preprocess_image:
        to_float_bchw -> [augment_gpu] -> [gamma] -> [clahe_gpu] -> [circular_crop] -> [normalize]
    Exactly one host-to-device copy; everything after it stays on the GPU. Crop + normalize
    are one fused kernel (fuse=True) or two in-place kernels (fuse=False, bit-exact).
    """
    p = cfg["preprocess"]
    x = to_float_bchw(images, device)
    if isinstance(images, torch.Tensor) and x.data_ptr() == images.data_ptr():
        x = x.clone()                                  # never modify the caller's tensor in place
    if augment:
        x = augment_gpu(x)
    if p.get("gamma", 1.0) != 1.0:
        x = gamma_gpu(x, p["gamma"])
    if p.get("clahe"):
        x = clahe_gpu(x)
    crop, norm = p.get("circular_crop"), p.get("normalize")
    if crop and norm and fuse:
        return crop_normalize_fused(x)
    if crop:
        x = circular_crop_gpu(x, inplace=True)
    if norm:
        x = normalize_gpu(x, inplace=True)
    return x

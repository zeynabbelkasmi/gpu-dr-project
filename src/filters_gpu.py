"""GPU (torch) equivalents of the reference filters in src/filters_cpu.py (assignment A4).

All functions take a (N,3,H,W) float tensor on the GPU and process the whole
batch at once. Tolerances are defined in tests/test_cpu_gpu_equivalence.py;
measured differences are in results/week2/equivalence.csv.

A4.11: the median filter has three GPU implementations, compared in
benchmarks/benchmark_gpu_processing.py:
  median_filter_gpu          torch, unfold + median (the reference GPU version)
  median_filter_cupy_kernel  CuPy RawKernel, one thread per pixel, 25 values in registers
  median_filter_cupy_ndimage cupyx.scipy.ndimage.median_filter (library baseline)
CuPy shares the tensor with torch through DLPack (no copy) and runs on torch's stream.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocess_gpu import div255_

_CONST = {}


def _odd(k):
    return k if k % 2 == 1 else k + 1


def _gauss_kernel_1d(k, sigma, device, dtype):
    """The exact 1D kernel cv2.GaussianBlur uses (cv2.getGaussianKernel), cached on device."""
    key = (k, float(sigma), str(device), dtype)
    if key not in _CONST:
        g = cv2.getGaussianKernel(k, sigma, cv2.CV_32F).ravel()
        _CONST[key] = torch.from_numpy(g).to(device=device, dtype=dtype)
    return _CONST[key]


def gaussian_blur_gpu(t, ksize=5, sigma=1.0, separable=True):
    """Gaussian blur of a batch: depthwise conv with reflect padding.

    OpenCV's default border BORDER_REFLECT_101 equals torch "reflect". separable=True runs
    two 1D passes (2k MACs per pixel instead of k^2); separable=False is the direct 2D
    kernel (outer product), kept for the comparison in the Week 2 report.
    """
    k = _odd(ksize)
    c, p = t.shape[1], k // 2
    g = _gauss_kernel_1d(k, sigma, t.device, t.dtype)
    x = F.pad(t, (p, p, p, p), mode="reflect")
    if separable:
        x = F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
        return F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    return F.conv2d(x, torch.outer(g, g).expand(c, 1, k, k), groups=c)


def _quantize_u8(t):
    """(img * 255).astype(np.uint8) of the CPU reference: same float32 product, truncation."""
    return (t * 255).to(torch.uint8)


def median_filter_gpu(t, ksize=5):
    """Median filter of a batch: replicate padding (cv2.medianBlur's border), all k*k
    neighbourhoods as a strided view, median along that axis.

    Works on the uint8-quantised image like the CPU reference, so the result is bit-exact.
    Memory: the view is free, but median() materialises N*3*H*W*k^2 bytes (25x the uint8
    batch, 6.25x the float32 batch for k=5): ~470 MiB for N=128 at 224x224.
    """
    k = _odd(ksize)
    p = k // 2
    q = _quantize_u8(t)
    # F.pad "replicate" needs a float tensor; index-based replicate padding keeps uint8
    n, c, h, w = q.shape
    rows = torch.arange(-p, h + p, device=t.device).clamp_(0, h - 1)
    cols = torch.arange(-p, w + p, device=t.device).clamp_(0, w - 1)
    qp = q[:, :, rows][:, :, :, cols]
    win = qp.unfold(2, k, 1).unfold(3, k, 1)                 # (N,C,H,W,k,k) view, no copy
    med = win.reshape(n, c, h, w, k * k).median(dim=-1).values
    return div255_(med.float())


# --- A4.11: second (and third) implementation of the median filter with CuPy ---------------

_MEDIAN_SRC = r"""
extern "C" __global__
void median_u8(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
               const int planes, const int h, const int w)
{
    // one thread per output pixel of one (image, channel) plane; replicate border
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int pl = blockIdx.z;
    if (x >= w || y >= h || pl >= planes) return;
    const unsigned char* s = src + (size_t)pl * h * w;
    unsigned char v[KK];
    int n = 0;
    #pragma unroll
    for (int dy = -R; dy <= R; ++dy) {
        const int yy = min(max(y + dy, 0), h - 1);
        #pragma unroll
        for (int dx = -R; dx <= R; ++dx) {
            const int xx = min(max(x + dx, 0), w - 1);
            v[n++] = s[yy * w + xx];
        }
    }
    // partial selection sort: after KK/2+1 passes v[KK/2] is the median
    #pragma unroll
    for (int i = 0; i <= KK / 2; ++i) {
        #pragma unroll
        for (int j = i + 1; j < KK; ++j) {
            const unsigned char a = v[i], b = v[j];
            v[i] = min(a, b); v[j] = max(a, b);
        }
    }
    dst[(size_t)pl * h * w + y * w + x] = v[KK / 2];
}
"""


def _cupy():
    import cupy as cp                      # optional dependency (see requirements.txt)
    return cp


def cupy_available():
    """True when CuPy is installed AND can run a kernel on this GPU/driver.

    `import cupy` succeeds even when the wheel's CUDA major version is newer than the
    driver supports (e.g. cupy-cuda13x on a CUDA 12 driver); only running code shows it.
    """
    if "cupy_ok" not in _CONST:
        try:
            cp = _cupy()
            _CONST["cupy_ok"] = int(cp.arange(3).sum()) == 3
        except Exception:
            _CONST["cupy_ok"] = False
    return _CONST["cupy_ok"]


def _median_kernel(k):
    key = ("median_kernel", k)
    if key not in _CONST:
        cp = _cupy()
        _CONST[key] = cp.RawKernel(_MEDIAN_SRC, "median_u8",
                                   options=(f"-DKK={k * k}", f"-DR={k // 2}"))
    return _CONST[key]


def _on_torch_stream(fn):
    """Run CuPy work on torch's current CUDA stream, so ordering with torch ops is kept."""
    cp = _cupy()
    with cp.cuda.Stream.from_external(torch.cuda.current_stream()):
        return fn(cp)


def median_filter_cupy_kernel(t, ksize=5, block=(32, 8)):
    """A4.11 custom kernel: thread per pixel, the k*k window in registers, partial selection.

    No k^2-sized intermediate (memory = input + output), uint8 traffic only.
    """
    k = _odd(ksize)
    q = _quantize_u8(t).contiguous()
    n, c, h, w = q.shape
    out = torch.empty_like(q)
    kern = _median_kernel(k)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], n * c)

    def run(cp):
        kern(grid, block, (cp.from_dlpack(q), cp.from_dlpack(out),
                           np.int32(n * c), np.int32(h), np.int32(w)))
    _on_torch_stream(run)
    return div255_(out.float())


def median_filter_cupy_ndimage(t, ksize=5):
    """A4.11 library version: cupyx.scipy.ndimage.median_filter, mode="nearest" == replicate."""
    k = _odd(ksize)
    q = _quantize_u8(t).contiguous()

    def run(cp):
        from cupyx.scipy import ndimage
        return torch.from_dlpack(ndimage.median_filter(cp.from_dlpack(q), size=(1, 1, k, k),
                                                       mode="nearest"))
    return div255_(_on_torch_stream(run).float())


def green_channel_gpu(t):
    """The green channel as a (N,1,H,W) tensor (a view, no copy)."""
    return t[:, 1:2]


def contrast_stretch_gpu(t):
    """Per-image contrast stretch between the 1st and 99th percentiles of all pixels of
    that image (np.percentile 'linear' definition, as filters_cpu.contrast_stretch).

    torch.quantile refuses inputs above 2^24 elements (a 128-image batch has 19M) and sorts
    the whole row. Only the 2 order statistics around each percentile are needed, so
    topk(k ~ 1.5k) from the bottom and from the top is enough: exact, and k << n.
    """
    n = t.shape[0]
    flat = t.reshape(n, -1)
    m = flat.shape[1]

    def bounds(q):
        pos = q / 100.0 * (m - 1)
        i = int(np.floor(pos))
        return i, pos - i, min(i + 1, m - 1)

    i1, f1, j1 = bounds(1)
    i99, f99, j99 = bounds(99)
    low = flat.topk(j1 + 1, dim=1, largest=False, sorted=True).values            # ascending
    high = flat.topk(m - i99, dim=1, largest=True, sorted=True).values           # descending
    lo = low[:, i1] + (low[:, j1] - low[:, i1]) * f1
    hi = high[:, m - 1 - i99] + (high[:, m - 1 - j99] - high[:, m - 1 - i99]) * f99
    lo, hi = lo.view(n, 1, 1, 1), hi.view(n, 1, 1, 1)
    return ((t - lo) / (hi - lo).clamp(min=1e-6)).clamp_(0, 1)

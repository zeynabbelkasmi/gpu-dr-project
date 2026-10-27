"""Overlap host-to-device copies with GPU compute using a side CUDA stream (A5)."""
import torch


class CudaPrefetcher:
    """TODO(A5.2): wrap a DataLoader that yields pinned CPU tensors.

    While the model computes on batch i, batch i+1 is copied to the GPU on a
    separate torch.cuda.Stream. The wrapper must:
      - __init__(loader, device): store both and create the side stream
      - __len__: the length of the wrapped loader
      - __iter__: yield tuples of device tensors, one per batch of the loader
    Details to get right:
      - issue the copies inside `with torch.cuda.stream(self.stream)` using
        non_blocking=True (this only overlaps when the source is pinned memory)
      - before handing a batch to the caller, make the current stream wait for
        the side stream (current_stream.wait_stream)
      - tell the allocator the tensors are used on the current stream
        (tensor.record_stream), otherwise memory can be reused too early
    Measure with benchmarks/profile_gpu.py whether the copy really overlaps.
    """

    def __init__(self, loader, device):
        raise NotImplementedError("TODO(A5.2): CudaPrefetcher")

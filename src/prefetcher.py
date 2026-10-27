"""Overlap host-to-device copies with GPU compute using a side CUDA stream (A5)."""
import torch


class CudaPrefetcher:
    """A5.2: wrap a DataLoader that yields pinned CPU tensors.

    While the model computes on batch i (default stream), batch i+1 is copied to the
    GPU on a separate torch.cuda.Stream. The copy is asynchronous only when the source
    tensors are in pinned (page-locked) memory, hence pin_memory=True in make_loaders:
    a pageable tensor is first staged by the driver into a pinned buffer synchronously,
    so non_blocking=True would not overlap anything.

    Correctness rules:
      - the copies are issued inside `with torch.cuda.stream(self.stream)`;
      - before a batch is handed to the caller, the current stream waits for the side
        stream (wait_stream), so no kernel reads a tensor whose copy is still running;
      - tensor.record_stream(current) tells the caching allocator that the tensor is
        used on the current stream; without it the memory, allocated on the side stream,
        could be handed to the next copy while the model still reads it.
    Whether the copy really overlaps compute is checked with benchmarks/profile_gpu.py (A7).
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

    def __len__(self):
        return len(self.loader)

    def _to_device(self, batch):
        """Asynchronous copy of every tensor of the batch (tuple/list) on the side stream."""
        with torch.cuda.stream(self.stream):
            return tuple(t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t
                         for t in batch)

    def __iter__(self):
        current = torch.cuda.current_stream(self.device)
        it = iter(self.loader)
        try:
            nxt = self._to_device(next(it))
        except StopIteration:
            return
        while nxt is not None:
            current.wait_stream(self.stream)           # copy of `nxt` finished before use
            batch = nxt
            for t in batch:
                if torch.is_tensor(t):
                    t.record_stream(current)
            try:                                        # start copying i+1 before returning i
                nxt = self._to_device(next(it))
            except StopIteration:
                nxt = None
            yield batch

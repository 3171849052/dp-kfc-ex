"""CUDA wall boundaries and deferred, asynchronous event breakdowns."""
from contextlib import contextmanager
import time
import torch


def timestamp(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


class EventProfiler:
    def __init__(self):
        self.events = []

    def resolve(self):
        """Caller must have synchronized the complete phase before reading events."""
        totals = {}
        for name, start, end in self.events:
            totals[name] = totals.get(name, 0.) + start.elapsed_time(end)/1000
        self.events.clear()
        return totals


@contextmanager
def timed(profiler, name, device):
    if profiler is None:  # Profiling is optional for numerical tests, not a CPU fallback.
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    yield
    end.record(torch.cuda.current_stream(device))
    profiler.events.append((name, start, end))


def phase_start(device):
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)


def phase_end(device, phase, start):
    allocated = torch.cuda.max_memory_allocated(device)
    return {f'{phase}_allocated_at_phase_start_bytes': start[0],
            f'{phase}_reserved_at_phase_start_bytes': start[1],
            f'{phase}_peak_cuda_allocated_bytes': allocated,
            f'{phase}_peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
            f'{phase}_incremental_peak_bytes': allocated-start[0]}

"""Synchronized timing and independent CUDA phase peaks; stores numbers only."""
from contextlib import contextmanager
import time
import torch


def timestamp(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


@contextmanager
def timed(stats, name, device):
    start = timestamp(device)
    yield
    stats[name] = stats.get(name, 0.) + timestamp(device) - start


def phase_start(device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)


def phase_end(device, phase, start):
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    return {f'{phase}_allocated_at_phase_start_bytes': start[0],
            f'{phase}_reserved_at_phase_start_bytes': start[1],
            f'{phase}_peak_cuda_allocated_bytes': allocated,
            f'{phase}_peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
            f'{phase}_incremental_peak_bytes': allocated-start[0]}

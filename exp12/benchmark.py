import time
import torch


def measure(fn, device):
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    factors, stats = fn()
    if cuda:
        torch.cuda.synchronize(device)
    stats.update(build_seconds=time.perf_counter()-start,
                 peak_cuda_allocated=torch.cuda.max_memory_allocated(device) if cuda else 0,
                 peak_cuda_reserved=torch.cuda.max_memory_reserved(device) if cuda else 0)
    return factors, stats

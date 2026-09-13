"""Warm each estimator separately; measure synchronized builds with clean caches."""
import gc
import statistics
import time
import torch


def measure(fn, device, warmup=2, repeats=5):
    assert warmup >= 1 and repeats >= 1
    cuda = torch.device(device).type == 'cuda'
    for _ in range(warmup):
        output, _ = fn()
        if cuda:
            torch.cuda.synchronize(device)
        del output
    rows = []
    for repeat in range(repeats):
        gc.collect()
        baseline_allocated = baseline_reserved = 0
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output, stats = fn()
        if cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start
        allocated = torch.cuda.max_memory_allocated(device) if cuda else 0
        reserved = torch.cuda.max_memory_reserved(device) if cuda else 0
        rows.append(dict(repeat=repeat, build_seconds=elapsed,
                         baseline_allocated=baseline_allocated,
                         peak_allocated_absolute=allocated,
                         peak_allocated_increment=allocated-baseline_allocated,
                         baseline_reserved=baseline_reserved,
                         peak_reserved_absolute=reserved,
                         peak_reserved_increment=reserved-baseline_reserved))
        # Diagnostic/reference outputs live on CPU, outside the timed region.
        if repeat == repeats-1:
            factors = {n: {key: v.detach().cpu().clone() for key, v in f.items()} for n, f in output.items()}
        del output
    times = [r['build_seconds'] for r in rows]
    stats.update(median_build_seconds=statistics.median(times),
                 build_median_seconds=statistics.median(times),
                 min_build_seconds=min(times), max_build_seconds=max(times),
                 warmup=warmup, repeats=repeats)
    for field in rows[0]:
        if field not in ('repeat', 'build_seconds'):
            stats[field] = max(r[field] for r in rows)
    return factors, stats, rows

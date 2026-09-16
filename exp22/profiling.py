"""Minimal phase and CUDA memory measurements for Exp22."""

from __future__ import annotations

import time

import torch


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    if torch.device(device).type != "cuda":
        return {"cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}
    return {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def timed_call(fn, device):
    synchronize(device)
    start = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, time.perf_counter() - start


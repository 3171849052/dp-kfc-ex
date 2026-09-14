"""Synthetic KFAC-U only; no private data enters the builder."""
import torch
from exp12.curvature import estimate
from exp13.builders import synthetic_cache
from exp14 import config as cfg
from exp14.operator import Operator


def build_from_cache(model, beta, cache, seed, epoch):
    with torch.random.fork_rng():
        factors, budget = estimate(model, cache, 'KFAC-U', seed=seed + 20000 + epoch)
        operator = Operator(factors, beta)
    return operator, {f'builder_{k}': v for k, v in budget.items()}, factors


def build_preconditioner(model, beta, seed, epoch, device, *,
                         batches=cfg.SYNTHETIC_BATCHES,
                         batch_size=cfg.SYNTHETIC_BATCH_SIZE):
    cache = synthetic_cache(seed, epoch, device, batches, batch_size)
    return build_from_cache(model, beta, cache, seed, epoch)

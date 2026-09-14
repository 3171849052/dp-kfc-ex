"""Synthetic KFLR / KFRA-block only; no private data enters the builder."""
import torch
from dp_kfac.models import SimpleCNN
from exp12.curvature import estimate
from exp13.builders import synthetic_cache
from exp14c import config as cfg
from exp14c.operator import Operator


def build_from_cache(model, estimator, beta, cache, seed, epoch):
    assert type(model) is SimpleCNN
    assert estimator in cfg.ESTIMATORS
    with torch.random.fork_rng():
        factors, budget = estimate(model, cache, estimator, seed=seed + 20000 + epoch)
        operator = Operator(factors, beta)
    stats = {f'builder_{k}': v for k, v in budget.items()}
    stats.update(operator.diagnostics)
    return operator, stats, factors


def build_preconditioner(model, estimator, beta, seed, epoch, device, *,
                         batches=cfg.SYNTHETIC_BATCHES,
                         batch_size=cfg.SYNTHETIC_BATCH_SIZE):
    cache = synthetic_cache(seed, epoch, device, batches, batch_size)
    return build_from_cache(model, estimator, beta, cache, seed, epoch)

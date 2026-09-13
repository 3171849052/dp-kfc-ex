"""Builders only accept a model and synthetic probe settings, never private data."""
import torch
from dp_kfac.models import SimpleCNN
from dp_kfac.optimizer import generate_pink_noise
from exp12.curvature import estimate
from exp12.state_metrics import synthetic_state_metrics
from exp13 import config as cfg
from exp13.operator import Operator

ESTIMATORS = dict(zip(cfg.METHODS, ('KFAC-U', 'KFLR', 'KFRA-block')))


def synthetic_cache(seed, epoch, device, batches, batch_size):
    # manual_seed seeds all CUDA generators, so preserve all of them as well.
    with torch.random.fork_rng():
        torch.manual_seed(seed + 10000 + epoch)
        return [generate_pink_noise(batch_size, (1, 28, 28), device) for _ in range(batches)]


def build_preconditioner(model, method, seed, epoch, device, *,
                         batches=cfg.SYNTHETIC_BATCHES,
                         batch_size=cfg.SYNTHETIC_BATCH_SIZE):
    assert type(model) is SimpleCNN
    assert method in cfg.METHODS
    with torch.random.fork_rng():
        cache = synthetic_cache(seed, epoch, device, batches, batch_size)
        factors, budget = estimate(model, cache, ESTIMATORS[method], seed=seed + 20000 + epoch)
        state = synthetic_state_metrics(model, cache)
        operator = Operator(factors)
    return operator, {**state, **{f'builder_{k}': v for k, v in budget.items()},
                      'state_metric_forward_calls': len(cache)}

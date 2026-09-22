"""Exp36b calibration and A accumulation, with bounded spectral mixing."""
from types import SimpleNamespace
import torch
from exp36c.config import cfg, ALPHAS, ORACLE_INDICES
from exp35.adapters import bind
from exp35.geometry import SpectralAOperator, spectral_operator, TOL
from exp36b.oracle_geometry import calibration as reference_calibration, build as reference_build

# Read the original CSV directly on every epoch; never regenerate its indices.
calibration = bind(reference_calibration, cfg=SimpleNamespace(RESULTS=ORACLE_INDICES.parent))


def bounded_spectral(covariance, alpha):
    assert alpha in (.25, .5)
    matrix, gains = spectral_operator(covariance, 'norm')
    matrix = (1-alpha)*torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype) + alpha*matrix
    gains = (1-alpha) + alpha*gains
    assert gains.min().item() >= 1-alpha-TOL
    assert gains.max().item() <= 1+TOL
    return matrix, gains


class BoundedAOperator(SpectralAOperator):
    __init__ = bind(SpectralAOperator.__init__, spectral_operator=bounded_spectral)


def builder(method):
    alpha = ALPHAS[method]
    inherited = bind(reference_build,
        SpectralAOperator=lambda factors, variant: BoundedAOperator(factors, alpha))

    def build(model, exp22_method, batches, seed=42, epoch=1, damping=.001, power=.4):
        assert exp22_method == 'dp_kfc_a_bk'
        assert (seed, damping, power) == (42, .001, .4)
        return inherited(model, exp22_method, batches, seed, epoch, damping, power)
    return build

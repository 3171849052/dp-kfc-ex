"""Use a raw activation spectrum with the Exp22 A-only builder."""
from exp36d.config import cfg
import torch
from exp22 import geometry as reference
from exp35.adapters import bind
from exp35.geometry import SpectralAOperator


def raw_spectrum(covariance, power):
    values, vectors = torch.linalg.eigh(((covariance + covariance.T) * .5).double())
    gains = (values.clamp_min(0) + cfg.DAMPING).pow(-power)
    return ((vectors * gains) @ vectors.T).to(covariance.dtype), gains


class RawAOperator(SpectralAOperator):
    def __init__(self, factors, power, damping):
        assert damping == cfg.DAMPING
        bind(SpectralAOperator.__init__, spectral_operator=raw_spectrum)(self, factors, power)
        self.power = power


def builder(power):
    a_build = torch.no_grad()(bind(reference.build_a_operator.__wrapped__, AOnlyOperator=RawAOperator))
    return bind(reference.build_from_batches, build_a_operator=a_build)

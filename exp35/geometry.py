"""Per-layer spectral gains, shared by CNN and ViT geometry builders."""
import torch
from exp35.config import A_POWER, DAMPING
from exp22.geometry import AOnlyOperator

TOL = 2e-6

def spectral_operator(covariance, variant):
    values, vectors = torch.linalg.eigh(((covariance + covariance.T) * .5).double())
    values = values.clamp_min(0)
    gains = (values + DAMPING).pow(-A_POWER)
    if variant in ('norm', 'mix05'):
        gains = gains / gains.max()
        assert abs(gains.max().item() - 1) <= TOL
    if variant == 'mix05':
        gains = .5 + .5 * gains
        assert gains.min() >= .5 - TOL
    if variant in ('norm', 'mix05'):
        assert gains.max() <= 1 + TOL
    return ((vectors * gains) @ vectors.T).to(covariance.dtype), gains


def diagnostics(gains):
    values = torch.cat(list(gains.values())) if gains else torch.ones(1, dtype=torch.float64)
    result = {f'operator_gain_p{p}': values.quantile(p / 100).item() for p in (10, 50, 90, 99)}
    result.update(operator_gain_min=values.min().item(), operator_gain_max=values.max().item())
    for name, gain in gains.items():
        result[f'layer_gain_min_{name}'] = gain.min().item()
        result[f'layer_gain_max_{name}'] = gain.max().item()
    return result


class SpectralAOperator(AOnlyOperator):
    """Reuse the Exp22 activation/gradient transforms with unit outer scale."""
    def __init__(self, factors, variant):
        self.power, self.damping, self.scale = A_POWER, DAMPING, 1.0
        self.data, gains = {}, {}
        for name, factor in factors.items():
            self.data[name], gains[name] = spectral_operator(factor['A'], variant)
        self.preconditioned_layers = sorted(factors)
        self.moments = {'scale_match': 1.0}
        self.diagnostics = diagnostics(gains)
        self.operator_state_bytes = sum(t.numel() * t.element_size() for t in self.data.values()) + 8

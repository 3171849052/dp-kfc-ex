"""One synthetic-KFAC RMS scale shared by all layer powers."""
import math
import torch
from exp14.operator import Operator as RawOperator
from exp14b.config import DAMPING


def scale_diagnostics(factors, beta):
    moments = {beta: 0., .5: 0.}
    for factors_layer in factors.values():
        spectra = []
        for key in ('C', 'A'):
            factor = factors_layer[key].double()
            spectra.append(torch.linalg.eigvalsh((factor + factor.T) / 2).clamp_min(0))
        for power in moments:
            c, a = [(e * (e + DAMPING).pow(-2 * power)).sum() for e in spectra]
            moments[power] += (c * a).item()
    raw, reference = moments[beta], moments[.5]
    scale = math.sqrt(reference / raw)
    return dict(predicted_second_moment_raw=raw,
                predicted_second_moment_ref=reference, scale_match=scale,
                predicted_rms_raw=math.sqrt(raw),
                predicted_rms_matched=scale * math.sqrt(raw))


class Operator(RawOperator):
    def __init__(self, factors, beta):
        self.diagnostics = scale_diagnostics(factors, beta)
        super().__init__(factors, beta)
        side_scale = math.sqrt(self.diagnostics['scale_match'])
        self.data = {name: (left * side_scale, right * side_scale)
                     for name, (left, right) in self.data.items()}

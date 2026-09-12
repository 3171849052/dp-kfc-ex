"""Adapters for existing operators; no new preconditioner definition."""
import torch
from exp10 import run_exp10 as base
from exp10b import config as cfg
from exp10b.operators import Operator, augmented, equil_scales
from exp6.run_exp6 import kfac_factors
from dp_kfac.precondition import precondition_per_sample_gradients


class StructuredOperator:
    def __init__(self, kind, data=None, reference=None):
        self.kind, self.data, self.reference = kind, data or {}, reference

    def transform_backprop(self, name, b):
        if self.kind == 'DP-SGD':
            return b
        left, _ = self.data[name]
        return b * left[None, :, None] if left.ndim == 1 else left @ b

    def transform_activation(self, name, a):
        if self.kind == 'DP-SGD':
            return a
        _, right = self.data[name]
        return a * right[None, :, None] if right.ndim == 1 else right.T @ a

    def transform_matrix(self, name, matrix):
        if self.kind == 'DP-SGD':
            return matrix
        left, right = self.data[name]
        if left.ndim == 1:
            return matrix * left[:, None] * right[None, :]
        return left @ matrix @ right

    @torch.no_grad()
    def transform_aggregate_gradient(self, model):
        for name, m in model.named_modules():
            if not isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
            matrix = m.weight.grad.flatten(1)
            if m.bias is not None:
                matrix = torch.cat((matrix, m.bias.grad[:, None]), 1)
            matrix = self.transform_matrix(name, matrix)
            m.weight.grad.copy_((matrix[:, :-1] if m.bias is not None else matrix).reshape_as(m.weight))
            if m.bias is not None:
                m.bias.grad.copy_(matrix[:, -1])

    def apply_exact(self, model):
        if self.kind == 'Factorized Equil':
            self.reference.apply(model)
        elif self.kind == 'DP-KFC':
            precondition_per_sample_gradients(model,
                {k: r for k, (l, r) in self.data.items()},
                {k: l for k, (l, r) in self.data.items()})


def build(model, kind, seed, epoch, device):
    if kind == 'DP-SGD':
        return StructuredOperator(kind)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed + 10000 + epoch)
        if kind == 'Factorized Equil':
            wrapped = base.GradSampleModule(model, loss_reduction='sum')
            batches = base.pink_batches(cfg.PRECONDITIONER_BATCHES, device)
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(seed + 30000 + epoch)
                probes = base.rademacher(wrapped.parameters(), cfg.PROBES)
            statistic = base.layerwise_statistics(wrapped, batches, device, probes)
            reference = Operator(kind, augmented(wrapped, equil_scales(statistic)))
            result = StructuredOperator(kind, reference.data, reference)
            wrapped.zero_grad(set_to_none=True)
            wrapped.to_standard_module()
        else:
            a, g = kfac_factors(model, cfg.SYNTHETIC_BATCH_SIZE, device)
            result = StructuredOperator(kind, {k: (g[k], a[k]) for k in a})
    return result

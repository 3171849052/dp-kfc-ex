"""Synthetic two-stage Kronecker preconditioning, before clipping and noise."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dp_kfac.models import SimpleCNN
from dp_kfac.recorder import KFACRecorder
from dp_kfac.standalone.trainer import pink_batches
from exp15.preconditioner import SyntheticKLBFGS, rows


class FisherFactor:
    def __init__(self, covariance, damping, q):
        self.covariance = (covariance + covariance.T) / 2
        self.damped = self.covariance + damping * torch.eye(
            len(covariance), device=covariance.device, dtype=covariance.dtype)
        values, vectors = torch.linalg.eigh(self.damped)
        assert torch.isfinite(values).all() and (values > 0).all()
        self.condition = (values[-1] / values[0]).item()
        # Same damping and eigenvalue floor as dp_kfac.covariance.
        self.inverse_power = (vectors * values.clamp(min=1e-6).pow(-q)) @ vectors.T

    def power(self, v):
        return v @ self.inverse_power.to(v.dtype)


class Preconditioner:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.mode, self.q = config['mode'], config['q']
        self.hessian = SyntheticKLBFGS(config, device)
        self.fisher = {}
        self.norm_metrics = {}

    def refresh(self, model, epoch):
        if self.q == 0:
            return
        if self.mode in ('hessian', 'nested'):
            self.hessian.refresh(model, epoch)  # Stage A; identical to exp15.
        if self.mode in ('fisher', 'nested'):
            self.refresh_fisher(model, epoch)  # Stage B; never appends H pairs.

    def refresh_fisher(self, model, epoch):
        totals = {}
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == 'cuda' else []):
            torch.manual_seed(self.config['seed'] + 20000 + epoch)
            probe = SimpleCNN().to(self.device)
            probe.load_state_dict(model._module.state_dict())
            recorder = KFACRecorder(probe)
            recorder.enable()
            for x, y in pink_batches(self.config, self.device):
                probe.zero_grad(set_to_none=True)
                F.cross_entropy(probe(x), y, reduction='sum').backward()
                with torch.no_grad():
                    for name, module in probe.named_modules():
                        if not isinstance(module, (nn.Conv2d, nn.Linear)):
                            continue
                        a = recorder.activations[name]
                        if isinstance(module, nn.Conv2d):
                            a = F.unfold(a, module.kernel_size, padding=module.padding,
                                         stride=module.stride).transpose(1, 2).flatten(0, 1)
                        a = torch.cat((a, torch.ones_like(a[:, :1])), 1).double()
                        g = rows(recorder.backprops[name]).double()
                        if self.mode == 'nested':
                            ha, hg = self.hessian.factors[name]
                            a, g = ha.power(a, self.q), hg.power(g, self.q)
                        covs = (a.T @ a / len(a), g.T @ g / len(g))
                        if name not in totals:
                            totals[name] = covs
                        else:
                            for total, cov in zip(totals[name], covs):
                                total.add_(cov)
                recorder.clear()
            recorder.remove()
        batches = self.config['synthetic']['samples'] // self.config['synthetic']['batch_size']
        self.fisher = {name: tuple(FisherFactor(cov / batches, self.config['kfac']['damping'], self.q)
                                   for cov in covs) for name, covs in totals.items()}

    @staticmethod
    def mean_norm(model):
        return sum(p.grad_sample.flatten(1).square().sum(1)
                   for p in model.parameters()).sqrt().mean().item()

    @torch.no_grad()
    def apply(self, model, q):
        assert q == self.q
        if q != 0 and self.mode in ('hessian', 'nested'):
            self.hessian.apply(model, q)
        self.norm_metrics = {'hessian_stage_norm_mean': self.mean_norm(model)}
        if q != 0 and self.mode in ('fisher', 'nested'):
            for name, module in model._module.named_modules():
                if name not in self.fisher:
                    continue
                a, g = self.fisher[name]
                weight = module.weight.grad_sample
                matrix = torch.cat((weight.flatten(2), module.bias.grad_sample.unsqueeze(-1)), -1)
                matrix = g.power(a.power(matrix).transpose(-1, -2)).transpose(-1, -2)
                module.weight.grad_sample = matrix[..., :-1].reshape_as(weight)
                module.bias.grad_sample = matrix[..., -1]
        self.norm_metrics['final_preconditioned_norm_mean'] = self.mean_norm(model)

    def diagnostics(self):
        return {f'{name}/{side}_damped_condition': factor.condition
                for name, factors in self.fisher.items()
                for side, factor in zip(('A', 'G'), factors)}

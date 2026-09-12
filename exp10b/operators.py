"""Unbounded Equil operators. Dense materialization is diagnostic-only."""
import hashlib
import numpy as np
import torch
from scipy.stats import rankdata
from exp10.run_exp10 import layerwise_statistics, fisher_statistics, pink_batches, rademacher
from exp10b.config import LAYERS, PROBES, PRECONDITIONER_BATCHES, TAU, STRUCTURED


@torch.no_grad()
def equil_scales(statistic):
    values = torch.cat([v.flatten() for v in statistic.values()])
    gamma = TAU * values.median()
    raw = {p: (v.double() + gamma.double()).rsqrt() for p, v in statistic.items()}
    log_mean = sum(v.log().sum() for v in raw.values()) / values.numel()
    return {p: (v.log() - log_mean).exp().to(p.dtype) for p, v in raw.items()}


def augmented(model, scales):
    return {name: torch.cat((scales[m.weight].flatten(1), scales[m.bias][:, None]), 1)
            for name in LAYERS for m in [getattr(model._module, name)]}


class Operator:
    def __init__(self, kind, target):
        self.kind = kind
        self.shapes = {k: tuple(v.shape) for k, v in target.items()}
        logs = {k: v.double().log() for k, v in target.items()}
        if kind == "Factorized Equil":
            self.data = {k: (v.mean(1), v.mean(0)-v.mean()) for k, v in logs.items()}
            offset = sum(v.numel()*(self.data[k][0].mean()+self.data[k][1].mean())
                         for k, v in logs.items()) / sum(v.numel() for v in logs.values())
            self.data = {k: ((r-offset).exp().to(target[k].dtype), c.exp().to(target[k].dtype))
                         for k, (r, c) in self.data.items()}
        elif kind == "Layer-Scalar Equil":
            offset = sum(v.sum() for v in logs.values()) / sum(v.numel() for v in logs.values())
            self.data = {k: (v.mean()-offset).exp().to(target[k].dtype) for k, v in logs.items()}
        else:
            self.data = target

    def module_dof(self, name):
        a, b = self.shapes[name]
        if self.kind == "Factorized Equil":
            return a+b-1  # One row/column gauge per module.
        if self.kind == "Layer-Scalar Equil":
            return 1
        return a*b

    @property
    def dof(self):
        return sum(self.module_dof(k) for k in self.shapes)-1  # Global GM constraint.

    def dense(self):
        """Only for diagnostics/tests; never used by apply."""
        if self.kind == "Factorized Equil":
            return {k: r[:, None]*c[None, :] for k, (r, c) in self.data.items()}
        if self.kind == "Layer-Scalar Equil":
            return {k: s.expand(self.shapes[k]) for k, s in self.data.items()}
        return self.data

    @torch.no_grad()
    def apply(self, model):
        for name, value in self.data.items():
            m = getattr(model._module, name)
            g = m.weight.grad_sample
            if self.kind == "Factorized Equil":
                r, c = value
                g.mul_(r.view(1, -1, *([1]*(g.ndim-2))))
                g.mul_(c[:-1].view(1, 1, *m.weight.shape[1:]))
                m.bias.grad_sample.mul_(r[None, :]).mul_(c[-1])
            elif self.kind == "Layer-Scalar Equil":
                g.mul_(value)
                m.bias.grad_sample.mul_(value)
            else:
                g.mul_(value[:, :-1].reshape_as(m.weight))
                m.bias.grad_sample.mul_(value[:, -1])


def fingerprint(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def correlation(x, y):
    # Correlation with a constant vector is undefined, not a numerical failure.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def diagnostics(actual, target):
    x = target.double().log().flatten().cpu().numpy()
    y = actual.double().log().flatten().cpu().numpy()
    q = np.exp(np.quantile(y, [0, .01, .1, .5, .9, .99, 1]))
    return dict(log_scale_rmse=float(np.sqrt(np.mean((x-y)**2))),
                log_scale_pearson=correlation(x, y),
                log_scale_spearman=correlation(rankdata(x), rankdata(y)),
                correlation_defined=bool(np.ptp(x)>0 and np.ptp(y)>0),
                gain_geomean=float(np.exp(y.mean())),
                log10_gain_range=float((y.max()-y.min())/np.log(10)),
                **dict(zip(("gain_min", "gain_p1", "gain_p10", "gain_median",
                            "gain_p90", "gain_p99", "gain_max"), q)))


def build(model, method, seed, epoch, device):
    if method == "DP-SGD":
        ones = torch.ones(2)
        return None, dict(operator_dof=0, **diagnostics(ones, ones)), []
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed+10000+epoch)
        batches = pink_batches(PRECONDITIONER_BATCHES, device)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed+30000+epoch)
            probes = rademacher((p for p in model.parameters() if p.requires_grad), PROBES)
        if method == "Full-Fisher Equil":
            _, statistic = fisher_statistics(model, batches, device, probes)
        else:
            statistic = layerwise_statistics(model, batches, device, probes)
        target = augmented(model, equil_scales(statistic))
    # All projections consume this same statistic and target on this model.
    kinds = (method,) if method == "Full-Fisher Equil" else STRUCTURED
    stat_hash, target_hash = fingerprint(statistic.values()), fingerprint(target.values())
    records = []
    for kind in kinds:
        operator = Operator(kind, target)
        dense = operator.dense()
        common = dict(source_method=method, method=kind, seed=seed, epoch=epoch,
                      statistic_sha256=stat_hash, target_sha256=target_hash)
        for name in LAYERS:
            records.append(dict(**common, module=name, coordinates=target[name].numel(),
                                operator_dof=operator.module_dof(name),
                                **diagnostics(dense[name], target[name])))
        overall = diagnostics(torch.cat([v.flatten() for v in dense.values()]),
                              torch.cat([v.flatten() for v in target.values()]))
        records.append(dict(**common, module="__global__", coordinates=sum(v.numel() for v in target.values()),
                            operator_dof=operator.dof, **overall))
        if kind == method:
            selected, gains = operator, dict(operator_dof=operator.dof, **overall)
    return selected, gains, records

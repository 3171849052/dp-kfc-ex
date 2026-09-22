"""Synthetic-only Wiener-A noise-temperature filters and post-noise optimizer boundary."""
import torch
from torch import nn
from exp22.methods import Clipper, _group
from exp33d import config as cfg
from exp22.geometry import synthetic_stream


def matrix(module):
    weight = module.weight.grad
    return weight if module.bias is None else torch.cat((weight, module.bias.grad[:, None]), 1)


def spectrum(s):
    values, vectors = torch.linalg.eigh((s + s.T) / 2)
    # Covariances are PSD; negative eigenvalues can only be roundoff.
    assert values.min() >= -1e-6 * s.abs().max().clamp_min(1e-30)
    return values.clamp_min(0), vectors


def describe(prefix, values):
    values = values.flatten().float().cpu()
    result = {prefix + '_mean': values.mean().item(), prefix + '_median': values.quantile(.5).item()}
    for p in (10, 90, 99):
        result[f'{prefix}_p{p}'] = values.quantile(p / 100).item()
    result.update({prefix + '_min': values.min().item(), prefix + '_max': values.max().item()})
    return result


def spectral_diagnostics(gains, ratios):
    stats = {**describe('wiener_gain', gains), **describe('signal_to_noise', ratios)}
    stats.update(eig_zero_fraction=(ratios == 0).double().mean().item(),
                 eig_near_zero_fraction=(ratios < 1e-8).double().mean().item(),
                 effective_rank_fraction=(ratios >= 1e-8).double().mean().item())
    for label, threshold in (('001', .01), ('01', .1), ('05', .5), ('09', .9)):
        stats[f'gain_frac_gt_{label}'] = (gains > threshold).double().mean().item()
    return stats


def empty_wiener_diagnostics():
    keys = spectral_diagnostics(torch.ones(1), torch.ones(1))
    result = dict.fromkeys(keys)
    for group in cfg.UPDATE_GROUPS[:-1]:
        result.update(dict.fromkeys(f'group_{key}_{group}' for key in keys))
    result.update(dict.fromkeys(('operator_state_bytes', 'tau2', 'cos_filtered_clean', 'nsr_filtered',
                                'wiener_norm_ratio', 'wiener_norm_ratio_p10',
                                'wiener_norm_ratio_p50', 'wiener_norm_ratio_p90')))
    return result


class Wiener:
    def __init__(self, factors, tau2, beta):
        assert beta > 0 and tau2 > 0
        self.data = {}
        gains, ratios, groups = [], [], {}
        for name, factor in factors.items():
            va, ua = spectrum(factor['A'])
            gain = va / (va + beta * tau2)
            ratio = va / tau2
            assert torch.isfinite(gain).all() and gain.min() >= 0 and gain.max() <= 1
            self.data[name] = (ua.float(), gain.float())
            gains.append(gain.cpu())
            ratios.append(ratio.cpu())
            groups.setdefault(_group(name), []).append((gain.cpu(), ratio.cpu()))
        self.diagnostics = spectral_diagnostics(torch.cat(gains), torch.cat(ratios))
        self.diagnostics['tau2'] = tau2
        for group, values in groups.items():
            stats = spectral_diagnostics(torch.cat([v[0] for v in values]), torch.cat([v[1] for v in values]))
            self.diagnostics.update({f'group_{key}_{group}': value for key, value in stats.items()})
        self.operator_state_bytes = sum(t.numel() * t.element_size() for entry in self.data.values() for t in entry)

    def transform(self, name, z):
        ua, gain = self.data[name]
        return ((z @ ua) * gain) @ ua.T

    @torch.no_grad()
    def apply(self, model):
        groups = {}
        for name, module in model.named_modules():
            if name not in self.data:
                continue
            z = matrix(module)
            filtered = self.transform(name, z)
            before, after = z.square().sum(), filtered.square().sum()
            group = groups.setdefault(_group(name), [z.new_zeros(()), z.new_zeros(())])
            group[0].add_(before)
            group[1].add_(after)
            if module.bias is None:
                module.weight.grad.copy_(filtered)
            else:
                module.weight.grad.copy_(filtered[:, :-1])
                module.bias.grad.copy_(filtered[:, -1])
        return {f'group_wiener_norm_ratio_{g}': (a / b.clamp_min(1e-30)).sqrt().item() for g, (b, a) in groups.items()}


def build_wiener(model, seed, epoch, sigma, beta, batches=None, physical_batch_size=128):
    device = next(model.parameters()).device
    modules = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    factors = {}
    for name, module in modules.items():
        m, n = module.out_features, module.in_features + int(module.bias is not None)
        factors[name] = {'A': torch.zeros(n, n, device=device, dtype=torch.float64)}
    clipper = Clipper(model, operator=None, max_grad_norm=cfg.MAX_GRAD_NORM)
    label_rng = torch.Generator(device=device).manual_seed(seed + 20000 + epoch)
    count = samples = 0
    probes = synthetic_stream(seed, epoch, device) if batches is None else batches
    for x in probes:
        y = torch.randint(10, (len(x),), device=device, generator=label_rng)
        clipper.aggregate_logical(x, y, physical_batch_size)
        with torch.no_grad():
            for name, module in modules.items():
                z = matrix(module).double() / len(x)
                m, n = z.shape
                factors[name]['A'].add_(z.T @ z / m)
        count += 1
        samples += len(x)
    clipper.remove()
    model.zero_grad(set_to_none=True)
    for factor in factors.values():
        for value in factor.values():
            value.div_(count)
    tau2 = (sigma * cfg.MAX_GRAD_NORM / cfg.LOGICAL_BATCH_SIZE) ** 2
    operator = Wiener(factors, tau2, beta)
    return operator, {'builder_samples': samples, 'builder_logical_batches': count,
                      'builder_noise_events': 0, 'operator_state_bytes': operator.operator_state_bytes,
                      **operator.diagnostics}


def shadow(parameters, clean):
    dot = sum((p.grad * c).sum() for p, c in zip(parameters, clean))
    g2 = sum(p.grad.square().sum() for p in parameters)
    c2 = sum(c.square().sum() for c in clean)
    error = sum((p.grad - c).square().sum() for p, c in zip(parameters, clean))
    return (dot / (g2 * c2).sqrt().clamp_min(1e-30)).item(), (error / c2.clamp_min(1e-30)).sqrt().item(), g2


@torch.no_grad()
def private_step(clipper, optimizer, sigma, batch_size, generator, wiener):
    # Clean shadow is read only by diagnostics; never passed to the operator.
    clean = [p.grad.detach().clone().div_(batch_size) for p in clipper.parameters]
    # Exact Exp30 noise shape, order, and scaling, before any Wiener operation.
    for p in clipper.parameters:
        noise = torch.randn(p.numel(), device=p.device, dtype=p.dtype, generator=generator).reshape_as(p)
        p.grad.add_(noise, alpha=sigma * clipper.max_grad_norm).div_(batch_size)
    cos, nsr, raw2 = shadow(clipper.parameters, clean)
    stats = {'cos_raw_clean': cos, 'nsr_raw': nsr}
    if wiener is not None:
        stats.update(wiener.apply(clipper.model))
        cos, nsr, filtered2 = shadow(clipper.parameters, clean)
        stats.update(cos_filtered_clean=cos, nsr_filtered=nsr,
                     wiener_norm_ratio=(filtered2 / raw2.clamp_min(1e-30)).sqrt().item())
    before = [p.detach().clone() for p in clipper.parameters]
    optimizer.step()
    # Measure the actual AdamW change, including moments, epsilon and decay.
    parameter_groups = diagnostic_parameter_groups(clipper.model)
    sums = {group: [before[0].new_zeros(()), before[0].new_zeros(())] for group in cfg.UPDATE_GROUPS}
    for p, old in zip(clipper.parameters, before):
        update2, parameter2 = sums[parameter_groups[id(p)]]
        update2.add_((p - old).square().sum())
        parameter2.add_(old.square().sum())
    for group, (update2, parameter2) in sums.items():
        stats[f'update_norm_{group}'] = update2.sqrt().item()
        stats[f'relative_update_norm_{group}'] = (update2.sqrt() / (parameter2.sqrt() + 1e-30)).item()
    clipper.optimizer_steps += 1
    clipper.noise_events += 1
    return stats


def diagnostic_parameter_groups(model):
    groups = {id(p): 'identity' for p in model.parameters() if p.requires_grad}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    groups[id(p)] = _group(name)
    return groups


def make_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LEARNING_RATE, betas=cfg.ADAM_BETAS, eps=cfg.ADAM_EPS,
        weight_decay=cfg.WEIGHT_DECAY,
    )

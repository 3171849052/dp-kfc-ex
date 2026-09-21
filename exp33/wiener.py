"""Synthetic-only bounded spectral filters and post-noise optimizer boundary."""
import torch
from torch import nn
from exp22.methods import Clipper, _group
from exp33 import config as cfg
from exp33.geometry import synthetic_stream


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


class Wiener:
    def __init__(self, factors, tau2, full):
        self.data = {}
        gains, signals, groups = [], [], {}
        for name, factor in factors.items():
            va, ua = spectrum(factor['A'])
            ug = None
            signal = va
            if full:
                vg, ug = spectrum(factor['G'])
                alpha = ((va.mean() + vg.mean()) / 2).clamp_min(1e-12)
                signal = vg[:, None] * va[None, :] / alpha
            gain = signal / (signal + tau2)
            assert torch.isfinite(gain).all() and gain.min() >= 0 and gain.max() <= 1
            self.data[name] = (ua.float(), None if ug is None else ug.float(), gain.float())
            gains.append(gain.flatten().cpu())
            signals.append(signal.flatten().cpu())
            groups.setdefault(_group(name), []).append(gain.flatten().cpu())
        self.diagnostics = {**describe('wiener_gain', torch.cat(gains)), **describe('signal', torch.cat(signals)), 'tau2': tau2}
        self.diagnostics.update({f'group_wiener_gain_{g}': torch.cat(v).mean().item() for g, v in groups.items()})
        self.operator_state_bytes = sum(t.numel() * t.element_size() for entry in self.data.values() for t in entry if t is not None)

    def transform(self, name, z):
        ua, ug, gain = self.data[name]
        if ug is None:
            return ((z @ ua) * gain) @ ua.T
        return ug @ (((ug.T @ z @ ua) * gain) @ ua.T)

    @torch.no_grad()
    def apply(self, model):
        groups = {}
        for name, module in model.named_modules():
            if name not in self.data:
                continue
            z = matrix(module)
            filtered = self.transform(name, z)
            before, after = z.square().sum(), filtered.square().sum()
            assert after.sqrt() <= before.sqrt() * (1 + 2e-5) + 1e-8, name
            group = groups.setdefault(_group(name), [z.new_zeros(()), z.new_zeros(())])
            group[0].add_(before)
            group[1].add_(after)
            if module.bias is None:
                module.weight.grad.copy_(filtered)
            else:
                module.weight.grad.copy_(filtered[:, :-1])
                module.bias.grad.copy_(filtered[:, -1])
        return {f'group_wiener_norm_ratio_{g}': (a / b.clamp_min(1e-30)).sqrt().item() for g, (b, a) in groups.items()}


def build_wiener(model, seed, epoch, sigma, full, batches=None, physical_batch_size=128):
    device = next(model.parameters()).device
    modules = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    factors = {}
    for name, module in modules.items():
        m, n = module.out_features, module.in_features + int(module.bias is not None)
        factors[name] = {'A': torch.zeros(n, n, device=device, dtype=torch.float64)}
        if full:
            factors[name]['G'] = torch.zeros(m, m, device=device, dtype=torch.float64)
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
                if full:
                    factors[name]['G'].add_(z @ z.T / n)
        count += 1
        samples += len(x)
    clipper.remove()
    model.zero_grad(set_to_none=True)
    for factor in factors.values():
        for value in factor.values():
            value.div_(count)
    tau2 = (sigma * cfg.MAX_GRAD_NORM / cfg.LOGICAL_BATCH_SIZE) ** 2
    operator = Wiener(factors, tau2, full)
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
    optimizer.step()
    clipper.optimizer_steps += 1
    clipper.noise_events += 1
    return stats

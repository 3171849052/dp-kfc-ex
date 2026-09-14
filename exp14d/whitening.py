"""Streaming second moments from held-out synthetic tensors only."""
import math
import torch
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from dp_kfac.optimizer import generate_pink_noise
from exp12.curvature import layers
from exp14d import config as cfg


def synthetic_probe(seed, epoch, device, samples=cfg.PROBE_SAMPLES):
    # Separate input and label streams; neither contains beta or consumes builder RNG.
    with torch.random.fork_rng():
        torch.manual_seed(seed + 50000 + epoch)
        x = generate_pink_noise(samples, (1, 28, 28), device)
    rng = torch.Generator(device=device).manual_seed(seed + 60000 + epoch)
    y = torch.randint(10, (samples,), generator=rng, device=device)
    return x, y


def energy_metrics(energy):
    e = energy.double().flatten()
    d = e.numel()
    ranked = e.sort(descending=True).values
    total = e.sum()
    effective = total.square() / e.square().sum()
    return dict(d=d, effective_dimension=effective.item(),
                W=(effective / d).item(), isotropy_score=(effective / d).item(),
                whitened_dimension_fraction=(effective / d).item(),
                top10_energy_share=(ranked[:math.ceil(.1*d)].sum()/total).item(),
                directions_for_90pct=(torch.searchsorted(ranked.cumsum(0), .9*total).item()+1)/d)


def coordinates(operator, name, matrices, ua, uc):
    # Use the actual RMS-matched training operator, including its float32 powers.
    left, right = operator.data[name]
    transformed = left @ matrices @ right
    return uc.T @ transformed.double() @ ua


def diagnose(model, factors, operator, probe_x, probe_y, batch_size=cfg.PROBE_BATCH_SIZE):
    """Only model/factors/operator and held-out synthetic tensors enter this API.

    Gradients are materialized for one microbatch, then reduced and discarded.
    Weight rows are output channels; flattened input weights precede bias.
    No .backward(), parameter .grad write, optimizer call, or dataset access.
    """
    modules = layers(model)
    bases = {n: tuple(torch.linalg.eigh((f[k].double()+f[k].double().T)/2).eigenvectors
                      for k in ('A', 'C')) for n, f in factors.items()}
    sums = {n: torch.zeros(f['C'].shape[0], f['A'].shape[0],
                          device=probe_x.device, dtype=torch.float64) for n, f in factors.items()}
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def loss(p, x, y):
        return F.cross_entropy(functional_call(model, (p, buffers), (x[None],)), y[None])

    per_sample = vmap(grad(loss), in_dims=(None, 0, 0))
    for start in range(0, len(probe_x), batch_size):
        gs = per_sample(params, probe_x[start:start+batch_size], probe_y[start:start+batch_size])
        for n, module in modules.items():
            weight = gs[n+'.weight'].flatten(2)
            matrices = torch.cat((weight, gs[n+'.bias'].unsqueeze(-1)), dim=2)
            ua, uc = bases[n]
            z = coordinates(operator, n, matrices, ua, uc)
            sums[n] += z.square().sum(0)
        del gs, weight, matrices, z
    energies = {n: e.flatten()/len(probe_x) for n, e in sums.items()}
    rows = [dict(layer=n, probe_sample_count=len(probe_x), **energy_metrics(e))
            for n, e in energies.items()]
    total_d = sum(r['d'] for r in rows)
    aggregate = {key: sum(r['d']*r[key] for r in rows)/total_d
                 for key in ('W', 'top10_energy_share', 'directions_for_90pct')}
    aggregate['W_global'] = aggregate.pop('W')
    aggregate['probe_sample_count'] = len(probe_x)
    # Compress sorted aggregate normalized energies onto a fixed percentile grid.
    normalized = torch.cat([e/e.mean() for e in energies.values()])
    percentiles = torch.linspace(0, 1, 1001, device=probe_x.device, dtype=torch.float64)
    profile = torch.quantile(normalized, 1-percentiles).cpu().tolist()
    return aggregate, rows, profile

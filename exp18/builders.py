"""KFAC-U samples from Exp12; compressed statistics never request dense factors."""
import torch
from torch.nn import functional as F
from exp12.curvature import forward, layers, estimate
from exp13.builders import synthetic_cache
from exp18.operators import Factor, Operator
from exp18.lowrank import Sketch


def activation_samples(a, m):
    if isinstance(m, torch.nn.Conv2d):
        a = F.unfold(a, m.kernel_size, dilation=m.dilation, padding=m.padding,
                     stride=m.stride).transpose(1, 2).reshape(-1, m.weight[0].numel())
    return torch.cat((a, torch.ones_like(a[:, :1])), 1)


def samples(model, cache, seed, *, need_a, need_c):
    gen = torch.Generator(device=cache[0].device).manual_seed(seed) if need_c else None
    for x in cache:
        with torch.set_grad_enabled(need_c):
            z, acts, zs = forward(model, x)
        if need_c:
            p = z.detach().softmax(-1)
            y = torch.multinomial(torch.ones_like(p), 1, generator=gen)
            v = p - torch.zeros_like(p).scatter_(1, y, 1)
            grads = dict(zip(zs, torch.autograd.grad(z, tuple(zs.values()), v)))
        for name, m in layers(model).items():
            if need_a:
                yield name, 'A', activation_samples(acts[name], m)
            if need_c:
                b = grads[name].detach()
                if isinstance(m, torch.nn.Conv2d):
                    b = b.permute(0, 2, 3, 1).reshape(-1, b.shape[1])
                yield name, 'C', b


def build(model, method, cache, seed, epoch):
    diagnostics = []
    unique = sum(map(len, cache))
    passes = 1
    with torch.random.fork_rng(), torch.enable_grad():
        if method in ('refresh2', 'frozen'):
            dense, budget = estimate(model, cache, 'KFAC-U', seed=seed+20000+epoch)
            factors = {n: {s: Factor('dense', len(f), f) for s, f in fs.items()} for n, fs in dense.items()}
        else:
            kinds = dict(diag=('diagonal', 'diagonal'), a_only=('dense', 'identity'),
                         c_only=('identity', 'dense'), fullA_diagC=('dense', 'diagonal'),
                         diagA_fullC=('diagonal', 'dense'))
            rank = int(method[4:]) if method.startswith('rank') else None
            need_a, need_c = method != 'c_only', method != 'a_only'
            state, counts = {}, {}
            gen = torch.Generator(device=cache[0].device).manual_seed(seed+30000+epoch)
            for n, side, x in samples(model, cache, seed+20000+epoch, need_a=need_a, need_c=need_c):
                key = n, side
                kind = 'lowrank' if rank else kinds[method][side == 'C']
                if key not in state:
                    d = x.shape[1]
                    state[key] = (Sketch(d, rank, x.device, gen) if rank else
                                  x.new_zeros((d, d) if kind == 'dense' else (d,)))
                    counts[key] = 0
                if rank:
                    # Bound temporary sample projection storage for convolutional sites.
                    for chunk in x.split(8192):
                        state[key].first(chunk)
                elif kind != 'identity':
                    state[key].add_(x.T @ x if kind == 'dense' else x.square().sum(0))
                counts[key] += len(x)
            if rank:
                for sketch in state.values():
                    sketch.prepare()
                for n, side, x in samples(model, cache, seed+20000+epoch, need_a=need_a, need_c=need_c):
                    for chunk in x.split(8192):
                        state[n, side].second(chunk)
            factors = {n: {} for n in layers(model)}
            for n, m in layers(model).items():
                if not need_a:
                    factors[n]['A'] = Factor('identity', m.weight[0].numel()+1)
                if not need_c:
                    factors[n]['C'] = Factor('identity', m.weight.shape[0])
            for (n, side), value in state.items():
                if rank:
                    factor, diag = value.finish()
                    diagnostics.append(dict(layer=n, side=side, **diag))
                else:
                    kind = kinds[method][side == 'C']
                    factor = Factor(kind, value.shape[0], value/counts[n, side])
                factors[n][side] = factor
            passes = 2 if rank else 1
            budget = dict(forward_calls=passes*len(cache), vjp_calls=passes*len(cache) if need_c else 0,
                          reverse_vectors=passes*unique if need_c else 0)
        budget.pop('samples', None)
        budget.update(unique_samples=unique, processed_samples=passes*unique)
        operator = Operator(factors)
    return operator, {**operator.diagnostics, **{f'builder_{k}': v for k, v in budget.items()}}, diagnostics

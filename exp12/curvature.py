import torch
from torch import nn
from torch.nn import functional as F


def output_factor(p):
    h = torch.diag_embed(p) - p.unsqueeze(-1)*p.unsqueeze(-2)
    e, q = torch.linalg.eigh(h)
    # Softmax has one null direction; keep at most classes - 1 columns.
    return q[..., 1:] * e[..., 1:].clamp_min(0).sqrt().unsqueeze(-2)


def layers(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}


def forward(model, x):
    acts, zs = {}, {}
    def hook(name):
        def record(module, args, output):
            acts[name] = args[0].detach()
            zs[name] = output
        return record
    handles = [m.register_forward_hook(hook(n)) for n, m in layers(model).items()]
    z = model(x.detach().requires_grad_(True))
    for h in handles:
        h.remove()
    return z, acts, zs


def activation_sum(a, m):
    if isinstance(m, nn.Conv2d):
        a = F.unfold(a, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride).transpose(1, 2).reshape(-1, m.weight[0].numel())
    a = torch.cat([a, torch.ones_like(a[:, :1])], 1)
    return a.T @ a, len(a)


def covariance(v):
    if v.ndim == 4:
        v = v.permute(0, 2, 3, 1).reshape(-1, v.shape[1])
    return v.T @ v / len(v)


@torch.no_grad()
def pool_gate_sum(z):
    """Sum of joint ReLU/argmax indicators at each pre-pool site."""
    _, idx = F.max_pool2d(z.relu(), 2, 2, return_indices=True)
    mask = torch.zeros_like(z).flatten(2).scatter_(2, idx.flatten(2), 1).reshape_as(z)*(z > 0)
    return torch.einsum('bihw,bjhw->hwij', mask, mask)


def unpool_curvature(cov, gates):
    return cov.repeat_interleave(2, 0).repeat_interleave(2, 1) * gates


def linear_to_blocks(weight, cov, shape):
    w = weight.reshape(weight.shape[0], *shape)
    return torch.einsum('oihw,op,pjhw->hwij', w, cov, w)


def conv_to_blocks(weight, cov):
    """Stride-one, padding-one 3x3 convolution, dropping cross-site blocks."""
    h, w = cov.shape[:2]
    out = cov.new_zeros(h, w, weight.shape[1], weight.shape[1])
    for dy in range(3):
        for dx in range(3):
            kernel = weight[:, :, dy, dx]
            v = torch.einsum('oi,hwop,pj->hwij', kernel, cov, kernel)
            padded = F.pad(v.permute(2, 3, 0, 1), (1, 1, 1, 1))
            out += padded[:, :, 2-dy:h+2-dy, 2-dx:w+2-dx].permute(2, 3, 0, 1)
    return out


@torch.no_grad()
def kfra(model, output_curvature, gates, method):
    """Recursion on compact expected statistics; no forward or sample cache."""
    modules = layers(model)
    names = list(modules)
    c = output_curvature
    out = {names[-1]: c}
    if method == 'KFRA':
        assert isinstance(model, nn.Sequential)
        children = list(model.children())
        assert all(isinstance(m, nn.Linear if i % 2 == 0 else nn.ReLU)
                   for i, m in enumerate(children))
        assert isinstance(children[-1], nn.Linear)
        for name, next_name in reversed(list(zip(names[:-1], names[1:]))):
            w = modules[next_name].weight
            c = (w.T @ c @ w) * gates[name]
            out[name] = c
        return out
    assert method == 'KFRA-block'
    assert names == ['conv1', 'conv2', 'fc1', 'fc2']
    w = model.fc2.weight
    c = (w.T @ c @ w) * gates['fc1']
    out['fc1'] = c
    c = linear_to_blocks(model.fc1.weight, c, (32, 7, 7))
    c = unpool_curvature(c, gates['conv2'])
    out['conv2'] = c.mean((0, 1))
    c = conv_to_blocks(model.conv2.weight, c)
    c = unpool_curvature(c, gates['conv1'])
    out['conv1'] = c.mean((0, 1))
    return out


def estimate(model, cache, method, k=1, seed=0, label_cache=None):
    assert method in ('KFLR', 'KFAC-U', 'KFAC-M', 'KFRA', 'KFRA-block')
    assert label_cache is None or (method == 'KFAC-U' and k == 1)
    recursive = method in ('KFRA', 'KFRA-block')
    modules = layers(model)
    total, gates, activation_counts = {}, {}, {}
    count = sum(len(x) for x in cache)
    gen = torch.Generator(device=cache[0].device).manual_seed(seed)
    calls = vectors = forwards = 0
    hsum = None
    for batch, x in enumerate(cache):
        with torch.set_grad_enabled(not recursive):
            z, acts, zs = forward(model, x)
        forwards += 1
        p = z.detach().softmax(-1)
        cs = {n: z.new_zeros(m.weight.shape[0], m.weight.shape[0]) for n, m in modules.items()}
        if recursive:
            h = (torch.diag_embed(p)-p.unsqueeze(-1)*p.unsqueeze(-2)).sum(0)
            hsum = h if hsum is None else hsum+h
            for n in list(zs)[:-1]:
                if zs[n].ndim == 4:
                    g = pool_gate_sum(zs[n])
                else:
                    mask = (zs[n] > 0).to(p.dtype)
                    g = mask.T @ mask
                gates[n] = g if n not in gates else gates[n]+g
        else:
            rank = z.shape[-1]-1 if method == 'KFLR' else k
            if method == 'KFLR':
                s = output_factor(p)
            for j in range(rank):
                if method == 'KFLR':
                    v = s[:, :, j]
                else:
                    probs = torch.ones_like(p) if method == 'KFAC-U' else p
                    y = label_cache[batch].reshape(-1, 1) if label_cache is not None else torch.multinomial(probs, 1, generator=gen)
                    v = p - torch.zeros_like(p).scatter_(1, y, 1)
                grads = torch.autograd.grad(z, tuple(zs.values()), v, retain_graph=j < rank-1)
                calls += 1
                vectors += len(x)
                for n, grad in zip(zs, grads):
                    cs[n] += covariance(grad.detach()) / (1 if method == 'KFLR' else k)
                del grads, grad
        for n, m in modules.items():
            a, positions = activation_sum(acts[n], m)
            if n not in total:
                total[n] = {'A': torch.zeros_like(a), 'C': torch.zeros_like(cs[n])}
                activation_counts[n] = 0
            total[n]['A'] += a
            activation_counts[n] += positions
            total[n]['C'] += cs[n] * len(x)
        del z, acts, zs, cs, a
    for n, f in total.items():
        f['A'] /= activation_counts[n]
        f['C'] /= count
    if recursive:
        for n, c in kfra(model, hsum/count, {n: g/count for n, g in gates.items()}, method).items():
            total[n]['C'] = c
    return total, {'reverse_vectors': vectors, 'vjp_calls': calls,
                   'forward_calls': forwards, 'samples': count}


def estimate_kfac_with_labels(model, cache, label_cache):
    return estimate(model, cache, 'KFAC-U', label_cache=label_cache)

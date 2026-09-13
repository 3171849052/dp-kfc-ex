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
    z = model(x.requires_grad_(True))
    for h in handles:
        h.remove()
    return z, acts, zs


def activation(a, m):
    if isinstance(m, nn.Conv2d):
        a = F.unfold(a, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride).transpose(1, 2).reshape(-1, m.weight[0].numel())
    a = torch.cat([a, torch.ones_like(a[:, :1])], 1)
    return a.T @ a / len(a)


def covariance(v):
    if v.ndim == 4:
        v = v.permute(0, 2, 3, 1).reshape(-1, v.shape[1])
    return v.T @ v / len(v)


@torch.no_grad()
def unpool_curvature(cov, z):
    """Expected local J^T C J for ReLU + 2x2 maxpool, no cross-site blocks."""
    b, ch, h, w = z.shape
    _, idx = F.max_pool2d(z.relu(), 2, 2, return_indices=True)
    mask = torch.zeros_like(z).flatten(2).scatter_(2, idx.flatten(2), 1).reshape_as(z)*(z > 0)
    expanded = cov.repeat_interleave(2, 0).repeat_interleave(2, 1)
    gates = torch.einsum('bihw,bjhw->hwij', mask, mask)/b
    return expanded * gates


@torch.no_grad()
def kfra(model, cache):
    # Average output curvature globally before recursion, not separately per batch.
    records = [forward(model, x)[1:] for x in cache]
    acts = {n: torch.cat([a[n] for a, _ in records]) for n in layers(model)}
    zs = {n: torch.cat([z[n] for _, z in records]) for n in layers(model)}
    last = list(zs)[-1]
    p = zs[last].softmax(-1)
    c = (torch.diag_embed(p)-p.unsqueeze(-1)*p.unsqueeze(-2)).mean(0)
    out = {last: c}
    if len(zs) == 1:
        return out
    # Explicit SimpleCNN recursion. Covariances remain channel blocks after flatten.
    assert list(zs) == ['conv1', 'conv2', 'fc1', 'fc2']
    gate = (zs['fc1'] > 0).to(c.dtype)
    c = (model.fc2.weight.T @ c @ model.fc2.weight) * (gate.T @ gate/len(gate))
    out['fc1'] = c
    w = model.fc1.weight.reshape(128, 32, 7, 7)
    c = torch.einsum('oihw,op,pjhw->hwij', w, c, w)
    c = unpool_curvature(c, zs['conv2'])
    out['conv2'] = c.mean((0, 1))
    prev = c.new_zeros(28, 28, 16, 16)
    # Each offset contributes W_offset^T C_site W_offset to an input site.
    for dy in range(3):
        for dx in range(3):
            w = model.conv2.weight[:, :, dy, dx]
            v = torch.einsum('oi,hwop,pj->hwij', w, c, w)
            padded = F.pad(v.permute(2, 3, 0, 1), (1, 1, 1, 1))
            prev[:14, :14] += padded[:, :, 2-dy:16-dy, 2-dx:16-dx].permute(2, 3, 0, 1)
    c = unpool_curvature(prev[:14, :14], zs['conv1'])
    out['conv1'] = c.mean((0, 1))
    return out


def estimate(model, cache, method, k=1, seed=0):
    modules = layers(model)
    total = {}
    count = sum(len(x) for x in cache)
    gen = torch.Generator(device=cache[0].device).manual_seed(seed)
    calls = vectors = 0
    for x in cache:
        z, acts, zs = forward(model, x)
        p = z.detach().softmax(-1)
        cs = {n: z.new_zeros(m.weight.shape[0], m.weight.shape[0]) for n, m in modules.items()}
        rank = z.shape[-1]-1 if method == 'KFLR' else k
        if method == 'KFLR':
            s = output_factor(p)
        if method != 'KFRA':
            for j in range(rank):
                if method == 'KFLR':
                    v = s[:, :, j]
                else:
                    probs = torch.ones_like(p) if method == 'KFAC-U' else p
                    y = torch.multinomial(probs, 1, generator=gen)
                    v = p - torch.zeros_like(p).scatter_(1, y, 1)
                grads = torch.autograd.grad(z, tuple(zs.values()), v, retain_graph=j < rank-1)
                calls += 1
                vectors += len(x)
                for n, g in zip(zs, grads):
                    cs[n] += covariance(g.detach()) / (1 if method == 'KFLR' else k)
        for n, m in modules.items():
            a = activation(acts[n], m)
            if n not in total:
                total[n] = {'A': torch.zeros_like(a), 'C': torch.zeros_like(cs[n])}
            total[n]['A'] += a * (len(x)/count)
            total[n]['C'] += cs[n] * (len(x)/count)
    if method == 'KFRA':
        for n, c in kfra(model, cache).items():
            total[n]['C'] = c
    return total, {'reverse_vectors': vectors, 'vjp_calls': calls, 'samples': count}

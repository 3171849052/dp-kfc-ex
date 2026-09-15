"""Original full KFC and forward-only A operators with exact/structured clipping."""
import math
import torch
from torch.nn import functional as F
from opacus import GradSampleModule
from exp12.curvature import layers, forward, activation_sum, covariance
from exp13.builders import synthetic_cache
from exp13.operator import Operator as FullOperator, inverse_sqrt
from exp13.ghost import GhostNorm as StructuredGhost, noise_and_step
from exp19.config import DAMPING, METHODS
from exp19.profiling import timed


class AOperator(FullOperator):
    def __init__(self, factors, power):
        self.data = {}
        raw = ref = 0.
        for n, f in factors.items():
            a = f['A'].double()
            e, q = torch.linalg.eigh((a+a.T)/2)
            e = e.clamp_min(0)
            raw += f['output_dimension'] * (e*(e+DAMPING).pow(-2*power)).sum().item()
            ref += f['output_dimension'] * (e*(e+DAMPING).pow(-1)).sum().item()
            self.data[n] = ((q*(e+DAMPING).pow(-power))@q.T).float()
        self.scale = math.sqrt(ref/raw) if power == .25 else 1.
        self.moments = dict(m_raw=raw, m_reference=ref, scale_match=self.scale)

    def transform_activation(self, name, a):
        return self.scale * (self.data[name] @ a)

    def transform_backprop(self, name, b):
        return b

    def transform_matrix(self, name, g):
        return self.scale * (g @ self.data[name])


def uniform_labels(p, generator):
    return torch.multinomial(torch.ones_like(p), 1, generator=generator)


def build_from_cache(model, method, cache, seed, epoch):
    full = method in METHODS[:2]
    device = cache[0].device
    stats = dict(activation_forward_seconds=0., curvature_backward_seconds=0.,
                 factor_accumulation_seconds=0., matrix_function_seconds=0.,
                 builder_forward_calls=0, builder_vjp_calls=0, builder_reverse_vectors=0,
                 builder_samples=sum(map(len, cache)))
    factors, counts = {}, {}
    generator = torch.Generator(device=device).manual_seed(seed+20000+epoch) if full else None
    for x in cache:
        with timed(stats, 'activation_forward_seconds', device), torch.set_grad_enabled(full):
            z, acts, zs = forward(model, x)
        stats['builder_forward_calls'] += 1
        if full:
            with timed(stats, 'curvature_backward_seconds', device):
                p = z.detach().softmax(-1)
                y = uniform_labels(p, generator)
                v = p-torch.zeros_like(p).scatter_(1, y, 1)
                grads = torch.autograd.grad(z, tuple(zs.values()), v)
            stats['builder_vjp_calls'] += 1
            stats['builder_reverse_vectors'] += len(x)
        with timed(stats, 'factor_accumulation_seconds', device), torch.no_grad():
            for i, (n, m) in enumerate(layers(model).items()):
                a, count = activation_sum(acts[n], m)
                if n not in factors:
                    factors[n] = dict(A=torch.zeros_like(a), output_dimension=m.weight.shape[0])
                    counts[n] = 0
                    if full:
                        factors[n]['C'] = a.new_zeros(m.weight.shape[0], m.weight.shape[0])
                factors[n]['A'].add_(a)
                counts[n] += count
                if full:
                    factors[n]['C'].add_(covariance(grads[i].detach()), alpha=len(x))
            del a
        del z, acts, zs
        if full:
            del grads, p, y, v
    with timed(stats, 'factor_accumulation_seconds', device):
        for n, f in factors.items():
            f['A'].div_(counts[n])
            if full:
                f['C'].div_(stats['builder_samples'])
    with timed(stats, 'matrix_function_seconds', device):
        operator = FullOperator(factors) if full else AOperator(factors, .5 if method == METHODS[2] else .25)
    tensors = [t for pair in operator.data.values() for t in pair] if full else list(operator.data.values())
    stats.update(operator_state_bytes=sum(t.numel()*t.element_size() for t in tensors)+(0 if full else 8),
                 stored_scalar_count=sum(t.numel() for t in tensors)+(0 if full else 1))
    if not full:
        stats.update(operator.moments)
    return operator, stats


class GhostNorm(StructuredGhost):
    """Reuse production spatial-Gram norm; retain only per-layer scalar norms."""
    @torch.no_grad()
    def norm_hook(self, name, module, backprop):
        total = self.norm_sq
        self.norm_sq = torch.zeros_like(total)
        super().norm_hook(name, module, backprop)
        self.layer_sq[name] = self.norm_sq
        self.norm_sq = total + self.norm_sq


def ghost_aggregate(model, hooks, x, y):
    stats = dict(grad_sample_bytes=0)
    with timed(stats, 'ghost_first_pass_seconds', x.device):
        model.zero_grad(set_to_none=True)
        hooks.norm_sq = x.new_zeros(len(x))
        hooks.layer_sq = {}
        hooks.enabled = True
        losses = F.cross_entropy(model(x), y, reduction='none')
        losses.sum().backward()
        norms = hooks.norm_sq.clamp_min(0).sqrt()
        factors = (1/(norms+1e-6)).clamp(max=1).detach()
    with timed(stats, 'ghost_second_pass_seconds', x.device):
        hooks.enabled = False
        model.zero_grad(set_to_none=True)
        (F.cross_entropy(model(x), y, reduction='none')*factors).sum().backward()
    with timed(stats, 'aggregate_transform_seconds', x.device):
        hooks.operator.transform_aggregate_gradient(model)
    return losses.detach().sum(), norms, factors, hooks.layer_sq, stats


def exact_aggregate(wrapper, operator, x, y):
    stats = {}
    with timed(stats, 'exact_forward_backward_seconds', x.device):
        wrapper.zero_grad(set_to_none=True)
        loss = F.cross_entropy(wrapper(x), y, reduction='sum')
        loss.backward()
    stats['grad_sample_bytes'] = sum(p.grad_sample.numel()*p.grad_sample.element_size() for p in wrapper.parameters())
    with timed(stats, 'per_sample_precondition_clip_seconds', x.device), torch.no_grad():
        layer_sq = {}
        for n, m in layers(wrapper._module).items():
            g = torch.cat((m.weight.grad_sample.flatten(2), m.bias.grad_sample.unsqueeze(-1)), -1)
            g = operator.transform_matrix(n, g)
            m.weight.grad_sample = g[:, :, :-1].reshape_as(m.weight.grad_sample).contiguous()
            m.bias.grad_sample = g[:, :, -1].contiguous()
            layer_sq[n] = g.square().sum((1, 2))
            del g
        norms = sum(layer_sq.values()).sqrt()
        factors = (1/(norms+1e-6)).clamp(max=1)
        for p in wrapper.parameters():
            p.grad = torch.einsum('b,bp->p', factors, p.grad_sample.flatten(1)).reshape_as(p)
            p.grad_sample = None
    return loss.detach(), norms, factors, layer_sq, stats

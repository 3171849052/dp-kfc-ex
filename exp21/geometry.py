"""Corrected Exp21 A geometry; Exp20 matrix function/scale matching is unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from exp20.methods import AOperator as Exp20AOperator, synthetic_cache
from exp21.handlers import LINEAR_LAYOUTS
from exp21.profiling import timed


def affine(module):
    return isinstance(module, (nn.Linear, nn.Conv2d)) or type(module) in LINEAR_LAYOUTS


def activation_matrix(x, module):
    if isinstance(module, nn.Conv2d):
        x = F.unfold(x, module.kernel_size, dilation=module.dilation,
                     padding=module.padding, stride=module.stride).transpose(1, 2)
    a = x.reshape(-1, x.shape[-1])
    if module.bias is not None:
        a = torch.cat((a, torch.ones_like(a[:, :1])), -1)
    return a


class AOperator(Exp20AOperator):
    @torch.no_grad()
    def transform_aggregate_gradient(self, model):
        from exp21.bk import transform_aggregate
        transform_aggregate(model, self)


@torch.no_grad()
def build_from_cache(model, power, cache, seed, epoch, profiler=None, layer_names=None):
    """Forward-only builder. Shared weights default to explicit identity geometry."""
    owners = {}
    for name, module in model.named_modules():
        for p in module.parameters(recurse=False):
            if p.requires_grad:
                owners.setdefault(id(p), []).append(name)
    candidates = {n: m for n, m in model.named_modules()
                  if affine(m) and any(p.requires_grad for p in m.parameters(recurse=False))}
    shared = {n for names in owners.values() if len(names) > 1 for n in names}
    selected = set(candidates)-shared if layer_names is None else set(layer_names)
    if not selected <= candidates.keys():
        raise ValueError(f'A geometry requires Linear/Conv2d/registered Conv1D: {selected-candidates.keys()}')
    if selected & shared:
        raise NotImplementedError('Shared weights must use identity geometry; exclude their modules from A coverage')
    if not selected:
        raise ValueError('No affine layers selected for A geometry; use operator=None')
    modules = {n: m for n, m in candidates.items() if n in selected}
    for n, m in modules.items():
        if isinstance(m, nn.Conv2d) and (m.groups != 1 or m.padding_mode != 'zeros'):
            raise NotImplementedError('A builder requires groups=1 and zero padding')
        if not all(p.requires_grad for p in m.parameters(recurse=False)):
            raise NotImplementedError(f'Partial freezing within preconditioned layer: {n}')
    device = cache[0].device
    stats = dict(builder_forward_calls=0, builder_vjp_calls=0, builder_reverse_vectors=0,
                 builder_samples=sum(map(len, cache)), curvature_backward_seconds=0.)
    factors, counts = {}, {}
    def capture(name):
        def hook(module, args, output):
            with timed(profiler, 'factor_accumulation_seconds', device):
                a = activation_matrix(args[0].detach(), module)
                if name not in factors:
                    out_dim = module.weight.shape[1] if type(module) in LINEAR_LAYOUTS else module.weight.shape[0]
                    factors[name] = dict(A=a.new_zeros(a.shape[-1], a.shape[-1]), output_dimension=out_dim)
                    counts[name] = 0
                factors[name]['A'].add_(a.T @ a)
                counts[name] += len(a)
        return hook
    handles = [m.register_forward_hook(capture(n)) for n, m in modules.items()]
    try:
        for x in cache:
            with timed(profiler, 'activation_and_factor_forward_seconds', device):
                model(x)
            stats['builder_forward_calls'] += 1
    finally:
        for h in handles:
            h.remove()
    if set(factors) != selected:
        raise ValueError(f'A calibration did not execute selected layers: {selected-set(factors)}')
    for n, factor in factors.items():
        factor['A'].div_(counts[n])
    with timed(profiler, 'matrix_function_seconds', device):
        op = AOperator(factors, power)
    stats.update(op.moments)
    stats.update(op.diagnostics)
    stats['operator_state_bytes'] = sum(t.numel()*t.element_size() for t in op.data.values())+8
    stats['builder_preconditioned_layers'] = list(modules)
    stats['builder_identity_shared_layers'] = sorted(shared)
    return op, stats

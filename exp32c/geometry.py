"""Exp21 Conv2d im2col factors with Exp22 matrix functions and pink noise."""
from exp32c import config as cfg
import torch
from torch import nn
from exp21.geometry import activation_matrix
from exp22.geometry import matrix_function, synthetic_stream


def conv_modules(model):
    modules = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)
               and m.weight.requires_grad}
    assert all(m.groups == 1 and m.padding_mode == 'zeros' for m in modules.values())
    assert {id(p) for m in modules.values() for p in m.parameters()} == {
        id(p) for p in model.parameters() if p.requires_grad}
    return modules


class Operator:
    def __init__(self, factors, method, damping, power):
        self.scale = 1.0
        self.data = {n: matrix_function(f['A'], power if method == 'dp_kfc_a' else 0.5, damping)
                     for n, f in factors.items()}
        self.left = {n: matrix_function(f['G'], 0.5, damping) for n, f in factors.items()
                     if method == 'dp_kfc'}
        self.operator_state_bytes = sum(t.numel()*t.element_size()
                                       for t in [*self.data.values(), *self.left.values()])

    def transform_activation(self, name, a):
        # Exp21 Record passes [batch, input+bias, spatial].
        return self.data[name] @ a

    def transform_backprop(self, name, b):
        return (self.left[name] @ b.flatten(2)).reshape_as(b) if name in self.left else b

    def transform_matrix(self, name, g):
        result = g @ self.data[name]
        return self.left[name] @ result if name in self.left else result


def build_from_batches(model, method, batches, seed, epoch, *, damping, power=cfg.A_POWER):
    stats = dict(builder_forward_calls=0, builder_vjp_calls=0, builder_reverse_vectors=0,
                 builder_samples=0, builder_logical_batches=0, operator_state_bytes=0)
    if method == 'dp_adam':
        return None, stats
    assert method in ('dp_kfc', 'dp_kfc_a') and damping > 0
    modules = conv_modules(model)
    factors, counts = {}, {}
    full = method == 'dp_kfc'
    def capture(name):
        def hook(module, args, output):
            with torch.no_grad():
                a = activation_matrix(args[0].detach(), module)
                if name not in factors:
                    factors[name] = {'A': a.new_zeros(a.shape[1], a.shape[1])}
                    if full:
                        factors[name]['G'] = a.new_zeros(module.out_channels, module.out_channels)
                    counts[name] = 0
                factors[name]['A'].add_(a.T @ a)
                counts[name] += len(a)
            if full:
                def backward(grad):
                    with torch.no_grad():
                        b = grad.detach().permute(0, 2, 3, 1).reshape(-1, module.out_channels)
                        factors[name]['G'].add_(b.T @ b)
                output.register_hook(backward)
        return hook
    handles = [m.register_forward_hook(capture(n)) for n, m in modules.items()]
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed + 20000 + epoch)
    try:
        for x in batches:
            assert len(x) <= cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE
            with torch.set_grad_enabled(full):
                logits = model(x)
                if full:
                    y = torch.randint(cfg.NUM_CLASSES, (len(x),), device=x.device, generator=generator)
                    torch.nn.functional.cross_entropy(logits, y, reduction='sum').backward()
                    model.zero_grad(set_to_none=True)
            stats['builder_forward_calls'] += 1
            stats['builder_logical_batches'] += 1
            stats['builder_samples'] += len(x)
            stats['builder_vjp_calls'] += int(full)
            stats['builder_reverse_vectors'] += len(x) if full else 0
            del logits, x
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    for name in modules:
        for value in factors[name].values():
            value.div_(counts[name])
    operator = Operator(factors, method, damping, power)
    stats.update(operator_state_bytes=operator.operator_state_bytes, scale_match=1.0,
                 builder_preconditioned_layers=list(modules))
    return operator, stats

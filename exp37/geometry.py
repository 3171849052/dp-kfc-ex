"""Trace-only synthetic calibration; no covariance or matrix operator."""
import math
import torch
from exp37.config import cfg
from exp22 import geometry as reference
from exp22.handlers import flatten_linear_input


class LayerScaleOperator:
    """Only layer->scalar state. Scaling augmented activations scales W and bias."""
    def __init__(self, scales):
        self.data = dict(scales)

    def transform_activation(self, name, activation):
        return activation * self.data[name]

    def transform_gradient(self, name, gradient):
        return gradient * self.data[name]

    transform_matrix = transform_gradient


def scalar_scales(traces_a, traces_g, dimensions, power, damping):
    raw = {name: ((a + damping) if traces_g is None else
                  (a + damping) * (traces_g[name] + damping)) ** (-power)
           for name, a in traces_a.items()}
    c = math.sqrt(sum(dimensions.values()) /
                  sum(dimensions[n] * r * r for n, r in raw.items()))
    return raw, {n: c * r for n, r in raw.items()}


def scale_statistics(scales):
    values = torch.tensor(list(scales.values()), dtype=torch.float64)
    return dict(scale_min=values.min().item(), scale_max=values.max().item(),
                **{f'scale_p{p}': values.quantile(p / 100).item() for p in (10, 50, 90)})


def build_from_batches(model, method, batches, seed=0, epoch=1, damping=.001, power=.4):
    modules = reference.affine_modules(model)
    if method == 'baseline':
        operator, stats = reference.build_from_batches(model, 'dp_sgd', batches)
        stats.update(scale_statistics({n: 1.0 for n in modules}))
        for n in modules:
            stats.update({f'raw_scale_{n}': 1.0, f'scale_{n}': 1.0})
        return operator, stats
    assert method in ('trace_a', 'trace_ag') and damping == cfg.DAMPING
    ag = method == 'trace_ag'
    device = next(model.parameters()).device
    sums_a = {n: torch.zeros((), dtype=torch.float64, device=device) for n in modules}
    sums_g = {n: torch.zeros_like(v) for n, v in sums_a.items()} if ag else None
    counts = {n: 0 for n in modules}

    def capture(name):
        def hook(module, args, output):
            a = flatten_linear_input(args[0].detach(), module)
            sums_a[name].add_(a.square().sum(dtype=torch.float64))
            counts[name] += a.shape[0]
            if ag:
                def backward(error):
                    sums_g[name].add_(error.detach().square().sum(dtype=torch.float64))
                output.register_hook(backward)
        return hook

    handles = [module.register_forward_hook(capture(n)) for n, module in modules.items()]
    forwards = logical_batches = samples = 0
    if ag:
        label_generator = torch.Generator(device=device).manual_seed(seed + 20000 + epoch)
        num_classes = getattr(model, 'num_classes', None)
        if num_classes is None and hasattr(model, 'head'):
            num_classes = model.head.out_features
    try:
        for logical_x in batches:
            logical_batches += 1
            samples += len(logical_x)
            if ag:
                labels = reference.synthetic_labels(seed, epoch, device, len(logical_x),
                    num_classes or reference.NUM_CLASSES, label_generator)
            for start in range(0, len(logical_x), cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE):
                x = logical_x[start:start + cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE]
                with torch.set_grad_enabled(ag):
                    if ag:
                        model.zero_grad(set_to_none=True)
                    logits = model(x)
                    if ag:
                        torch.nn.functional.cross_entropy(logits, labels[start:start + len(x)],
                                                          reduction='sum').backward()
                forwards += 1
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    assert all(counts.values())
    a = {n: sums_a[n].item() / counts[n] / (m.in_features + (m.bias is not None))
         for n, m in modules.items()}
    g = {n: sums_g[n].item() / counts[n] / m.out_features for n, m in modules.items()} if ag else None
    dimensions = {n: m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
                  for n, m in modules.items()}
    raw, scales = scalar_scales(a, g, dimensions, power, damping)
    stats = dict(builder_forward_calls=forwards, builder_logical_batches=logical_batches,
                 builder_vjp_calls=forwards if ag else 0, builder_reverse_vectors=samples if ag else 0,
                 builder_samples=samples, preconditioned_layers=sorted(modules),
                 operator_state_bytes=8 * len(scales), **scale_statistics(scales))
    for n in modules:
        stats.update({f'trace_a_per_dim_{n}': a[n], f'raw_scale_{n}': raw[n], f'scale_{n}': scales[n]})
        if ag:
            stats[f'trace_g_per_dim_{n}'] = g[n]
    return LayerScaleOperator(scales), stats

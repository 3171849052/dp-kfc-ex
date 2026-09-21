"""Exp22 synthetic covariance estimation with explicit LoRA selection."""
import torch
from torch import nn
from typing import Iterable
from exp22.geometry import (_selected, matrix_function, build_full_operator, synthetic_stream)
from exp22.handlers import flatten_linear_input
from exp31.config import A_POWER, SYNTHETIC_PHYSICAL_BATCH_SIZE
from exp31.lora import selected_layers

class AOnlyOperator:
    def __init__(self, factors, power, damping):
        self.factors, self.power, self.damping = factors, power, damping
        self.scale = 1.0
        self.data = {n: matrix_function(f['A'], power, damping) for n, f in factors.items()}
        self.preconditioned_layers = sorted(self.data)
        self.operator_state_bytes = sum(t.numel() * t.element_size() for t in self.data.values())
        self.moments = {'scale_match': 1.0}
        self.diagnostics = {}

    def transform_activation(self, name, activation):
        return activation @ self.data[name].T

    def transform_gradient(self, name, gradient):
        return gradient @ self.data[name]

    transform_matrix = transform_gradient

@torch.no_grad()
def build_a_operator(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    power: float = A_POWER,
    damping: float = 1e-3,
    layer_names: Iterable[str] | None = None,
) -> tuple[AOnlyOperator, dict]:
    modules = _selected(model, layer_names)
    factors = {
        name: {
            "A": torch.zeros(module.in_features + (1 if module.bias is not None else 0),
                              module.in_features + (1 if module.bias is not None else 0),
                              device=next(module.parameters()).device),
            "output_dimension": module.out_features,
        }
        for name, module in modules.items()
    }
    counts = {name: 0 for name in modules}
    handles = []

    def capture(name):
        def hook(module, args, output):
            a = flatten_linear_input(args[0].detach(), module)
            factors[name]["A"].add_(a.T @ a)
            counts[name] += a.shape[0]
        return hook

    handles = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
    forward_calls = 0
    logical_batches = 0
    samples = 0
    try:
        for logical_x in batches:
            logical_batches += 1
            samples += len(logical_x)
            for start in range(0, len(logical_x), SYNTHETIC_PHYSICAL_BATCH_SIZE):
                model(logical_x[start:start + SYNTHETIC_PHYSICAL_BATCH_SIZE])
                forward_calls += 1
            del logical_x
    finally:
        for handle in handles:
            handle.remove()
    if any(counts[name] == 0 for name in modules):
        raise RuntimeError(f"calibration did not execute {sorted(name for name, n in counts.items() if n == 0)}")
    for name in modules:
        factors[name]["A"].div_(counts[name])
    operator = AOnlyOperator(factors, power, damping)
    stats = {
        "builder_forward_calls": forward_calls,
        "builder_logical_batches": logical_batches,
        "builder_vjp_calls": 0,
        "builder_reverse_vectors": 0,
        "builder_samples": samples,
        "preconditioned_layers": sorted(modules),
        "operator_state_bytes": operator.operator_state_bytes,
        **operator.moments,
        **operator.diagnostics,
    }
    return operator, stats


def build_geometry(model, method, batches, seed, epoch, *, damping):
    names = selected_layers(model)
    if method == 'dp_adamw':
        assert damping is None
        return None, {'operator_state_bytes': 0, 'preconditioned_layers': []}
    if method == 'dp_kfc_a':
        return build_a_operator(model, batches, power=A_POWER, damping=damping, layer_names=names)
    if method == 'dp_kfc':
        return build_full_operator(model, batches, seed, epoch, damping=damping, layer_names=names)
    raise ValueError(method)

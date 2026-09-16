"""A-only and full synthetic KFAC geometry for Exp22."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from .config import (
    A_POWER,
    DAMPING,
    NUM_CLASSES,
    SYNTHETIC_ALPHA,
    SYNTHETIC_BATCHES,
    SYNTHETIC_BATCH_SIZE,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from .handlers import flatten_linear_input


def affine_modules(model: nn.Module) -> dict[str, nn.Linear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(parameter.requires_grad for parameter in module.parameters(recurse=False))
    }


def matrix_function(covariance: torch.Tensor, power: float, damping: float = DAMPING) -> torch.Tensor:
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    eigenvalues = eigenvalues.clamp_min(0)
    return ((eigenvectors * (eigenvalues + damping).pow(-power)) @ eigenvectors.T).to(covariance.dtype)


def inverse_sqrt(covariance: torch.Tensor, damping: float = DAMPING) -> torch.Tensor:
    return matrix_function(covariance, 0.5, damping)


class AOnlyOperator:
    def __init__(self, factors: dict[str, dict], power: float = A_POWER, damping: float = DAMPING):
        self.power = power
        self.damping = damping
        self.factors = factors
        self.data: dict[str, torch.Tensor] = {}
        self.preconditioned_layers = sorted(factors)
        raw = torch.zeros((), dtype=torch.float64, device=next(iter(factors.values()))["A"].device)
        reference = torch.zeros_like(raw)
        spectra, gains = [], []
        for name, factor in factors.items():
            eigenvalues, eigenvectors = torch.linalg.eigh(((factor["A"] + factor["A"].T) * 0.5).double())
            eigenvalues = eigenvalues.clamp_min(0)
            raw += factor["output_dimension"] * (eigenvalues * (eigenvalues + damping).pow(-2 * power)).sum()
            reference += factor["output_dimension"] * (eigenvalues * (eigenvalues + damping).pow(-1)).sum()
            self.data[name] = ((eigenvectors * (eigenvalues + damping).pow(-power)) @ eigenvectors.T).to(
                factor["A"].dtype
            )
            spectra.append(eigenvalues * (eigenvalues + damping).pow(-2 * power))
            gains.append((eigenvalues + damping).pow(-power))
        raw_value, reference_value = raw.item(), reference.item()
        self.scale = 1.0 if power == 0.5 else math.sqrt(reference_value / raw_value)
        self.moments = {"m_p": raw_value, "m_reference": reference_value, "scale_match": self.scale}
        eig = torch.cat(spectra) * self.scale**2
        gain = torch.cat(gains) * self.scale
        q = torch.tensor((0.1, 0.5, 0.9, 0.99), device=eig.device, dtype=eig.dtype)
        values = torch.cat((gain.quantile(q), eig.quantile(q))).cpu().tolist()
        self.diagnostics = {
            f"{prefix}_p{p}": value
            for prefix, offset in (("operator_gain", 0), ("transformed_eig", 4))
            for p, value in zip((10, 50, 90, 99), values[offset : offset + 4])
        }
        self.diagnostics["transformed_condition_proxy"] = (
            values[6] / values[4] if values[4] > 0 else float("inf")
        )
        self.operator_state_bytes = sum(t.numel() * t.element_size() for t in self.data.values()) + 8

    def transform_activation(self, name: str, activation: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.matmul(activation, self.data[name].T)

    def transform_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.matmul(gradient, self.data[name])

    transform_matrix = transform_gradient


class FullKFACOperator:
    def __init__(self, factors: dict[str, dict], damping: float = DAMPING):
        self.damping = damping
        self.factors = factors
        self.data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, factor in factors.items():
            ug = inverse_sqrt(factor["G"], damping)
            ua = inverse_sqrt(factor["A"], damping)
            self.data[name] = (ug, ua)
        self.preconditioned_layers = sorted(self.data)
        self.operator_state_bytes = sum(t.numel() * t.element_size() for pair in self.data.values() for t in pair)

    def transform_activation(self, name: str, activation: torch.Tensor) -> torch.Tensor:
        return torch.matmul(activation, self.data[name][1].T)

    def transform_backprop(self, name: str, backprop: torch.Tensor) -> torch.Tensor:
        return torch.matmul(backprop, self.data[name][0].T)

    def transform_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        ug, ua = self.data[name]
        return torch.matmul(ug, torch.matmul(gradient, ua))

    transform_matrix = transform_gradient


def _pink_noise(batch_size: int, device: torch.device, alpha: float = SYNTHETIC_ALPHA) -> torch.Tensor:
    from dp_kfac.optimizer import generate_pink_noise

    return generate_pink_noise(batch_size, (3, 32, 32), device, alpha=alpha)


def synthetic_cache(
    seed: int,
    epoch: int,
    device: torch.device | str,
    batches: int = SYNTHETIC_BATCHES,
    batch_size: int = SYNTHETIC_BATCH_SIZE,
    alpha: float = SYNTHETIC_ALPHA,
) -> list[torch.Tensor]:
    device = torch.device(device)
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed + 10000 + epoch)
        if device.type == "cuda":
            torch.cuda.default_generators[devices[0]].manual_seed(seed + 10000 + epoch)
        return [_pink_noise(batch_size, device, alpha) for _ in range(batches)]


def synthetic_labels(
    seed: int,
    epoch: int,
    device: torch.device | str,
    batch_size: int,
    num_classes: int = NUM_CLASSES,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    device = torch.device(device)
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed + 20000 + epoch)
    return torch.randint(num_classes, (batch_size,), device=device, generator=generator)


def synthetic_label_stream(
    seed: int,
    epoch: int,
    device: torch.device | str,
    batch_sizes: Iterable[int],
    num_classes: int = NUM_CLASSES,
) -> list[torch.Tensor]:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed + 20000 + epoch)
    return [synthetic_labels(seed, epoch, device, size, num_classes, generator) for size in batch_sizes]


def _selected(model: nn.Module, layer_names: Iterable[str] | None) -> dict[str, nn.Linear]:
    candidates = affine_modules(model)
    selected = set(candidates) if layer_names is None else set(layer_names)
    missing = selected - set(candidates)
    if missing:
        raise ValueError(f"unknown/non-Linear geometry layers: {sorted(missing)}")
    if not selected:
        raise ValueError("no affine layers selected")
    return {name: candidates[name] for name in candidates if name in selected}


@torch.no_grad()
def build_a_operator(
    model: nn.Module,
    cache: list[torch.Tensor],
    power: float = A_POWER,
    damping: float = DAMPING,
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
    try:
        for logical_x in cache:
            for start in range(0, len(logical_x), SYNTHETIC_PHYSICAL_BATCH_SIZE):
                model(logical_x[start:start + SYNTHETIC_PHYSICAL_BATCH_SIZE])
                forward_calls += 1
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
        "builder_logical_batches": len(cache),
        "builder_vjp_calls": 0,
        "builder_reverse_vectors": 0,
        "builder_samples": sum(len(x) for x in cache),
        "preconditioned_layers": sorted(modules),
        "operator_state_bytes": operator.operator_state_bytes,
        **operator.moments,
        **operator.diagnostics,
    }
    return operator, stats


def build_full_operator(
    model: nn.Module,
    cache: list[torch.Tensor],
    seed: int,
    epoch: int,
    damping: float = DAMPING,
    layer_names: Iterable[str] | None = None,
) -> tuple[FullKFACOperator, dict]:
    modules = _selected(model, layer_names)
    factors = {}
    for name, module in modules.items():
        width = module.in_features + (1 if module.bias is not None else 0)
        factors[name] = {
            "A": torch.zeros(width, width, device=next(module.parameters()).device),
            "G": torch.zeros(module.out_features, module.out_features, device=next(module.parameters()).device),
            "output_dimension": module.out_features,
        }
    counts = {name: 0 for name in modules}
    activations, backprops = {}, {}
    handles = []

    def capture(name):
        def hook(module, args, output):
            activations[name] = args[0].detach()
            output.register_hook(lambda grad: backprops.__setitem__(name, grad.detach()))
        return hook

    handles = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
    forward_calls = 0
    vjp_calls = 0
    reverse_vectors = 0
    num_classes = getattr(model, "num_classes", None)
    if num_classes is None and hasattr(model, "head"):
        num_classes = model.head.out_features
    label_generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed + 20000 + epoch)
    try:
        for logical_x in cache:
            labels = synthetic_labels(seed, epoch, logical_x.device, len(logical_x), num_classes or NUM_CLASSES, label_generator)
            for start in range(0, len(logical_x), SYNTHETIC_PHYSICAL_BATCH_SIZE):
                x = logical_x[start:start + SYNTHETIC_PHYSICAL_BATCH_SIZE]
                y = labels[start:start + SYNTHETIC_PHYSICAL_BATCH_SIZE]
                model.zero_grad(set_to_none=True)
                logits = model(x)
                torch.nn.functional.cross_entropy(logits, y, reduction="sum").backward()
                for name, module in modules.items():
                    a = flatten_linear_input(activations[name], module)
                    b = backprops[name].reshape(-1, module.out_features)
                    factors[name]["A"].add_(a.T @ a)
                    factors[name]["G"].add_(b.T @ b)
                    counts[name] += a.shape[0]
                activations.clear()
                backprops.clear()
                forward_calls += 1
                vjp_calls += 1
                reverse_vectors += len(x)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    if any(counts[name] == 0 for name in modules):
        raise RuntimeError(f"calibration did not execute {sorted(name for name, n in counts.items() if n == 0)}")
    for name in modules:
        factors[name]["A"].div_(counts[name])
        factors[name]["G"].div_(counts[name])
    operator = FullKFACOperator(factors, damping)
    stats = {
        "builder_forward_calls": forward_calls,
        "builder_logical_batches": len(cache),
        "builder_vjp_calls": vjp_calls,
        "builder_reverse_vectors": reverse_vectors,
        "builder_samples": sum(len(x) for x in cache),
        "preconditioned_layers": sorted(modules),
        "operator_state_bytes": operator.operator_state_bytes,
    }
    return operator, stats


def build_from_cache(
    model: nn.Module,
    method: str,
    cache: list[torch.Tensor],
    seed: int = 0,
    epoch: int = 1,
    damping: float = DAMPING,
    power: float = A_POWER,
    layer_names: Iterable[str] | None = None,
):
    if method == "dp_sgd":
        return None, {
            "builder_forward_calls": 0,
            "builder_logical_batches": 0,
            "builder_vjp_calls": 0,
            "builder_reverse_vectors": 0,
            "builder_samples": 0,
            "preconditioned_layers": [],
            "operator_state_bytes": 0,
        }
    if method == "dp_kfc_a_bk":
        return build_a_operator(model, cache, power, damping, layer_names)
    if method == "dp_kfc":
        return build_full_operator(model, cache, seed, epoch, damping, layer_names)
    raise ValueError(f"method {method!r} has no preconditioner")


AOperator = AOnlyOperator
FullOperator = FullKFACOperator

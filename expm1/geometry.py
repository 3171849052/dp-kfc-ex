"""KFAC factor construction and diagnostics for the ExpM1 mechanisms.

Only explicitly tagged public, synthetic, or fixed-oracle batches are accepted.
The private training loop never calls this module with its current batch.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


DAMPING = 1e-3
ALLOWED_PROVENANCE = frozenset({"public", "pink", "oracle"})


@dataclass(frozen=True)
class FactorBatch:
    x: torch.Tensor
    y: torch.Tensor
    provenance: str

    def __post_init__(self) -> None:
        assert self.provenance in ALLOWED_PROVENANCE
        assert len(self.x) == len(self.y)


def affine_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
        and any(parameter.requires_grad for parameter in module.parameters(recurse=False))
    }
    assert modules
    for module in modules.values():
        if isinstance(module, nn.Conv2d):
            assert module.groups == 1 and module.padding_mode == "zeros"
    return modules


def augmented_width(module: nn.Module) -> int:
    if isinstance(module, nn.Conv2d):
        width = module.weight[0].numel()
    else:
        width = module.in_features
    return width + int(module.bias is not None)


def output_width(module: nn.Module) -> int:
    return module.out_channels if isinstance(module, nn.Conv2d) else module.out_features


def affine_factors(
    activation: torch.Tensor,
    backprop: torch.Tensor | None,
    module: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return row-layout KFAC samples, with bias as the last A coordinate."""
    if isinstance(module, nn.Conv2d):
        a = F.unfold(
            activation,
            module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        ).transpose(1, 2)
        b = None if backprop is None else backprop.flatten(2).transpose(1, 2)
    else:
        a = activation.reshape(len(activation), -1, activation.shape[-1])
        b = None if backprop is None else backprop.reshape(
            len(backprop), -1, backprop.shape[-1]
        )
    if module.bias is not None:
        a = torch.cat((a, torch.ones_like(a[..., :1])), dim=-1)
    return a, b


def _sym_eig(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix64 = ((matrix + matrix.T) * 0.5).double()
    values, vectors = torch.linalg.eigh(matrix64)
    return values.clamp_min(0), vectors


def matrix_power(
    covariance: torch.Tensor,
    power: float,
    damping: float = DAMPING,
) -> torch.Tensor:
    if power == 0:
        return torch.eye(len(covariance), device=covariance.device, dtype=covariance.dtype)
    values, vectors = _sym_eig(covariance)
    powered = (values + damping).pow(power)
    return ((vectors * powered) @ vectors.T).to(covariance.dtype)


def spectrum(covariance: torch.Tensor, damping: float = DAMPING) -> dict[str, float]:
    values, _ = _sym_eig(covariance)
    damped = values + damping
    probabilities = damped / damped.sum()
    entropy = -(probabilities * probabilities.log()).sum()
    effective_rank = entropy.exp()
    logs = damped.log()
    return {
        "eigenvalue_min": float(values.min()),
        "eigenvalue_max": float(values.max()),
        "damped_condition": float(damped.max() / damped.min()),
        "log_eigenvalue_spread": float(logs.std(unbiased=False)),
        "effective_rank": float(effective_rank),
    }


def factor_alignment(reference: torch.Tensor, estimate: torch.Tensor) -> tuple[float, float]:
    a, b = reference.double(), estimate.double()
    denominator = a.norm() * b.norm()
    cosine = ((a * b).sum() / denominator).clamp(-1, 1)
    relative = (a - b).norm() / a.norm()
    return float(cosine), float(relative)


def build_factors(
    model: nn.Module,
    batches: Iterable[FactorBatch],
    *,
    need_g: bool,
    physical_batch_size: int,
) -> tuple[dict[str, dict[str, torch.Tensor | int]], dict[str, float | int]]:
    """Build raw A/G factors. A-only construction performs no backward pass."""
    assert physical_batch_size > 0
    modules = affine_modules(model)
    device = next(model.parameters()).device
    factors: dict[str, dict[str, torch.Tensor | int]] = {}
    a_counts = {name: 0 for name in modules}
    g_counts = {name: 0 for name in modules}
    for name, module in modules.items():
        width, out = augmented_width(module), output_width(module)
        factors[name] = {
            "A": torch.zeros(width, width, device=device),
            "output_dimension": out,
        }
        if need_g:
            factors[name]["G"] = torch.zeros(out, out, device=device)

    activations: dict[str, torch.Tensor] = {}
    backprops: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor):
            activations[name] = args[0].detach()
            if need_g:
                output.register_hook(
                    lambda gradient, layer=name: backprops.__setitem__(layer, gradient.detach())
                )

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(capture(name)))

    was_training = model.training
    model.eval()
    forward_calls = backward_calls = samples = logical_batches = 0
    started = time.perf_counter()
    try:
        for logical in batches:
            assert isinstance(logical, FactorBatch)
            assert logical.provenance in ALLOWED_PROVENANCE
            assert logical.x.device == device and logical.y.device == device
            assert len(logical.x) == 256
            logical_batches += 1
            samples += len(logical.x)
            for start in range(0, len(logical.x), physical_batch_size):
                x = logical.x[start : start + physical_batch_size]
                y = logical.y[start : start + physical_batch_size]
                activations.clear()
                backprops.clear()
                model.zero_grad(set_to_none=True)
                if need_g:
                    logits = model(x)
                    F.cross_entropy(logits, y, reduction="sum").backward()
                    backward_calls += 1
                else:
                    with torch.no_grad():
                        model(x)
                assert set(activations) == set(modules)
                if need_g:
                    assert set(backprops) == set(modules)
                with torch.no_grad():
                    for name, module in modules.items():
                        a, b = affine_factors(
                            activations[name], backprops.get(name), module
                        )
                        flat_a = a.reshape(-1, a.shape[-1])
                        factors[name]["A"].add_(flat_a.T @ flat_a)
                        a_counts[name] += len(flat_a)
                        if need_g:
                            assert b is not None
                            flat_b = b.reshape(-1, b.shape[-1])
                            factors[name]["G"].add_(flat_b.T @ flat_b)
                            g_counts[name] += len(flat_b)
                forward_calls += 1
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        activations.clear()
        backprops.clear()

    assert logical_batches == 10 and samples == 2560
    assert all(count > 0 for count in a_counts.values())
    if need_g:
        assert all(count > 0 for count in g_counts.values())
    for name in modules:
        factors[name]["A"].div_(a_counts[name])
        if need_g:
            factors[name]["G"].div_(g_counts[name])
    seconds = time.perf_counter() - started
    state_bytes = sum(
        value.numel() * value.element_size()
        for factor in factors.values()
        for value in factor.values()
        if isinstance(value, torch.Tensor)
    )
    return factors, {
        "geometry_build_seconds": seconds,
        "builder_logical_batches": logical_batches,
        "builder_forward_calls": forward_calls,
        "builder_backward_calls": backward_calls,
        "builder_samples": samples,
        "factor_state_bytes": state_bytes,
    }


def geometry_rows(
    factors: dict[str, dict[str, torch.Tensor | int]],
    oracle: dict[str, dict[str, torch.Tensor | int]] | None,
    *,
    task: str,
    method: str,
    source: str,
    beta: float | None,
    seed: int,
    epoch: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, factor in factors.items():
        row: dict[str, object] = {
            "task": task,
            "method": method,
            "source": source,
            "beta": beta,
            "seed": seed,
            "epoch": epoch,
            "layer": name,
            "research_only": True,
        }
        for key in ("A", "G"):
            if key not in factor:
                continue
            stats = spectrum(factor[key])
            row.update({f"{key}_{label}": value for label, value in stats.items()})
            if oracle is not None:
                assert key in oracle[name]
                cosine, relative = factor_alignment(oracle[name][key], factor[key])
                row[f"cos{key}"] = cosine
                row[f"relative_error_{key}"] = relative
        rows.append(row)
    return rows


def combined_condition(factor: dict[str, torch.Tensor | int], a_only: bool) -> float:
    a = spectrum(factor["A"])["damped_condition"]
    if a_only:
        return a
    return a * spectrum(factor["G"])["damped_condition"]

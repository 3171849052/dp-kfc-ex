"""Per-example BK clipping and one-step DP accounting primitives."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .handlers import (
    layernorm_per_example_gradient,
    per_example_linear_gradient,
    split_linear_gradient,
    tensor_bytes,
)

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _group(name: str) -> str:
    if name.endswith(("q_proj", "k_proj", "v_proj")):
        return "attention_qkv"
    if name.endswith("out_proj"):
        return "attention_out"
    if name.endswith("ffn"):
        return "ffn"
    if name in ("patch_embed", "head"):
        return "patch_head"
    return "identity"


def _retained_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class Clipper:
    """Exp21-style API with exact chunked logical-batch semantics.

    The implementation deliberately keeps the intermediate per-example
    matrices local to one physical batch.  Only the clipped aggregate is
    retained between physical chunks.
    """

    def __init__(
        self,
        model: nn.Module,
        operator=None,
        method: str = "bk",
        max_grad_norm: float = 1.0,
    ):
        if method not in ("bk", "ghost", "bk_gd", "exact", "dp_sgd", "dp_kfc_a_bk", "dp_kfc"):
            raise ValueError(method)
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.model = model
        self.operator = operator
        self.method = method
        self.max_grad_norm = max_grad_norm
        self.linear_modules = dict((n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear))
        self.norm_modules = dict((n, m) for n, m in model.named_modules() if isinstance(m, nn.LayerNorm))
        self.pos_embed = getattr(model, "pos_embed", None)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.parameter_ids = {id(p) for p in self.parameters}
        handled = {id(p) for m in self.linear_modules.values() for p in m.parameters(recurse=False)}
        handled.update(id(p) for m in self.norm_modules.values() for p in m.parameters(recurse=False))
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            handled.add(id(self.pos_embed))
        if handled != self.parameter_ids:
            missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in handled]
            raise NotImplementedError(f"unsupported trainable parameters: {missing}")
        selected = set() if operator is None else set(operator.data)
        if not selected <= set(self.linear_modules):
            raise ValueError(f"operator layers not in model: {sorted(selected - set(self.linear_modules))}")
        self.preconditioned_layers = sorted(selected)
        self.identity_geometry_layers = sorted(set(self.linear_modules) - selected)
        self.identity_geometry_layers += sorted(self.norm_modules)
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            self.identity_geometry_layers.append("pos_embed")
        self.identity_geometry_layers.sort()
        self.pending = []
        self.records = []
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.noise_events = 0
        self.accountant_steps = 0
        self._last_stats = {}

    def _capture_hooks(self, activations, backprops, position_backprops):
        handles = []
        modules = {**self.linear_modules, **self.norm_modules}
        for name, module in modules.items():
            def forward_hook(module, args, output, name=name):
                activations[name] = args[0].detach()
                output.register_hook(lambda grad, name=name: backprops.__setitem__(name, grad.detach()))
                if name == "patch_embed" and self.pos_embed is not None:
                    output.register_hook(lambda grad: position_backprops.append(grad.detach()))
            handles.append(module.register_forward_hook(forward_hook))
        return handles

    def _one_batch(self, x, y, loss_fn: LossFunction | None = None):
        if loss_fn is None:
            loss_fn = lambda output, target: F.cross_entropy(output, target, reduction="none")
        activations, backprops, position_backprops = {}, {}, []
        handles = self._capture_hooks(activations, backprops, position_backprops)
        self.model.zero_grad(set_to_none=True)
        try:
            logits = self.model(x)
            losses = loss_fn(logits, y)
            if losses.ndim != 1 or len(losses) != len(x):
                raise ValueError("loss_fn must return one scalar loss per example")
            losses.sum().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.backward_calls += 1
        self.model.zero_grad(set_to_none=True)
        if set(activations) != set(backprops):
            raise RuntimeError(f"missing layer backprops: {sorted(set(activations) - set(backprops))}")
        if self.pos_embed is not None and self.pos_embed.requires_grad and not position_backprops:
            raise RuntimeError("pos_embed did not receive per-example backprop")

        cache_bytes = _retained_bytes(
            list(activations.values()) + list(backprops.values()) + position_backprops
        )
        layer_sq = {}
        group_sq = defaultdict(lambda: torch.zeros(len(x), device=x.device, dtype=x.dtype))
        temporary_bytes = 0
        for name, module in self.linear_modules.items():
            g = per_example_linear_gradient(activations[name], backprops[name], module)
            if self.operator is not None and name in self.operator.data:
                g = self.operator.transform_gradient(name, g)
            temporary_bytes = max(temporary_bytes, tensor_bytes(g))
            layer_sq[name] = g.square().sum(dim=(1, 2))
            group_sq[_group(name)].add_(layer_sq[name])
            del g
        for name, module in self.norm_modules.items():
            grads = layernorm_per_example_gradient(activations[name], backprops[name], module)
            sq = torch.zeros(len(x), device=x.device, dtype=x.dtype)
            for parameter, value in grads.items():
                sq = sq + value.reshape(len(x), -1).square().sum(dim=1)
            layer_sq[name] = sq
            group_sq["identity"].add_(sq)
            temporary_bytes = max(temporary_bytes, sum(tensor_bytes(v) for v in grads.values()))
            del grads
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            pos = position_backprops[0]
            if pos.shape[1:] != self.pos_embed.shape[1:]:
                raise RuntimeError("unexpected pos_embed gradient shape")
            layer_sq["pos_embed"] = pos.reshape(len(x), -1).square().sum(dim=1)
            group_sq["identity"].add_(layer_sq["pos_embed"])
        norms = sum(layer_sq.values()).clamp_min(0).sqrt()
        factors = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1).detach()
        aggregate = {p: torch.zeros_like(p) for p in self.parameters}
        for name, module in self.linear_modules.items():
            g = per_example_linear_gradient(activations[name], backprops[name], module)
            if self.operator is not None and name in self.operator.data:
                g = self.operator.transform_gradient(name, g)
            clipped = torch.einsum("b,bod->od", factors, g)
            for parameter, value in split_linear_gradient(clipped, module).items():
                aggregate[parameter].add_(value)
            del g, clipped
        for name, module in self.norm_modules.items():
            grads = layernorm_per_example_gradient(activations[name], backprops[name], module)
            for parameter, value in grads.items():
                aggregate[parameter].add_(torch.einsum("b,b...->...", factors, value))
            del grads
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            aggregate[self.pos_embed].add_(torch.einsum("b,b...->...", factors, position_backprops[0]))
        activations.clear()
        backprops.clear()
        position_backprops.clear()
        cache_empty = not activations and not backprops and not position_backprops
        return {
            "loss": losses.detach().sum(),
            "norms": norms.detach(),
            "factors": factors,
            "layer_sq": {k: v.detach() for k, v in layer_sq.items()},
            "group_sq": {k: v.detach() for k, v in group_sq.items()},
            "aggregate": aggregate,
            "bk_cache_bytes": cache_bytes,
            "temporary_per_sample_grad_bytes": temporary_bytes,
            "fallback_temporary_grad_bytes": 0,
            "cache_empty_after_step": cache_empty,
        }

    def _finish(self, parts, total_size):
        aggregate = {p: torch.zeros_like(p) for p in self.parameters}
        for part in parts:
            for p, value in part["aggregate"].items():
                aggregate[p].add_(value)
        self.model.zero_grad(set_to_none=True)
        for p, value in aggregate.items():
            p.grad = value
        norms = torch.cat([part["norms"] for part in parts])
        factors = torch.cat([part["factors"] for part in parts])
        layer_sq = {}
        group_sq = {}
        for part in parts:
            for name, value in part["layer_sq"].items():
                layer_sq.setdefault(name, []).append(value)
            for name, value in part["group_sq"].items():
                group_sq.setdefault(name, []).append(value)
        layer_sq = {name: torch.cat(values) for name, values in layer_sq.items()}
        group_sq = {name: torch.cat(values) for name, values in group_sq.items()}
        stats = {
            "backward_calls": len(parts),
            "bk_cache_bytes": max((part["bk_cache_bytes"] for part in parts), default=0),
            "temporary_per_sample_grad_bytes": max(
                (part["temporary_per_sample_grad_bytes"] for part in parts), default=0
            ),
            "fallback_temporary_grad_bytes": max(
                (part["fallback_temporary_grad_bytes"] for part in parts), default=0
            ),
            "cache_empty_after_step": all(part["cache_empty_after_step"] for part in parts),
            "preconditioned_layers": self.preconditioned_layers,
            "identity_geometry_layers": self.identity_geometry_layers,
            "layer_strategies": {name: "bk_layer_local" for name in self.linear_modules},
            "layer_group_sq": group_sq,
            "logical_batch_size": total_size,
            "physical_batch_size": max(len(part["norms"]) for part in parts),
            "accumulation_steps": len(parts),
            "clip_fraction": (norms > self.max_grad_norm).float().mean().item(),
            "mean_clip_factor": factors.mean().item(),
            "transformed_norm_p50": torch.quantile(norms, 0.50).item(),
            "transformed_norm_p90": torch.quantile(norms, 0.90).item(),
            "transformed_norm_p99": torch.quantile(norms, 0.99).item(),
            "transformed_norm_max": norms.max().item(),
            "group_norm_contribution": {
                name: value.sum().sqrt().item() for name, value in group_sq.items()
            },
        }
        self._last_stats = stats
        return norms, factors, layer_sq, stats

    def aggregate(
        self,
        x,
        y,
        profiler=None,
        loss_fn: LossFunction | None = None,
        physical_batch_size: int | None = None,
    ):
        if isinstance(profiler, int) and physical_batch_size is None:
            physical_batch_size, profiler = profiler, None
        if physical_batch_size is not None:
            return self.aggregate_logical(x, y, physical_batch_size, loss_fn)
        started = time.perf_counter()
        part = self._one_batch(x, y, loss_fn)
        norms, factors, layer_sq, stats = self._finish([part], len(x))
        stats["private_batch_seconds"] = time.perf_counter() - started
        return part["loss"], norms, factors, layer_sq, stats

    def aggregate_logical(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physical_batch_size: int,
        loss_fn: LossFunction | None = None,
    ):
        if len(x) % physical_batch_size:
            raise ValueError("logical batch must be divisible by physical batch")
        started = time.perf_counter()
        parts = [
            self._one_batch(x[start:start + physical_batch_size], y[start:start + physical_batch_size], loss_fn)
            for start in range(0, len(x), physical_batch_size)
        ]
        norms, factors, layer_sq, stats = self._finish(parts, len(x))
        stats["private_batch_seconds"] = time.perf_counter() - started
        stats["physical_batch_size"] = physical_batch_size
        stats["accumulation_steps"] = len(parts)
        loss = sum(part["loss"] for part in parts)
        return loss, norms, factors, layer_sq, stats

    def step(self, optimizer, sigma: float, batch_size: int, generator, bound: float | None = None):
        bound = self.max_grad_norm if bound is None else bound
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for p in self.parameters:
            noise = torch.randn(p.numel(), device=p.device, dtype=p.dtype, generator=generator).reshape_as(p)
            p.grad.add_(noise, alpha=sigma * bound).div_(batch_size)
        optimizer.step()
        self.optimizer_steps += 1
        self.noise_events += 1
        return self.optimizer_steps

    def remove(self):
        self.pending.clear()
        self.records.clear()


def noise_and_step(model, optimizer, sigma, batch_size, generator, bound=1.0):
    parameters = [p for p in model.parameters() if p.requires_grad]
    for p in parameters:
        noise = torch.randn(p.numel(), device=p.device, dtype=p.dtype, generator=generator).reshape_as(p)
        p.grad.add_(noise, alpha=sigma * bound).div_(batch_size)
    optimizer.step()


def make_optimizer(model: nn.Module):
    from . import config

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.ADAM_EPS,
        weight_decay=config.WEIGHT_DECAY,
    )


def aggregate_logical_batch(model, operator, x, y, physical_batch_size, max_grad_norm=1.0, loss_fn=None):
    return Clipper(model, operator, max_grad_norm=max_grad_norm).aggregate_logical(
        x, y, physical_batch_size, loss_fn
    )


BKClipper = Clipper
BookKeeping = Clipper

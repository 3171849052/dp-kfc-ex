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
    tensor_bytes,
)

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _group(name: str) -> str:
    if name.endswith(("q_proj", "k_proj", "v_proj")):
        return "attention_qkv"
    if name.endswith(("out_proj", "attn.proj")):
        return "attention_out"
    if name.endswith(("ffn", "mlp.fc1", "mlp.fc2")):
        return "ffn"
    if name in ("patch_embed.proj", "head"):
        return "patch_head"
    return "identity"


def _retained_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class Clipper:
    """Exp21-style layer-local BookKeeping with logical-batch chunking."""

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
        self.cls_token = getattr(model, "cls_token", None)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.parameter_ids = {id(p) for p in self.parameters}
        handled = {id(p) for m in self.linear_modules.values() for p in m.parameters(recurse=False)}
        handled.update(id(p) for m in self.norm_modules.values() for p in m.parameters(recurse=False))
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            handled.add(id(self.pos_embed))
        if self.cls_token is not None and self.cls_token.requires_grad:
            handled.add(id(self.cls_token))
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
        if self.cls_token is not None and self.cls_token.requires_grad:
            self.identity_geometry_layers.append("cls_token")
        self.identity_geometry_layers.sort()
        self.pending = []
        self.records = []
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.noise_events = 0
        self._last_stats = {}

    def _empty_aggregate(self):
        return {parameter: torch.zeros_like(parameter) for parameter in self.parameters}

    def _output_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        layer_names = list(self.linear_modules) + list(self.norm_modules)
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            layer_names.append("pos_embed")
        if self.cls_token is not None and self.cls_token.requires_grad:
            layer_names.append("cls_token")
        layers = {name: torch.empty(batch_size, device=device, dtype=dtype) for name in layer_names}
        group_names = {_group(layer) for layer in self.linear_modules}
        if len(layer_names) > len(self.linear_modules):
            group_names.add("identity")
        groups = {
            name: torch.empty(batch_size, device=device, dtype=dtype)
            for name in group_names
        }
        return layers, groups

    def _linear_factors(self, name, activation, backprop, module):
        """Return transformed activation/backprop factors, never sample weights."""
        z = activation.reshape(len(activation), -1, activation.shape[-1])
        if module.bias is not None:
            z = torch.cat((z, torch.ones_like(z[..., :1])), dim=-1)
        b = backprop.reshape(len(backprop), -1, backprop.shape[-1])
        if self.operator is not None and name in self.operator.data:
            z = self.operator.transform_activation(name, z)
            if hasattr(self.operator, "transform_backprop"):
                b = self.operator.transform_backprop(name, b)
        return z, b

    @staticmethod
    def _linear_norm_squared(z, b, tile=8):
        """Ghost/BK norm for sum_t b_t^T z_t without forming sample weights."""
        total = z.new_zeros(len(z))
        b_transposed, z_transposed = b.transpose(1, 2), z.transpose(1, 2)
        rows = min(tile, max(1, b.shape[1] - 1))
        temporary_bytes = 0
        for start in range(0, b.shape[1], rows):
            b_gram = b[:, start:start + rows] @ b_transposed
            z_gram = z[:, start:start + rows] @ z_transposed
            temporary_bytes = max(
                temporary_bytes,
                tensor_bytes(b_gram) + tensor_bytes(z_gram),
            )
            b_gram.mul_(z_gram)
            total.add_(b_gram.sum((1, 2)))
            del b_gram, z_gram
        return total.clamp_min(0), temporary_bytes

    @staticmethod
    def _linear_aggregate(z, b, factors):
        """Reconstruct a clipped Linear aggregate without per-example weights."""
        weighted_b = b * factors[:, None, None]
        clipped = torch.einsum("bto,bti->oi", weighted_b, z)
        temporary_bytes = tensor_bytes(weighted_b) + tensor_bytes(clipped)
        return clipped, temporary_bytes

    def _capture_hooks(self, activations, backprops, position_backprops):
        handles = []
        modules = {**self.linear_modules, **self.norm_modules}
        for name, module in modules.items():
            def forward_hook(module, args, output, name=name):
                activations[name] = args[0].detach()
                output.register_hook(lambda grad, name=name: backprops.__setitem__(name, grad.detach()))
            handles.append(module.register_forward_hook(forward_hook))
        if self.pos_embed is not None:
            def token_hook(module, args):
                args[0].register_hook(lambda grad: position_backprops.append(grad.detach()))
            handles.append(self.model.pos_drop.register_forward_pre_hook(token_hook))
        return handles

    def _one_batch(self, x, y, aggregate, loss_fn: LossFunction | None = None):
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
            raw_activation, raw_backprop = activations[name], backprops[name]
            z, b = self._linear_factors(name, raw_activation, raw_backprop, module)
            cache_bytes = max(
                cache_bytes,
                _retained_bytes(
                    list(activations.values()) + list(backprops.values())
                    + [z, b] + position_backprops
                ),
            )
            activations[name], backprops[name] = z, b
            del raw_activation, raw_backprop
            layer_sq[name], layer_temporary_bytes = self._linear_norm_squared(z, b)
            temporary_bytes = max(temporary_bytes, layer_temporary_bytes)
            group_sq[_group(name)].add_(layer_sq[name])
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
        if self.cls_token is not None and self.cls_token.requires_grad:
            cls = position_backprops[0][:, :1]
            layer_sq["cls_token"] = cls.reshape(len(x), -1).square().sum(dim=1)
            group_sq["identity"].add_(layer_sq["cls_token"])
        norms = sum(layer_sq.values()).clamp_min(0).sqrt()
        factors = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1).detach()
        for name, module in self.linear_modules.items():
            z, b = activations[name], backprops[name]
            clipped, aggregate_temporary_bytes = self._linear_aggregate(z, b, factors)
            temporary_bytes = max(temporary_bytes, aggregate_temporary_bytes)
            if module.bias is None:
                aggregate[module.weight].add_(clipped)
            else:
                aggregate[module.weight].add_(clipped[:, :-1])
                aggregate[module.bias].add_(clipped[:, -1])
            del z, b, clipped
        for name, module in self.norm_modules.items():
            grads = layernorm_per_example_gradient(activations[name], backprops[name], module)
            norm_workspace_bytes = sum(tensor_bytes(value) for value in grads.values())
            temporary_bytes = max(temporary_bytes, norm_workspace_bytes)
            for parameter, value in grads.items():
                reconstructed = torch.einsum("b,b...->...", factors, value)
                temporary_bytes = max(
                    temporary_bytes,
                    norm_workspace_bytes + tensor_bytes(reconstructed),
                )
                aggregate[parameter].add_(reconstructed)
                del reconstructed
            del grads
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            reconstructed = torch.einsum("b,b...->...", factors, position_backprops[0])
            temporary_bytes = max(temporary_bytes, tensor_bytes(reconstructed))
            aggregate[self.pos_embed].add_(reconstructed)
            del reconstructed
        if self.cls_token is not None and self.cls_token.requires_grad:
            reconstructed = torch.einsum("b,btd->td", factors, position_backprops[0][:, :1])
            aggregate[self.cls_token].add_(reconstructed)
            temporary_bytes = max(temporary_bytes, tensor_bytes(reconstructed))
            del reconstructed
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
            "bk_cache_bytes": cache_bytes,
            "temporary_per_sample_grad_bytes": temporary_bytes,
            "fallback_temporary_grad_bytes": 0,
            "cache_empty_after_step": cache_empty,
        }

    def _finish(
        self,
        aggregate,
        norms,
        factors,
        layer_sq,
        group_sq,
        total_size,
        physical_batch_size,
        chunk_count,
        max_cache,
        max_temporary,
        max_fallback_temporary,
        cache_empty,
    ):
        self.model.zero_grad(set_to_none=True)
        for p, value in aggregate.items():
            p.grad = value
        stats = {
            "backward_calls": chunk_count,
            "bk_cache_bytes": max_cache,
            "temporary_per_sample_grad_bytes": max_temporary,
            "fallback_temporary_grad_bytes": max_fallback_temporary,
            "cache_empty_after_step": cache_empty,
            "preconditioned_layers": self.preconditioned_layers,
            "identity_geometry_layers": self.identity_geometry_layers,
            "layer_strategies": {
                **{name: "bk_ghost" for name in self.linear_modules},
                **{name: "identity_analytic" for name in self.norm_modules},
                **({"cls_token": "identity_direct"}
                   if self.cls_token is not None and self.cls_token.requires_grad else {}),
                **({"pos_embed": "identity_direct"}
                   if self.pos_embed is not None and self.pos_embed.requires_grad else {}),
            },
            "layer_group_sq": group_sq,
            "logical_batch_size": total_size,
            "physical_batch_size": physical_batch_size,
            "accumulation_steps": chunk_count,
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
        result = self._aggregate_chunks(x, y, len(x), loss_fn)
        norms, factors, layer_sq, stats, loss = result
        stats["private_batch_seconds"] = time.perf_counter() - started
        return loss, norms, factors, layer_sq, stats

    def _aggregate_chunks(self, x, y, physical_batch_size, loss_fn):
        aggregate = self._empty_aggregate()
        norms = torch.empty(len(x), device=x.device, dtype=x.dtype)
        factors = torch.empty_like(norms)
        layer_sq, group_sq = self._output_buffers(len(x), x.device, x.dtype)
        loss = x.new_zeros(())
        max_cache = max_temporary = max_fallback_temporary = 0
        cache_empty = True
        chunk_count = 0
        for start in range(0, len(x), physical_batch_size):
            stop = start + physical_batch_size
            part = self._one_batch(
                x[start:stop], y[start:stop], aggregate, loss_fn
            )
            norms[start:stop].copy_(part["norms"])
            factors[start:stop].copy_(part["factors"])
            for name, value in part["layer_sq"].items():
                layer_sq[name][start:stop].copy_(value)
            for name, value in part["group_sq"].items():
                group_sq[name][start:stop].copy_(value)
            loss.add_(part["loss"])
            max_cache = max(max_cache, part["bk_cache_bytes"])
            max_temporary = max(max_temporary, part["temporary_per_sample_grad_bytes"])
            max_fallback_temporary = max(
                max_fallback_temporary, part["fallback_temporary_grad_bytes"]
            )
            cache_empty = cache_empty and part["cache_empty_after_step"]
            chunk_count += 1
            del part
        finished = self._finish(
            aggregate, norms, factors, layer_sq, group_sq, len(x),
            physical_batch_size, chunk_count, max_cache, max_temporary,
            max_fallback_temporary, cache_empty,
        )
        return (*finished[:3], finished[3], loss)

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
        norms, factors, layer_sq, stats, loss = self._aggregate_chunks(
            x, y, physical_batch_size, loss_fn
        )
        stats["private_batch_seconds"] = time.perf_counter() - started
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

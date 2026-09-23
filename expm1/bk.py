"""Book-Keeping clipping with strictly separate metric and raw-signal branches."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from expm1.geometry import affine_factors, affine_modules
from expm1.mechanism import Shape, layer_group
from exp22.handlers import layernorm_per_example_gradient


NUMERICAL_EPS = 1e-6


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def retained_bytes(tensors: list[torch.Tensor]) -> int:
    storages: dict[tuple[torch.device, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def ghost_squared(a: torch.Tensor, b: torch.Tensor, tile: int = 32) -> tuple[torch.Tensor, int]:
    """Squared norm of sum_t b_t a_t^T without materializing sample gradients."""
    assert a.ndim == b.ndim == 3 and a.shape[:2] == b.shape[:2]
    total = a.new_zeros(len(a))
    at, bt = a.transpose(1, 2), b.transpose(1, 2)
    rows = min(tile, max(1, a.shape[1] - 1))
    temporary = 0
    for start in range(0, a.shape[1], rows):
        aa = a[:, start : start + rows] @ at
        bb = b[:, start : start + rows] @ bt
        temporary = max(temporary, tensor_bytes(aa) + tensor_bytes(bb))
        aa.mul_(bb)
        total.add_(aa.sum((1, 2)))
    return total.clamp_min(0), temporary


def layernorm_squared(
    activation: torch.Tensor,
    backprop: torch.Tensor,
    module: nn.LayerNorm,
) -> tuple[torch.Tensor, int]:
    dimensions = tuple(range(activation.ndim - len(module.normalized_shape), activation.ndim))
    mean = activation.mean(dimensions, keepdim=True)
    variance = activation.var(dimensions, unbiased=False, keepdim=True)
    normalized = (activation - mean) * torch.rsqrt(variance + module.eps)
    reduce_dimensions = tuple(range(1, activation.ndim - len(module.normalized_shape)))
    reduce = lambda value: value.sum(reduce_dimensions) if reduce_dimensions else value
    total = activation.new_zeros(len(activation))
    temporary = tensor_bytes(normalized)
    if module.weight is not None and module.weight.requires_grad:
        weight = reduce(backprop * normalized)
        total.add_(weight.reshape(len(weight), -1).square().sum(1))
        temporary += tensor_bytes(weight)
    if module.bias is not None and module.bias.requires_grad:
        bias = reduce(backprop)
        total.add_(bias.reshape(len(bias), -1).square().sum(1))
        temporary += tensor_bytes(bias)
    return total, temporary


@dataclass
class AggregateResult:
    loss_sum: torch.Tensor
    matched_norms: torch.Tensor
    raw_norms: torch.Tensor
    clip_factors: torch.Tensor
    raw_sum: dict[nn.Parameter, torch.Tensor]
    clipped_sum: dict[nn.Parameter, torch.Tensor]
    stats: dict[str, object]


class BKClipper:
    def __init__(self, model: nn.Module, shape: Shape, bound: float = 1.0):
        assert shape.model is model and bound > 0
        self.model = model
        self.shape = shape
        self.bound = bound
        self.parameters = shape.parameters
        self.affines = affine_modules(model)
        self.norms = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.LayerNorm)
            and any(parameter.requires_grad for parameter in module.parameters(recurse=False))
        }
        self.pos_embed = getattr(model, "pos_embed", None)
        self.cls_token = getattr(model, "cls_token", None)
        self.token_hook = getattr(model, "token_hook", None)
        handled = {
            id(parameter)
            for module in (*self.affines.values(), *self.norms.values())
            for parameter in module.parameters(recurse=False)
            if parameter.requires_grad
        }
        for parameter in (self.pos_embed, self.cls_token):
            if parameter is not None and parameter.requires_grad:
                handled.add(id(parameter))
        expected = {id(parameter) for parameter in self.parameters}
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in handled
        ]
        assert handled == expected, f"unsupported trainable parameters: {missing}"
        assert (self.pos_embed is None and self.cls_token is None) or self.token_hook is not None
        self.enabled = False
        self.activations: dict[str, torch.Tensor] = {}
        self.backprops: dict[str, torch.Tensor] = {}
        self.token_backprops: list[torch.Tensor] = []
        self.handles = []
        for name, module in {**self.affines, **self.norms}.items():
            self.handles.append(module.register_forward_hook(self._capture(name)))
        if self.token_hook is not None:
            self.handles.append(self.token_hook.register_forward_pre_hook(self._capture_tokens))

    def _capture(self, name: str):
        def hook(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor):
            if not self.enabled:
                return
            assert name not in self.activations
            self.activations[name] = args[0].detach()
            output.register_hook(
                lambda gradient, layer=name: self.backprops.__setitem__(
                    layer, gradient.detach()
                )
            )

        return hook

    def _capture_tokens(self, module: nn.Module, args: tuple[torch.Tensor, ...]):
        if self.enabled:
            assert not self.token_backprops
            args[0].register_hook(lambda gradient: self.token_backprops.append(gradient.detach()))

    def _empty(self) -> dict[nn.Parameter, torch.Tensor]:
        return {parameter: torch.zeros_like(parameter) for parameter in self.parameters}

    @staticmethod
    def _accumulate(
        target: dict[nn.Parameter, torch.Tensor], source: dict[nn.Parameter, torch.Tensor]
    ) -> None:
        for parameter in target:
            target[parameter].add_(source[parameter])

    def _per_example_squared(self, x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        int,
        int,
    ]:
        batch = len(x)
        raw_total = x.new_zeros(batch)
        metric_total = x.new_zeros(batch)
        layer_metric: dict[str, torch.Tensor] = {}
        group_metric: dict[str, torch.Tensor] = defaultdict(lambda: x.new_zeros(batch))
        tensors = list(self.activations.values()) + list(self.backprops.values()) + self.token_backprops
        cache_bytes = retained_bytes(tensors)
        temporary = 0
        for name, module in self.affines.items():
            raw_a, raw_b = affine_factors(
                self.activations[name], self.backprops[name], module
            )
            assert raw_b is not None
            if self.shape.method == "dp_sgd":
                metric_sq, metric_temp = ghost_squared(raw_a, raw_b)
                raw_sq, raw_temp = metric_sq, metric_temp
                cache_bytes = max(cache_bytes, retained_bytes(tensors))
            else:
                raw_sq, raw_temp = ghost_squared(raw_a, raw_b)
                metric_a, metric_b = self.shape.metric_factors(name, raw_a, raw_b)
                metric_sq, metric_temp = ghost_squared(metric_a, metric_b)
                cache_bytes = max(cache_bytes, retained_bytes(tensors + [metric_a, metric_b]))
            raw_total.add_(raw_sq)
            metric_total.add_(metric_sq)
            layer_metric[name] = metric_sq
            group_metric[layer_group(name)].add_(metric_sq)
            temporary = max(temporary, raw_temp, metric_temp)

        for name, module in self.norms.items():
            squared, workspace = layernorm_squared(
                self.activations[name], self.backprops[name], module
            )
            raw_total.add_(squared)
            metric_total.add_(squared)
            layer_metric[name] = squared
            group_metric["identity"].add_(squared)
            temporary = max(temporary, workspace)

        if self.pos_embed is not None and self.pos_embed.requires_grad:
            assert len(self.token_backprops) == 1
            token_gradient = self.token_backprops[0]
            assert token_gradient.shape[1:] == self.pos_embed.shape[1:]
            squared = token_gradient.reshape(batch, -1).square().sum(1)
            raw_total.add_(squared)
            metric_total.add_(squared)
            layer_metric["pos_embed"] = squared
            group_metric["identity"].add_(squared)
            temporary = max(temporary, tensor_bytes(token_gradient))
        if self.cls_token is not None and self.cls_token.requires_grad:
            assert len(self.token_backprops) == 1
            cls_gradient = self.token_backprops[0][:, :1]
            squared = cls_gradient.reshape(batch, -1).square().sum(1)
            raw_total.add_(squared)
            metric_total.add_(squared)
            layer_metric["cls_token"] = squared
            group_metric["identity"].add_(squared)
            temporary = max(temporary, tensor_bytes(cls_gradient))
        if self.shape.method != "dp_sgd":
            metric_total.mul_(self.shape.metric_scale)
        return raw_total, metric_total, layer_metric, dict(group_metric), cache_bytes, temporary

    @staticmethod
    def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        cpu = torch.random.get_rng_state()
        cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return cpu, cuda

    @staticmethod
    def _set_rng_state(device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
        torch.random.set_rng_state(state[0])
        if device.type == "cuda":
            assert state[1] is not None
            torch.cuda.set_rng_state(state[1], device)

    def _one(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, object]:
        assert len(x) == len(y)
        self.activations.clear()
        self.backprops.clear()
        self.token_backprops.clear()
        self.model.zero_grad(set_to_none=True)
        self.enabled = True
        logits = self.model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        assert losses.shape == (len(x),)
        losses.sum().backward()
        self.enabled = False
        assert set(self.activations) == set(self.affines) | set(self.norms)
        assert set(self.backprops) == set(self.activations)
        raw_sum = {parameter: parameter.grad.detach().clone() for parameter in self.parameters}
        raw_sq, metric_sq, layer_sq, group_sq, cache_bytes, temporary = self._per_example_squared(x)
        raw_norms = raw_sq.clamp_min(0).sqrt()
        matched_norms = metric_sq.clamp_min(0).sqrt()
        factors = (self.bound / (matched_norms + NUMERICAL_EPS)).clamp(max=1).detach()

        clipped_sum = self._empty()
        for name, module in self.affines.items():
            activation, backprop = affine_factors(
                self.activations[name], self.backprops[name], module
            )
            weighted_backprop = backprop * factors[:, None, None]
            matrix = torch.einsum("bto,bti->oi", weighted_backprop, activation)
            clipped_sum[module.weight].copy_(matrix[:, : module.weight.flatten(1).shape[1]].reshape_as(module.weight))
            if module.bias is not None:
                clipped_sum[module.bias].copy_(matrix[:, -1])

        for name, module in self.norms.items():
            per_example = layernorm_per_example_gradient(
                self.activations[name], self.backprops[name], module
            )
            for parameter, value in per_example.items():
                weighted = value * factors.reshape(len(factors), *([1] * (value.ndim - 1)))
                clipped_sum[parameter].add_(weighted.sum(0))

        if self.pos_embed is not None and self.pos_embed.requires_grad:
            clipped_sum[self.pos_embed].copy_(torch.einsum(
                "b,b...->...", factors, self.token_backprops[0]
            ).unsqueeze(0))
        if self.cls_token is not None and self.cls_token.requires_grad:
            clipped_sum[self.cls_token].copy_(torch.einsum(
                "b,b...->...", factors, self.token_backprops[0][:, :1]
            ).unsqueeze(0))
        self.model.zero_grad(set_to_none=True)
        self.activations.clear()
        self.backprops.clear()
        self.token_backprops.clear()
        return {
            "loss_sum": losses.detach().sum(),
            "raw_norms": raw_norms.detach(),
            "matched_norms": matched_norms.detach(),
            "factors": factors,
            "raw_sum": raw_sum,
            "clipped_sum": clipped_sum,
            "layer_sq": {name: value.detach() for name, value in layer_sq.items()},
            "group_sq": {name: value.detach() for name, value in group_sq.items()},
            "bk_cache_bytes": cache_bytes,
            "temporary_per_sample_bytes": temporary,
            "backward_calls": 1,
        }

    def aggregate_logical(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physical_batch_size: int,
    ) -> AggregateResult:
        assert physical_batch_size > 0
        matched = x.new_empty(len(x))
        raw = x.new_empty(len(x))
        clips = x.new_empty(len(x))
        loss = x.new_zeros(())
        raw_sum, clipped_sum = self._empty(), self._empty()
        max_cache = max_temporary = 0
        layer_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        group_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        chunks = 0
        backward_calls = 0
        if len(x) == 0:
            zero = self._empty()
            return AggregateResult(
                x.new_zeros(()), x.new_empty(0), x.new_empty(0), x.new_empty(0),
                zero, {parameter: value.clone() for parameter, value in zero.items()},
                {"bk_cache_bytes": 0, "temporary_per_sample_bytes": 0,
                 "backward_calls": 0, "physical_chunks": 0,
                 "layer_norm_contribution": {}, "group_norm_contribution": {}},
            )
        for start in range(0, len(x), physical_batch_size):
            stop = min(start + physical_batch_size, len(x))
            part = self._one(x[start:stop], y[start:stop])
            matched[start:stop].copy_(part["matched_norms"])
            raw[start:stop].copy_(part["raw_norms"])
            clips[start:stop].copy_(part["factors"])
            loss.add_(part["loss_sum"])
            self._accumulate(raw_sum, part["raw_sum"])
            self._accumulate(clipped_sum, part["clipped_sum"])
            for name, value in part["layer_sq"].items():
                layer_values[name].append(value)
            for name, value in part["group_sq"].items():
                group_values[name].append(value)
            max_cache = max(max_cache, int(part["bk_cache_bytes"]))
            max_temporary = max(max_temporary, int(part["temporary_per_sample_bytes"]))
            chunks += 1
            backward_calls += int(part["backward_calls"])
        stats: dict[str, object] = {
            "bk_cache_bytes": max_cache,
            "temporary_per_sample_bytes": max_temporary,
            "backward_calls": backward_calls,
            "physical_chunks": chunks,
            "layer_norm_contribution": {
                name: float(torch.cat(values).mean().sqrt())
                for name, values in layer_values.items()
            },
            "group_norm_contribution": {
                name: float(torch.cat(values).mean().sqrt())
                for name, values in group_values.items()
            },
        }
        return AggregateResult(loss, matched, raw, clips, raw_sum, clipped_sum, stats)

    def remove(self) -> None:
        self.enabled = False
        self.activations.clear()
        self.backprops.clear()
        self.token_backprops.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

"""Small tensor helpers used by the Exp22 BK engine."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def flatten_linear_input(x: torch.Tensor, module: nn.Linear, augmented: bool = True) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    if augmented and module.bias is not None:
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    return x


def flatten_backprop(b: torch.Tensor) -> torch.Tensor:
    return b.reshape(-1, b.shape[-1])


def linear_covariances(
    activation: torch.Tensor,
    backprop: torch.Tensor | None,
    module: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    a = flatten_linear_input(activation, module)
    A = a.T @ a / a.shape[0]
    G = None if backprop is None else (flatten_backprop(backprop).T @ flatten_backprop(backprop) / a.shape[0])
    return A, G, a.shape[0]


def per_example_linear_gradient(
    activation: torch.Tensor,
    backprop: torch.Tensor,
    module: nn.Linear,
) -> torch.Tensor:
    b, t, _ = backprop.reshape(len(backprop), -1, backprop.shape[-1]).shape
    a = activation.reshape(b, -1, activation.shape[-1])
    if module.bias is not None:
        a = torch.cat((a, torch.ones_like(a[..., :1])), dim=-1)
    bp = backprop.reshape(b, -1, backprop.shape[-1])
    return torch.einsum("bto,bti->boi", bp, a)


def split_linear_gradient(g: torch.Tensor, module: nn.Linear) -> dict[nn.Parameter, torch.Tensor]:
    result = {module.weight: g[..., :-1] if module.bias is not None else g}
    if module.bias is not None:
        result[module.bias] = g[..., -1]
    return result


def layernorm_per_example_gradient(
    activation: torch.Tensor,
    backprop: torch.Tensor,
    module: nn.LayerNorm,
) -> dict[nn.Parameter, torch.Tensor]:
    normalized = F.layer_norm(activation, module.normalized_shape, eps=module.eps)
    n = len(activation)
    trailing = len(module.normalized_shape)
    reduce_dims = tuple(range(1, backprop.ndim - trailing))
    if not reduce_dims:
        weight = backprop * normalized
        bias = backprop
    else:
        weight = (backprop * normalized).sum(reduce_dims)
        bias = backprop.sum(reduce_dims)
    result = {module.weight: weight.reshape(n, -1).reshape(n, *module.weight.shape)}
    if module.bias is not None:
        result[module.bias] = bias.reshape(n, -1).reshape(n, *module.bias.shape)
    return result


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()

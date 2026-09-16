"""TinyViT with explicit attention projections.

The reference model in ``dp_kfac.models`` uses packed MHA parameters.  This
module keeps the same computation but exposes q/k/v/out as four Linear maps.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ExplicitMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @classmethod
    def from_packed(cls, packed: nn.MultiheadAttention) -> "ExplicitMultiheadAttention":
        result = cls(packed.embed_dim, packed.num_heads, packed.in_proj_bias is not None)
        with torch.no_grad():
            q, k, v = packed.in_proj_weight.chunk(3, dim=0)
            result.q_proj.weight.copy_(q)
            result.k_proj.weight.copy_(k)
            result.v_proj.weight.copy_(v)
            if packed.in_proj_bias is not None:
                qb, kb, vb = packed.in_proj_bias.chunk(3, dim=0)
                result.q_proj.bias.copy_(qb)
                result.k_proj.bias.copy_(kb)
                result.v_proj.bias.copy_(vb)
            result.out_proj.load_state_dict(copy.deepcopy(packed.out_proj.state_dict()))
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        weights = scores.softmax(dim=-1)
        value = torch.matmul(weights, v)
        value = value.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        return self.out_proj(value)


class TinyViTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ExplicitMultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h)
        return x + self.ffn(self.norm2(x))


class TinyViT(nn.Module):
    """The repository TinyViT with explicit Q/K/V attention projections."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(TinyViTBlock(embed_dim, num_heads) for _ in range(num_blocks))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    @classmethod
    def from_packed(cls, packed: nn.Module) -> "TinyViT":
        """Copy an initialized repository TinyViT into the explicit model."""
        result = cls(
            img_size=packed.patch_size * int(packed.pos_embed.shape[1] ** 0.5),
            patch_size=packed.patch_size,
            in_channels=packed.patch_embed.in_features // (packed.patch_size ** 2),
            embed_dim=packed.patch_embed.out_features,
            num_heads=packed.blocks[1].num_heads,
            num_blocks=len(packed.blocks) // 4,
            num_classes=packed.head.out_features,
        ).to(device=packed.pos_embed.device, dtype=packed.pos_embed.dtype)
        with torch.no_grad():
            result.patch_embed.load_state_dict(copy.deepcopy(packed.patch_embed.state_dict()))
            result.pos_embed.copy_(packed.pos_embed)
            for block_index, block in enumerate(result.blocks):
                old = packed.blocks[4 * block_index:4 * block_index + 4]
                block.norm1.load_state_dict(copy.deepcopy(old[0].state_dict()))
                block.attn = ExplicitMultiheadAttention.from_packed(old[1]).to(
                    device=result.pos_embed.device, dtype=result.pos_embed.dtype)
                block.norm2.load_state_dict(copy.deepcopy(old[2].state_dict()))
                block.ffn.load_state_dict(copy.deepcopy(old[3].state_dict()))
            result.norm.load_state_dict(copy.deepcopy(packed.norm.state_dict()))
            result.head.load_state_dict(copy.deepcopy(packed.head.state_dict()))
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, -1, c * p * p)
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def convert_tinyvit(packed: nn.Module) -> TinyViT:
    return TinyViT.from_packed(packed)


def initialize(seed: int, device: torch.device | str = "cpu") -> TinyViT:
    """Initialize via the reference packed model, then split QKV."""
    torch.random.default_generator.manual_seed(seed)
    from dp_kfac.models import TinyViT as PackedTinyViT

    packed = PackedTinyViT(
        img_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_heads=4,
        num_blocks=2,
        num_classes=10,
    )
    return convert_tinyvit(packed).to(device)

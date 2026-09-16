"""Weight-preserving conversion from pretrained timm ViT to explicit BK maps."""
from __future__ import annotations

import copy
import os

from . import config as cfg

os.environ["HF_HOME"] = str(cfg.ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(cfg.ROOT / ".cache" / "torch")

import timm
import torch
from torch import nn
from torch.nn import functional as F


class ExplicitAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        packed = source.qkv
        width = packed.in_features
        self.num_heads = source.num_heads
        self.head_dim = width // self.num_heads
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            projection = nn.Linear(width, width, bias=packed.bias is not None,
                                   device=packed.weight.device, dtype=packed.weight.dtype)
            with torch.no_grad():
                projection.weight.copy_(packed.weight.chunk(3, dim=0)[index])
                if packed.bias is not None:
                    projection.bias.copy_(packed.bias.chunk(3, dim=0)[index])
            self.add_module(name, projection)
        self.out_proj = copy.deepcopy(source.proj)
        self.q_norm = copy.deepcopy(source.q_norm)
        self.k_norm = copy.deepcopy(source.k_norm)
        self.attn_drop = copy.deepcopy(source.attn_drop)
        self.proj_drop = copy.deepcopy(source.proj_drop)

    def forward(self, x):
        batch, tokens, width = x.shape
        q = self.q_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = F.scaled_dot_product_attention(
            self.q_norm(q), self.k_norm(k), v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        value = value.transpose(1, 2).reshape(batch, tokens, width)
        return self.proj_drop(self.out_proj(value))


class ExplicitBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = copy.deepcopy(source.norm1)
        self.attn = ExplicitAttention(source.attn)
        self.norm2 = copy.deepcopy(source.norm2)
        self.mlp = copy.deepcopy(source.mlp)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ExplicitViT(nn.Module):
    """ViT-Tiny/16, CLS pooling, and a parameter-free token backward hook."""

    def __init__(self, source):
        super().__init__()
        conv = source.patch_embed.proj
        self.img_size = source.patch_embed.img_size
        self.patch_size = source.patch_embed.patch_size
        self.embed_dim = source.embed_dim
        self.num_classes = source.num_classes
        self.patch_embed = nn.Linear(conv.weight[0].numel(), conv.out_channels,
                                     bias=conv.bias is not None,
                                     device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            self.patch_embed.weight.copy_(conv.weight.flatten(1))
            if conv.bias is not None:
                self.patch_embed.bias.copy_(conv.bias)
        self.cls_token = nn.Parameter(source.cls_token.detach().clone())
        self.pos_embed = nn.Parameter(source.pos_embed.detach().clone())
        self.token_hook = nn.Identity()
        self.pos_drop = copy.deepcopy(source.pos_drop)
        self.blocks = nn.ModuleList(ExplicitBlock(block) for block in source.blocks)
        self.norm = copy.deepcopy(source.norm)
        self.head = copy.deepcopy(source.head)

    def patchify(self, x):
        assert tuple(x.shape[-2:]) == self.img_size
        return F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

    def forward(self, x):
        x = self.patch_embed(self.patchify(x))
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), dim=1)
        x = self.pos_drop(self.token_hook(x + self.pos_embed))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def convert_vit(source):
    """Retain the source 1000-class classifier for conversion verification."""
    return ExplicitViT(source)


def initialize(seed: int, device: torch.device | str = "cpu"):
    pretrained = timm.create_model(cfg.MODEL_NAME, pretrained=True)
    model = convert_vit(pretrained)
    torch.random.default_generator.manual_seed(seed)
    model.head = nn.Linear(cfg.EMBED_DIM, cfg.NUM_CLASSES)
    model.num_classes = cfg.NUM_CLASSES
    model.requires_grad_(True)
    return model.to(device)

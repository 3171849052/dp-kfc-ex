"""Pretrained timm ViT-Tiny with equivalent Linear maps for BK/Ghost."""
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


class LinearPatchEmbed(nn.Module):
    """Non-overlapping Conv2d patches, preserving channel/kernel ordering."""

    def __init__(self, patch_embed):
        super().__init__()
        conv = patch_embed.proj
        self.img_size = patch_embed.img_size
        self.patch_size = patch_embed.patch_size
        self.grid_size = patch_embed.grid_size
        self.num_patches = patch_embed.num_patches
        self.proj = nn.Linear(conv.weight[0].numel(), conv.out_channels,
                              bias=conv.bias is not None,
                              device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            self.proj.weight.copy_(conv.weight.flatten(1))
            if conv.bias is not None:
                self.proj.bias.copy_(conv.bias)
        self.norm = copy.deepcopy(patch_embed.norm)

    def forward(self, x):
        assert tuple(x.shape[-2:]) == self.img_size
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        return self.norm(self.proj(patches))


class SplitQKV(nn.Module):
    """Keep timm attention unchanged while exposing three affine geometries."""

    def __init__(self, packed):
        super().__init__()
        width = packed.in_features
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            projection = nn.Linear(width, width, bias=packed.bias is not None,
                                   device=packed.weight.device, dtype=packed.weight.dtype)
            with torch.no_grad():
                projection.weight.copy_(packed.weight.chunk(3, dim=0)[index])
                if packed.bias is not None:
                    projection.bias.copy_(packed.bias.chunk(3, dim=0)[index])
            self.add_module(name, projection)

    def forward(self, x):
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)


def convert_vit(model):
    """Convert a copy; all pretrained parameters and timm computation survive."""
    result = copy.deepcopy(model)
    result.patch_embed = LinearPatchEmbed(model.patch_embed)
    for block in result.blocks:
        block.attn.qkv = SplitQKV(block.attn.qkv)
    return result


def initialize(seed: int, device: torch.device | str = "cpu"):
    torch.random.default_generator.manual_seed(seed)
    pretrained = timm.create_model(cfg.MODEL_NAME, pretrained=True)
    pretrained.reset_classifier(cfg.NUM_CLASSES)
    model = convert_vit(pretrained)
    model.requires_grad_(True)
    return model.to(device)

"""Offline pretrained ViT-Tiny task bindings for ExpM1b."""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch import nn

from expm1b import REPO_ROOT, ROOT, configure_environment
from expm1b import config as cfg
from expm1b import data


CHECKPOINT_DIRECTORY = "models--timm--" + cfg.MODEL_NAME
SOURCE_CHECKPOINT = (
    REPO_ROOT / "exp30" / ".cache" / "huggingface" / "hub" / CHECKPOINT_DIRECTORY
)
LOCAL_CHECKPOINT = (
    cfg.CACHE_ROOT / "huggingface" / "hub" / CHECKPOINT_DIRECTORY
)


# exp22.model routes Hugging Face paths through exp22.config.ROOT while it is
# imported. Point that one import at ExpM1b, then restore the shared config.
from exp22 import config as _exp22_config

_original_exp22_root = _exp22_config.ROOT
_exp22_config.ROOT = ROOT
from exp22.model import convert_vit

_exp22_config.ROOT = _original_exp22_root
configure_environment()


def _checkpoint_weight(checkpoint: Path) -> Path:
    reference = checkpoint / "refs" / "main"
    assert reference.is_file(), f"missing local checkpoint revision: {reference}"
    revision = reference.read_text().strip()
    assert revision
    weight = checkpoint / "snapshots" / revision / "model.safetensors"
    assert weight.is_file(), f"missing local pretrained weights: {weight}"
    return weight


def prepare_checkpoint() -> Path:
    """Copy Exp30's existing checkpoint into the isolated ExpM1b cache."""
    source_weight = _checkpoint_weight(SOURCE_CHECKPOINT)
    if not LOCAL_CHECKPOINT.exists():
        shutil.copytree(SOURCE_CHECKPOINT, LOCAL_CHECKPOINT, symlinks=False)
    local_weight = _checkpoint_weight(LOCAL_CHECKPOINT)
    assert not local_weight.is_symlink()
    assert local_weight.stat().st_size == source_weight.stat().st_size
    return local_weight


def initialize(seed: int, device: torch.device | str = "cpu") -> nn.Module:
    assert seed in cfg.SEEDS
    prepare_checkpoint()
    configure_environment()
    import timm

    pretrained = timm.create_model(cfg.MODEL_NAME, pretrained=True)
    model = convert_vit(pretrained)
    del pretrained
    torch.random.default_generator.manual_seed(seed)
    model.head = nn.Linear(cfg.EMBED_DIM, cfg.NUM_CLASSES)
    model.num_classes = cfg.NUM_CLASSES
    model.requires_grad_(True)
    assert all(parameter.requires_grad for parameter in model.parameters())
    return model.to(device)


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    protocol = cfg.VIT
    assert protocol.optimizer == "adamw"
    assert protocol.betas is not None and protocol.optimizer_eps is not None
    return torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
        betas=protocol.betas,
        eps=protocol.optimizer_eps,
    )


def load_data():
    return data.cifar10_datasets()


def public_calibration(seed: int, epoch: int, device: torch.device | str, *, need_g: bool):
    return data.public_calibration("vit", seed, epoch, device, need_g=need_g)


def pink_calibration(seed: int, epoch: int, device: torch.device | str, *, need_g: bool = True):
    return data.pink_calibration("vit", seed, epoch, device, need_g=need_g)


def oracle_calibration(device: torch.device | str):
    return data.oracle_calibration("vit", device)


initialize_model = initialize

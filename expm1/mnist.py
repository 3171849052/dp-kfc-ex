"""MNIST task bindings for the fixed ExpM1 protocol."""
from __future__ import annotations

import torch
from dp_kfac.models import SimpleCNN

from expm1 import config as cfg
from expm1 import data


def initialize(seed: int, device: torch.device | str = "cpu") -> SimpleCNN:
    assert seed in cfg.SEEDS
    torch.random.default_generator.manual_seed(seed)
    model = SimpleCNN(in_channels=1, num_classes=cfg.NUM_CLASSES, img_size=28)
    model.num_classes = cfg.NUM_CLASSES
    model.requires_grad_(True)
    return model.to(device)


def build_optimizer(model: torch.nn.Module) -> torch.optim.SGD:
    protocol = cfg.MNIST
    assert protocol.optimizer == "sgd"
    return torch.optim.SGD(
        model.parameters(),
        lr=protocol.learning_rate,
        momentum=protocol.momentum,
        weight_decay=protocol.weight_decay,
    )


def load_data():
    return data.mnist_datasets()


def public_calibration(seed: int, epoch: int, device: torch.device | str):
    return data.public_calibration("mnist", seed, epoch, device)


def pink_calibration(seed: int, epoch: int, device: torch.device | str):
    return data.pink_calibration("mnist", seed, epoch, device)


def oracle_calibration(device: torch.device | str):
    return data.oracle_calibration("mnist", device)


initialize_model = initialize


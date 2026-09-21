"""Random initialization followed by the exact Exp22 explicit conversion."""
from exp34 import config as cfg
from exp22 import config as base_cfg
base_cfg.ROOT = cfg.ROOT  # Exp22.model sets cache paths at import.
from exp22.model import convert_vit
import timm
import torch
from torch import nn


def initialize(seed, device="cpu"):
    torch.manual_seed(seed)
    source = timm.create_model(cfg.MODEL_NAME, pretrained=False)
    model = convert_vit(source)
    model.head = nn.Linear(192, 10)
    model.num_classes = 10
    model.requires_grad_(True)
    return model.to(device)

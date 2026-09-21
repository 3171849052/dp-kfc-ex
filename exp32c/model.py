"""Random initialization with a seeded ten-class convolution."""
from exp32c import config as cfg
import torch
from torch import nn
from torchvision.models import squeezenet1_1

def initialize(seed, device):
    torch.manual_seed(seed)
    model = squeezenet1_1(weights=None)
    model.classifier[0].p = 0.0
    model.classifier[1] = nn.Conv2d(512, cfg.NUM_CLASSES, 1)
    model.num_classes = cfg.NUM_CLASSES
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False  # Preserve convolution output hooks.
    model.requires_grad_(True)
    return model.to(device)

def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                            betas=cfg.BETAS, eps=cfg.ADAM_EPS, weight_decay=0.0)

"""Explicit q/v LoRA; base parameters never enter training or geometry."""
import torch
from torch import nn
from exp31 import config as cfg

class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = cfg.LORA_ALPHA / cfg.LORA_RANK
        self.lora_A = nn.Linear(base.in_features, cfg.LORA_RANK, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(cfg.LORA_RANK, base.out_features, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

def selected_layers(model):
    names = [f'blocks.{i}.attn.{target}.{factor}' for i in range(12)
             for target in ('q_proj', 'v_proj') for factor in ('lora_A', 'lora_B')]
    modules = dict(model.named_modules())
    assert len(model.blocks) == 12
    assert all(isinstance(modules[n], nn.Linear) and modules[n].weight.requires_grad for n in names)
    expected = {n + '.weight' for n in names} | {'head.weight', 'head.bias'}
    assert {n for n, p in model.named_parameters() if p.requires_grad} == expected
    return names

def inject_lora(model):
    model.requires_grad_(False)
    for block in model.blocks:
        for target in ('q_proj', 'v_proj'):
            setattr(block.attn, target, LoRALinear(getattr(block.attn, target)))
    model.head.requires_grad_(True)
    selected_layers(model)
    return model

def parameter_metrics(model, initial):
    parameters = dict(model.named_parameters())
    def norm(predicate):
        return sum(p.detach().double().square().sum() for n, p in parameters.items() if predicate(n)).sqrt().item()
    lora = lambda n: '.lora_' in n
    return {
        'total_parameter_count': sum(p.numel() for p in parameters.values()),
        'trainable_parameter_count': sum(p.numel() for p in parameters.values() if p.requires_grad),
        'lora_parameter_count': sum(p.numel() for n, p in parameters.items() if lora(n)),
        'head_parameter_count': sum(p.numel() for n, p in parameters.items() if n.startswith('head.')),
        'lora_a_norm': norm(lambda n: '.lora_A.' in n),
        'lora_b_norm': norm(lambda n: '.lora_B.' in n),
        'q_proj_lora_norm': norm(lambda n: '.q_proj.lora_' in n),
        'v_proj_lora_norm': norm(lambda n: '.v_proj.lora_' in n),
        'head_norm': norm(lambda n: n.startswith('head.')),
        'frozen_backbone_unchanged': all(torch.equal(initial[n], p.detach()) for n, p in parameters.items() if not p.requires_grad),
        'trainable_lora_updated': all(not torch.equal(initial[n], p.detach()) for n, p in parameters.items() if lora(n)),
        'head_updated': all(not torch.equal(initial[n], p.detach()) for n, p in parameters.items() if n.startswith('head.')),
    }

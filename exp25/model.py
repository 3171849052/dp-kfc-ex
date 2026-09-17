import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from dp_kfac.models import CrossViTClassifier


def make_model(seed, device='cpu', pretrained=True):
    # Construction consumes only this seeded scope, identically for all conditions.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CrossViTClassifier(pretrained=pretrained)
    model.eval()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert set(trainable) == {'classifier.weight', 'classifier.bias'}
    assert model.classifier.in_features == 288
    assert sum(p.numel() for p in trainable.values()) == 28900
    return model.to(device)

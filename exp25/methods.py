"""Single Linear specialization of Exp21's output-anchor BK+GD semantics."""
import torch
import torch.nn.functional as F
from .config import CLIP, SAMPLE_RATE


def first_pass(model, x, y):
    model.zero_grad(set_to_none=True)
    params = list(model.classifier.parameters())
    for p in params:
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            features = model.backbone(x)
        anchor = model.classifier(features).requires_grad_(True)
        losses = F.cross_entropy(anchor, y, reduction='none')
        backprop, = torch.autograd.grad(losses.sum(), anchor)
        count = sum(p.grad is not None for p in params)
        assert count == 0
    finally:
        for p in params:
            p.requires_grad_(True)
    a = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
    return losses.detach(), a, backprop.detach(), count


@torch.no_grad()
def reconstruct(a, b, operator=None, clip=CLIP):
    if operator is not None:
        a = operator.transform_activation('classifier', a)
        if hasattr(operator, 'transform_backprop'):
            b = operator.transform_backprop('classifier', b)
    # || b_i a_i^T ||_F = ||b_i|| ||a_i||; no B x O x I tensor.
    norms = a.norm(dim=1) * b.norm(dim=1)
    factors = (clip / (norms + 1e-6)).clamp(max=1.)
    aggregate = (b * factors[:, None]).T @ a
    return aggregate, norms, factors


def aggregate(model, x, y, operator=None, clip=CLIP):
    losses, a, b, count = first_pass(model, x, y)
    summed, norms, factors = reconstruct(a, b, operator, clip)
    return losses, summed, norms, factors, count


@torch.no_grad()
def update(model, optimizer, summed, sigma, batch_size, generator, accountant, clip=CLIP, sample_rate=SAMPLE_RATE):
    noise = torch.randn(summed.shape, dtype=summed.dtype, device=summed.device,
                        generator=generator) * (sigma * clip)
    gradient = (summed + noise) / batch_size
    model.classifier.weight.grad = gradient[:, :-1].contiguous()
    model.classifier.bias.grad = gradient[:, -1].contiguous()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

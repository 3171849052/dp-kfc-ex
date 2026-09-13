"""Two backwards; tiled spatial Gram norms, with no per-example parameter gradients."""
import torch
from torch.nn import functional as F
from exp12.curvature import layers


class GhostNorm:
    def __init__(self, model, operator):
        self.operator = operator
        self.enabled = False
        self.activations = {}
        self.handles = [m.register_forward_hook(self.forward_hook(n)) for n, m in layers(model).items()]

    def forward_hook(self, name):
        def hook(module, inputs, output):
            if self.enabled:
                self.activations[name] = inputs[0].detach()
                output.register_hook(lambda b: self.norm_hook(name, module, b))
        return hook

    @torch.no_grad()
    def norm_hook(self, name, module, backprop):
        a = self.activations.pop(name)
        if isinstance(module, torch.nn.Conv2d):
            a = F.unfold(a, module.kernel_size, padding=module.padding, stride=module.stride)
            b = backprop.flatten(2)
        else:
            a, b = a.unsqueeze(-1), backprop.unsqueeze(-1)
        a = self.operator.transform_activation(name, torch.cat((a, torch.ones_like(a[:, :1])), 1))
        b = self.operator.transform_backprop(name, b)
        for start in range(0, a.shape[2], 32):
            aa = a[:, :, start:start+32].transpose(1, 2) @ a
            bb = b[:, :, start:start+32].transpose(1, 2) @ b
            self.norm_sq.add_((aa * bb).sum((1, 2)))

    def remove(self):
        for h in self.handles:
            h.remove()


def ghost_aggregate(model, hooks, x, y, bound=1.):
    model.zero_grad(set_to_none=True)
    hooks.norm_sq = torch.zeros(len(x), device=x.device)
    hooks.enabled = True
    losses = F.cross_entropy(model(x), y, reduction='none')
    losses.sum().backward()
    norms = hooks.norm_sq.clamp_min(0).sqrt()
    factors = (bound / (norms + 1e-6)).clamp(max=1).detach()
    hooks.enabled = False
    model.zero_grad(set_to_none=True)
    (F.cross_entropy(model(x), y, reduction='none') * factors).sum().backward()
    hooks.operator.transform_aggregate_gradient(model)
    return losses.detach().sum(), norms, factors


@torch.no_grad()
def noise_and_step(model, optimizer, sigma, batch_size, generator, bound=1.):
    for p in model.parameters():
        noise = torch.randn(p.numel(), device=p.device, dtype=p.dtype, generator=generator).reshape_as(p)
        p.grad.add_(noise, alpha=sigma * bound).div_(batch_size)
    optimizer.step()

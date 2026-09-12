"""Exact spatial-Gram norms followed by one weighted ordinary backward."""
import torch
import torch.nn.functional as F
from exp10 import run_exp10 as base


class GhostNorm:
    def __init__(self, model, operator):
        self.operator = operator
        self.enabled = False
        self.activations = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                self.handles.append(module.register_forward_hook(self.forward_hook(name)))

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
            assert module.groups == 1
            a = F.unfold(a, module.kernel_size, dilation=module.dilation,
                         padding=module.padding, stride=module.stride)
            b = backprop.flatten(2)
        else:
            a = a.reshape(a.shape[0], -1, a.shape[-1]).transpose(1, 2)
            b = backprop.reshape(backprop.shape[0], -1, backprop.shape[-1]).transpose(1, 2)
        if module.bias is not None:
            a = torch.cat((a, torch.ones_like(a[:, :1])), 1)
        a = self.operator.transform_activation(name, a)
        b = self.operator.transform_backprop(name, b)
        # ||sum_t b_t a_t^T||² = sum_st <b_s,b_t><a_s,a_t>.
        # Tile spatial positions, never batch examples or parameter gradients.
        for start in range(0, a.shape[2], 32):
            aa = a[:, :, start:start+32].transpose(1, 2) @ a
            bb = b[:, :, start:start+32].transpose(1, 2) @ b
            self.norm_sq.add_((aa * bb).sum((1, 2)))

    def remove(self):
        for handle in self.handles:
            handle.remove()


def clip_factors(norm_sq, bound):
    # Match the existing exact clipping convention, including its 1e-6 guard.
    return base._compute_clip_factors(norm_sq, bound)


def ghost_aggregate(model, hooks, x, y, bound=1.):
    device = x.device
    model.zero_grad(set_to_none=True)
    hooks.norm_sq = torch.zeros(len(x), device=device)
    hooks.enabled = True
    start = base.timestamp(device)
    losses = F.cross_entropy(model(x), y, reduction='none')
    losses.sum().backward()
    norms = hooks.norm_sq.clamp_min(0).sqrt()
    factors = clip_factors(hooks.norm_sq.clamp_min(0), bound).detach()
    first = base.timestamp(device) - start
    hooks.enabled = False
    model.zero_grad(set_to_none=True)
    start = base.timestamp(device)
    (F.cross_entropy(model(x), y, reduction='none') * factors).sum().backward()
    second = base.timestamp(device) - start
    start = base.timestamp(device)
    hooks.operator.transform_aggregate_gradient(model)
    transform = base.timestamp(device) - start
    return losses.detach().sum(), norms, factors, dict(
        ghost_first_pass_norm_seconds=first, ghost_second_pass_backward_seconds=second,
        aggregate_transform_seconds=transform)


def exact_aggregate(model, operator, x, y, bound=1.):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y, reduction='sum')
    loss.backward()
    operator.apply_exact(model)
    params = list(model.parameters())
    norm_sq = base._compute_per_sample_norms_squared(params, len(x), x.device)
    factors = clip_factors(norm_sq, bound)
    with torch.no_grad():
        for p in params:
            p.grad = (p.grad_sample.flatten(1) * factors[:, None]).sum(0).reshape_as(p)
            p.grad_sample = None
    return loss.detach(), norm_sq.sqrt(), factors, dict(ghost_first_pass_norm_seconds=0.,
        ghost_second_pass_backward_seconds=0., aggregate_transform_seconds=0.)


@torch.no_grad()
def noise_and_step(model, optimizer, sigma, batch_size=256, bound=1.):
    for p in model.parameters():
        # Flatten noise in both paths to preserve paired RNG draws.
        noise = torch.randn_like(p.grad.flatten()).reshape_as(p)
        p.grad.add_(noise, alpha=sigma * bound).div_(batch_size)
    optimizer.step()

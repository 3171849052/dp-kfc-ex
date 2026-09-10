"""Shape-only CLW controller for the repository's SimpleCNN."""
import math
import torch

LAYERS = ("conv1", "conv2", "fc1", "fc2")


def gradient_matrix(layer, per_sample=False):
    if per_sample:
        w, b = layer.weight.grad_sample, layer.bias.grad_sample
        return torch.cat((w.reshape(w.shape[0], w.shape[1], -1), b.unsqueeze(-1)), -1)
    return torch.cat((layer.weight.grad.flatten(1), layer.bias.grad[:, None]), 1)


def identity_factors(model):
    a, g = {}, {}
    for name in LAYERS:
        w = getattr(model._module, name).weight
        m, n = w.shape[0], w[0].numel() + 1
        g[name] = torch.eye(m, device=w.device, dtype=w.dtype)
        a[name] = torch.eye(n, device=w.device, dtype=w.dtype)
    return a, g


class CLWKron:
    def __init__(self, model):
        self.U_A, self.U_G = identity_factors(model)
        self.E_A = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in self.U_A.items()}
        self.E_G = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in self.U_G.items()}
        self.scales = None

    @torch.no_grad()
    def observe(self, model):
        # Called exclusively after clip_and_noise_gradients, before SGD momentum.
        for name in LAYERS:
            z = gradient_matrix(getattr(model._module, name)).double()
            m, n = z.shape
            for e, q in ((self.E_G[name], z @ z.T / n),
                         (self.E_A[name], z.T @ z / m)):
                q.diagonal().sub_(q.trace() / q.shape[0])
                e.mul_(0.99).add_(q, alpha=0.01)

    @torch.no_grad()
    def update(self):
        if self.scales is None:
            self.scales = [{k: e.norm() / math.sqrt(e.shape[0]) + 1e-12
                            for k, e in es.items()} for es in (self.E_G, self.E_A)]
        for es, us, scales, left in ((self.E_G, self.U_G, self.scales[0], True),
                                     (self.E_A, self.U_A, self.scales[1], False)):
            for name, e in es.items():
                r = e / scales[name]
                spectral_norm = torch.linalg.eigvalsh(r).abs().max()
                r_hat = r * (2 / spectral_norm).clamp(max=1)
                k = torch.matrix_exp(-0.1 / 2 * r_hat)
                u = us[name].double()
                us[name] = (k @ u if left else u @ k).to(us[name].dtype)
                e.zero_()

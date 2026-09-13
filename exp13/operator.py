"""Raw damped inverse square roots, including augmented biases."""
import torch
from exp12.curvature import layers
from exp13.config import DAMPING


def inverse_sqrt(c):
    # Factorization in double avoids cancellation in nearly singular A factors.
    c = c.double()
    e, q = torch.linalg.eigh((c + c.T) / 2 + DAMPING * torch.eye(len(c), device=c.device))
    assert (e > 0).all()
    return ((q * e.rsqrt()) @ q.T).float()


class Operator:
    def __init__(self, factors):
        self.data = {n: (inverse_sqrt(f['C']), inverse_sqrt(f['A']))
                     for n, f in factors.items()}

    def transform_activation(self, name, a):
        return self.data[name][1].T @ a

    def transform_backprop(self, name, b):
        return self.data[name][0] @ b

    def transform_matrix(self, name, g):
        left, right = self.data[name]
        return left @ g @ right

    @torch.no_grad()
    def transform_aggregate_gradient(self, model):
        for name, m in layers(model).items():
            g = torch.cat((m.weight.grad.flatten(1), m.bias.grad[:, None]), 1)
            g = self.transform_matrix(name, g)
            m.weight.grad.copy_(g[:, :-1].reshape_as(m.weight))
            m.bias.grad.copy_(g[:, -1])

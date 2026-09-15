"""Two-pass streaming randomized range finder: FΩ, then QᵀFQ.

No sample cache or d×d covariance for r<d. The caller replays synthetic
forward/VJP samples with the same labels on pass two. Oversampling is four.
"""
import torch
from exp18.operators import Factor
from exp18.config import OVERSAMPLING


class Sketch:
    def __init__(self, dimension, rank, device, generator):
        self.dimension, self.rank = dimension, min(rank, dimension)
        self.exact = rank >= dimension
        width = min(dimension, rank + OVERSAMPLING)
        self.omega = torch.randn(dimension, width, device=device, dtype=torch.float64, generator=generator)
        self.y = torch.zeros_like(self.omega)
        self.trace = torch.zeros((), device=device, dtype=torch.float64)
        self.count = 0
        if self.exact:
            self.y = torch.zeros(dimension, dimension, device=device, dtype=torch.float64)

    def first(self, x):
        x = x.double()
        self.trace.add_(x.square().sum())
        self.count += len(x)
        self.y.add_(x.T @ x if self.exact else x.T @ (x @ self.omega))

    def prepare(self):
        if not self.exact:
            self.q = torch.linalg.qr(self.y, mode='reduced').Q
            self.small = self.q.new_zeros(self.q.shape[1], self.q.shape[1])
        del self.omega

    def second(self, x):
        if not self.exact:
            z = x.double() @ self.q
            self.small.add_(z.T @ z)

    def finish(self):
        d, r = self.dimension, self.rank
        if self.exact:
            f = Factor('dense', d, self.y / self.count)
            return f, dict(rank=d, dimension=d, tau=0., captured_trace_fraction=1.)
        e, v = torch.linalg.eigh((self.small + self.small.T) / (2*self.count))
        e = e[-r:].clamp_min(0)
        u = (self.q @ v[:, -r:]).float()
        trace = self.trace.item() / self.count
        captured = e.sum().item()
        tau = (trace - captured) / (d-r)
        # Same zero-floor PSD roundoff convention as Exp14b; substantive
        # negative residuals fail instead of being repaired.
        assert tau >= -1e-12 * trace
        tau = max(tau, 0.)
        return Factor('lowrank', d, e, u, tau), dict(rank=r, dimension=d, tau=tau,
            captured_trace_fraction=captured/trace)

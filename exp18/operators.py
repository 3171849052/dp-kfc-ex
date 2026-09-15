"""Factor actions; only dense factors store dense powers."""
import math
import torch
from exp13.operator import Operator as AggregateInterface
from exp18.config import BETA, DAMPING


class Factor:
    def __init__(self, kind, dimension, values=None, basis=None, tau=None):
        self.kind, self.dimension = kind, dimension
        self.values, self.basis, self.tau = values, basis, tau
        if kind == 'dense':
            f = values.double()
            e, q = torch.linalg.eigh((f + f.T) / 2)
            # Same PSD roundoff treatment as Exp14b scale diagnostics.
            self.values = e.clamp_min(0)
            self.power = ((q * (self.values + DAMPING).pow(-BETA)) @ q.T).float()
        elif kind in ('diagonal', 'lowrank'):
            assert (values >= 0).all()
            self.power = (values + DAMPING).pow(-BETA).float()
            if kind == 'lowrank':
                assert tau >= 0
                self.tail_power = float((tau + DAMPING) ** (-BETA))

    def action(self, x):
        if self.kind == 'identity':
            return x
        if self.kind == 'dense':
            return self.power @ x
        if self.kind == 'diagonal':
            return self.power[:, None] * x
        u = self.basis
        return self.tail_power * x + u @ ((self.power - self.tail_power)[:, None] * (u.T @ x))

    def moment(self, beta):
        if self.kind == 'identity':
            return float(self.dimension)
        e = self.values.double()
        moment = (e * (e + DAMPING).pow(-2 * beta)).sum().item()
        if self.kind == 'lowrank':
            moment += (self.dimension - len(e)) * self.tau * (self.tau + DAMPING) ** (-2 * beta)
        return moment

    def compact(self):
        # Moments have been computed: retain only state used by action.
        self.values = None
        self.tau = None

    def tensors(self):
        return [v for v in (getattr(self, 'power', None), self.basis) if isinstance(v, torch.Tensor)]


class Operator(AggregateInterface):
    def __init__(self, factors):
        self.data = factors
        raw, ref = [sum(f['A'].moment(b) * f['C'].moment(b) for f in factors.values())
                    for b in (BETA, .5)]
        self.scale = math.sqrt(ref / raw)
        self.diagnostics = dict(predicted_second_moment_raw=raw, predicted_second_moment_ref=ref,
            scale_match=self.scale, predicted_rms_raw=math.sqrt(raw),
            predicted_rms_matched=self.scale * math.sqrt(raw))
        for f in factors.values():
            for side in f.values():
                side.compact()
        ts = [t for f in factors.values() for side in f.values() for t in side.tensors()]
        # Numerical operator state: tensor payload plus global scale and scalar tail powers.
        scalars = 1 + sum(side.kind == 'lowrank' for f in factors.values() for side in f.values())
        self.diagnostics.update(operator_state_bytes=sum(t.numel()*t.element_size() for t in ts)+8*scalars,
                                stored_scalar_count=sum(t.numel() for t in ts)+scalars)

    def transform_activation(self, name, a):
        return math.sqrt(self.scale) * self.data[name]['A'].action(a)

    def transform_backprop(self, name, b):
        return math.sqrt(self.scale) * self.data[name]['C'].action(b)

    def transform_matrix(self, name, g):
        f = self.data[name]
        return self.scale * f['A'].action(f['C'].action(g).transpose(-1, -2)).transpose(-1, -2)

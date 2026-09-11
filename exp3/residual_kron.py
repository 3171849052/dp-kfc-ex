"""Shape-only multiplicative residual on an epoch's synthetic KFAC base."""
import math
import torch
from clw_kron import CLWKron, LAYERS


def interval_ends(steps):
    lengths = [steps // 5 + (i < steps % 5) for i in range(5)]
    return [sum(lengths[:i]) for i in range(1, 5)]


class ResidualKron(CLWKron):
    def __init__(self, model, base_a, base_g, feedback_type):
        super().__init__(model)
        self.C_A, self.C_G = self.U_A, self.U_G
        self.U_A = {k: v.clone() for k, v in base_a.items()}
        self.U_G = {k: v.clone() for k, v in base_g.items()}
        self.feedback_type = feedback_type
        self.steps = 0
        self.update_steps = []
        # A new instance each epoch resets identity residuals, EMA and scales.
        assert self.scales is None
        for side in (self.C_A, self.C_G):
            for c in side.values():
                assert torch.equal(c, torch.eye(len(c), device=c.device))

    @torch.no_grad()
    def observe(self, model):
        for name in LAYERS:
            layer = getattr(model._module, name)
            z = torch.cat((getattr(layer.weight, self.feedback_type).flatten(1),
                           getattr(layer.bias, self.feedback_type)[:, None]), 1).double()
            m, n = z.shape
            for e, q in ((self.E_G[name], z @ z.T / n),
                         (self.E_A[name], z.T @ z / m)):
                q.diagonal().sub_(q.trace() / len(q))
                e.mul_(0.99).add_(q, alpha=0.01)
        self.steps += 1

    @torch.no_grad()
    def update(self):
        if self.scales is None:
            self.scales = [{k: e.norm() / math.sqrt(len(e)) + 1e-12
                            for k, e in es.items()} for es in (self.E_G, self.E_A)]
        rows = {name: dict(layer=name) for name in LAYERS}
        for side, es, us, cs, scales in (
            ('G', self.E_G, self.U_G, self.C_G, self.scales[0]),
            ('A', self.E_A, self.U_A, self.C_A, self.scales[1]),
        ):
            for name, e in es.items():
                r = e / scales[name]
                r_hat = r * (2 / torch.linalg.eigvalsh(r).abs().max()).clamp(max=1)
                k = torch.matrix_exp(-0.1 / 2 * r_hat)
                u, c = us[name].double(), cs[name].double()
                us[name] = (k @ u if side == 'G' else u @ k).to(us[name].dtype)
                cs[name] = k @ c if side == 'G' else c @ k
                eig = torch.linalg.eigvalsh(k)
                rows[name].update({
                    f'E_{side}_norm': e.norm().item(),
                    f'correction_{side}_norm': ((cs[name] - torch.eye(len(c), device=c.device)).norm() / math.sqrt(len(c))).item(),
                    f'K_{side}_min_eigenvalue': eig[0].item(),
                    f'K_{side}_max_eigenvalue': eig[-1].item(),
                })
                e.zero_()
        self.update_steps.append(self.steps)
        return list(rows.values())

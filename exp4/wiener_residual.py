"""Causal Kronecker Wiener feedback; clean shadows are diagnostics only."""
from types import SimpleNamespace
import torch
from residual_kron import ResidualKron, interval_ends, LAYERS


class WienerResidual(ResidualKron):
    def __init__(self, model, base_a, base_g, method, tau2, smoke=False):
        super().__init__(model, base_a, base_g, 'grad')
        self.a_only = method.endswith('AOnly')
        self.tau2, self.smoke = tau2, smoke
        self.reset_statistics()

    def reset_statistics(self):
        self.history = {}
        self.diagnostics = {k: [] for k in LAYERS}
        self.interval_count = 0

    @torch.no_grad()
    def observe(self, model):
        proxy = SimpleNamespace()
        originals = []
        current = {}
        expected = {}
        for name in LAYERS:
            layer = getattr(model._module, name)
            raw = torch.cat((layer.weight.grad.flatten(1), layer.bias.grad[:, None]), 1).double()
            m, n = raw.shape
            if self.smoke:
                originals.extend((p, p.grad.clone()) for p in (layer.weight, layer.bias))
            alpha = raw.new_zeros(())
            gain = torch.ones_like(raw)
            if self.interval_count:
                sa, sg, count = self.history[name]
                assert count == self.interval_count
                eigs, vectors = [], []
                for s in (sa, sg):
                    q = s / count - self.tau2 * torch.eye(len(s), device=s.device, dtype=s.dtype)
                    v, u = torch.linalg.eigh((q + q.T) / 2)
                    eigs.append(v.clamp_min(0))
                    vectors.append(u)
                va, vg = eigs
                ua, ug = vectors
                alpha = ((va.mean() + vg.mean()) / 2).clamp_min(1e-12)
                signal = vg[:, None] * va[None, :] / alpha
                gain = signal / (signal + self.tau2)
                filtered = ug @ (gain * (ug.T @ raw @ ua)) @ ua.T
            else:
                assert not self.history
                filtered = raw.clone()
                assert torch.equal(filtered, raw)
            # A separate view object lets Exp3 observe the filtered matrix without
            # ever assigning to the optimizer's parameters or gradient buffers.
            setattr(proxy, name, SimpleNamespace(
                weight=SimpleNamespace(grad=filtered[:, :-1]),
                bias=SimpleNamespace(grad=filtered[:, -1])))
            if self.smoke:
                for side, q in (('A', filtered.T @ filtered / m), ('G', filtered @ filtered.T / n)):
                    q.diagonal().sub_(q.trace()/len(q))
                    expected[side, name] = getattr(self, 'E_'+side)[name]*.99 + q*.01
            current[name] = raw
            clean = torch.cat((layer.weight.summed_grad.flatten(1), layer.bias.summed_grad[:, None]), 1).double()
            def cosine(z):
                return (z.flatten() @ clean.flatten() / (z.norm()*clean.norm()).clamp_min(1e-12)).item()
            quantiles = torch.quantile(gain.flatten(), gain.new_tensor([.1, .5, .9]))
            self.diagnostics[name].append(dict(
                alpha=alpha.item(), alpha_over_tau2=alpha.item()/self.tau2,
                wiener_gain_mean=gain.mean().item(), wiener_gain_median=quantiles[1].item(),
                wiener_gain_p10=quantiles[0].item(), wiener_gain_p90=quantiles[2].item(),
                wiener_gain_min=gain.min().item(), wiener_gain_max=gain.max().item(),
                cos_raw=cosine(raw), cos_filtered=cosine(filtered),
                NSR_raw=((raw-clean).norm()/(clean.norm()+1e-12)).item(),
                NSR_filtered=((filtered-clean).norm()/(clean.norm()+1e-12)).item()))
        super().observe(SimpleNamespace(_module=proxy))
        for (side, name), value in expected.items():
            assert torch.allclose(getattr(self, 'E_'+side)[name], value)
        # Accumulate only AFTER filtering and controller observation.
        for name, raw in current.items():
            m, n = raw.shape
            qa, qg = raw.T @ raw / m, raw @ raw.T / n
            if self.interval_count:
                sa, sg, count = self.history[name]
                self.history[name] = (sa + qa, sg + qg, count + 1)
            else:
                self.history[name] = (qa, qg, 1)
        self.interval_count += 1
        for p, original in originals:
            assert torch.equal(p.grad, original), 'Feedback changed optimizer gradient'

    def interval_metrics(self):
        rows = []
        for name, batches in self.diagnostics.items():
            # Alpha is undefined on identity warmup; average it only over fitted filters.
            fitted = batches[1:]
            row = {key: sum(b[key] for b in batches)/len(batches) for key in batches[0]}
            for key in ('alpha', 'alpha_over_tau2'):
                row[key] = sum(b[key] for b in fitted)/len(fitted) if fitted else 0.
            row.update(layer=name, tau2=self.tau2, interval_steps=len(batches),
                       fitted_batches=len(fitted), first_batch_identity=True)
            rows.append(row)
        return rows

    @torch.no_grad()
    def update(self):
        if self.a_only:
            # Exp3's update loops over E_G; an empty mapping skips G entirely.
            original_g = {k: v.clone() for k, v in self.U_G.items()}
            eg = self.E_G
            self.E_G = {}
        rows = super().update()
        if self.a_only:
            self.E_G = eg
            for row in rows:
                name = row['layer']
                assert torch.equal(self.U_G[name], original_g[name])
                c = self.C_G[name]
                assert torch.equal(c, torch.eye(len(c), device=c.device, dtype=c.dtype))
                row.update(E_G_norm=eg[name].norm().item(), correction_G_norm=0.,
                           K_G_min_eigenvalue=1., K_G_max_eigenvalue=1.)
                eg[name].zero_()
        self.reset_statistics()
        return rows

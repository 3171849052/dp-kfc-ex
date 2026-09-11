"""Fixed SimpleCNN controller with interval-level clean/noisy diagnostics."""
import torch
from clw_kron import CLWKron, LAYERS


class Exp2Controller(CLWKron):
    def __init__(self, model, sensor_type):
        super().__init__(model)
        self.sensor_type = sensor_type
        self.shadow = {
            sensor: {side: {name: torch.zeros_like(e) for name, e in es.items()}
                     for side, es in (("A", self.E_A), ("G", self.E_G))}
            for sensor in ("clean", "noisy")
        }
        self.E_A = self.shadow[sensor_type]["A"]
        self.E_G = self.shadow[sensor_type]["G"]
        self.update_count = 0

    @torch.no_grad()
    def observe(self, model):
        for sensor, attribute in (("clean", "summed_grad"), ("noisy", "grad")):
            for name in LAYERS:
                layer = getattr(model._module, name)
                x = torch.cat((getattr(layer.weight, attribute).flatten(1),
                               getattr(layer.bias, attribute)[:, None]), 1).double()
                m, n = x.shape
                for side, q in (("G", x @ x.T / n), ("A", x.T @ x / m)):
                    q.diagonal().sub_(q.trace() / q.shape[0])
                    self.shadow[sensor][side][name].mul_(0.99).add_(q, alpha=0.01)

    @torch.no_grad()
    def feedback_metrics(self):
        rows = []
        for name in LAYERS:
            row = dict(layer=name)
            for side in ("G", "A"):
                clean = self.shadow["clean"][side][name]
                noisy = self.shadow["noisy"][side][name]
                cn, nn = clean.norm(), noisy.norm()
                row.update({f"NSR_{side}": ((noisy-clean).norm()/(cn+1e-12)).item(),
                            f"cos_{side}": ((noisy*clean).sum()/(nn*cn+1e-12)).item(),
                            f"clean_{side}_norm": cn.item(), f"noisy_{side}_norm": nn.item()})
            rows.append(row)
        return rows

    @torch.no_grad()
    def update(self):
        super().update()  # Exact Exp1 scales, spectral scaling and multiplicative update.
        for sides in self.shadow.values():
            for es in sides.values():
                for e in es.values():
                    e.zero_()
        self.update_count += 1


def interval_ends(steps, frequency):
    return [steps * i // frequency for i in range(1, frequency + 1)]

"""Exp21 BK reconstruction and norm routing, extended to left KFC factors."""
import torch
from exp21.bk import BookKeeping
from exp22.methods import noise_and_step
from exp32c.geometry import conv_modules


class Clipper(BookKeeping):
    def __init__(self, model, operator, method='bk', max_grad_norm=1.0):
        assert method == 'bk'
        modules = conv_modules(model)
        super().__init__(model, operator, max_grad_norm=max_grad_norm)
        assert not self.fallback_params
        self.preconditioned_layers = sorted(operator.data) if operator else []
        self.identity_geometry_layers = sorted(set(modules)-set(self.preconditioned_layers))
        self.optimizer_steps = self.noise_events = 0

    def capture(self, name):
        # Transform only the detached BK record, never the propagated gradient.
        def hook(module, args, output):
            if not self.enabled:
                return
            for p in module.parameters(recurse=False):
                self.use_counts[id(p)] = self.use_counts.get(id(p), 0) + 1
            activation = [args[0].detach()]
            def backward(grad):
                if self.enabled:
                    with torch.no_grad():
                        b = grad.detach()
                        if self.operator is not None:
                            b = self.operator.transform_backprop(name, b)
                        self.pending.append((name, module, activation.pop(), b))
            output.register_hook(backward)
        return hook

    def aggregate_logical(self, x, y, physical_batch_size):
        assert len(x) == physical_batch_size
        loss, norms, factors, layer_sq, stats = self.aggregate(x, y, method='bk')
        stats['cache_empty_after_step'] = not self.pending and not self.records
        stats['layer_group_sq'] = {'conv': sum(layer_sq.values())}
        return loss, norms, factors, layer_sq, stats

    def step(self, optimizer, sigma, batch_size, generator):
        noise_and_step(self.model, optimizer, sigma, batch_size, generator, self.max_grad_norm)
        self.optimizer_steps += 1
        self.noise_events += 1

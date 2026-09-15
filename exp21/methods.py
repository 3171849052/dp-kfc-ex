"""Exp19 explicit baseline and Exp21 hybrid/two-pass engines."""
import torch
from opacus import GradSampleModule
from exp19.methods import noise_and_step, noise_and_normalize
from exp20.methods import AOperator, build_from_cache, synthetic_cache
from exp21.bk import BookKeeping, transform_aggregate
from exp21.profiling import timed, PHASES


class Clipper:
    def __init__(self, model, operator, method, strategy='auto'):
        self.model, self.operator, self.method = model, operator, method
        if method == 'exact':
            self.wrapper = GradSampleModule(model, loss_reduction='sum')
            self.hooks = None
        else:
            strategy = {'fast2': 'fast', 'ghost2': 'ghost'}.get(method, strategy)
            self.hooks = BookKeeping(model, operator, strategy)

    def aggregate(self, x, y, profiler=None):
        if self.method != 'exact':
            return self.hooks.aggregate(x, y, self.method, profiler)
        stats = {k: 0. for k in PHASES}
        stats.update(bk_cache_bytes=0, fallback_layers=0, fallback_layer_names=[], fallback_layer_count=0,
                     requires_second_backward=False, ghost_layer_count=0, fast_layer_count=0,
                     bk_ghost_layers=[], bk_fast_layers=[], layer_strategies={}, backward_calls=1)
        with timed(profiler, 'first_pass_seconds', x.device):
            self.wrapper.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(self.wrapper(x), y, reduction='sum')
            loss.backward()
        stats['first_pass_parameter_grad_count'] = sum(p.grad is not None for p in self.model.parameters())
        stats['temporary_per_sample_grad_bytes'] = sum(p.grad_sample.numel()*p.grad_sample.element_size() for p in self.model.parameters())
        with timed(profiler, 'norm_seconds', x.device), torch.no_grad():
            layer_sq = {}
            for name, m in self.model.named_modules():
                if name not in self.operator.data:
                    continue
                g = torch.cat((m.weight.grad_sample.flatten(2), m.bias.grad_sample.unsqueeze(-1)), -1)
                g = self.operator.transform_matrix(name, g)
                m.weight.grad_sample = g[:, :, :-1].reshape_as(m.weight.grad_sample).contiguous()
                m.bias.grad_sample = g[:, :, -1].contiguous()
                layer_sq[name] = g.square().sum((1, 2))
            norms = sum(layer_sq.values()).sqrt()
            factors = (1/(norms+1e-6)).clamp(max=1)
            for p in self.model.parameters():
                p.grad = torch.einsum('b,bp->p', factors, p.grad_sample.flatten(1)).reshape_as(p)
                p.grad_sample = None
        return loss.detach(), norms, factors, layer_sq, stats

    def remove(self):
        if self.hooks is not None:
            self.hooks.remove()
        else:
            self.wrapper.remove_hooks()

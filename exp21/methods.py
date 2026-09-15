"""Exp19 explicit baseline and Exp21 hybrid/two-pass engines."""
import torch
from opacus import GradSampleModule
from exp19.methods import noise_and_step, noise_and_normalize
from exp21.geometry import AOperator, build_from_cache, synthetic_cache
from exp21.bk import BookKeeping, transform_aggregate
from exp21.routing import Routes
from exp21.fallback import transform_samples
from exp21.profiling import timed, PHASES


class Clipper:
    def __init__(self, model, operator=None, method='bk', strategy='auto', max_grad_norm=1.):
        if method not in ('exact', 'fast2', 'ghost2', 'bk', 'bk_gd') or max_grad_norm <= 0:
            raise ValueError('Invalid method/max_grad_norm')
        self.model, self.operator, self.method = model, operator, method
        self.max_grad_norm = max_grad_norm
        self.execution_method = method
        if method == 'exact':
            self.routes = Routes(model, operator)
            if self.routes.fallback_layer_names:
                self.hooks = BookKeeping(model, operator, 'fast', max_grad_norm=max_grad_norm)
                self.execution_method = 'fast2'
            else:
                self.wrapper = GradSampleModule(model, loss_reduction='sum')
                self.hooks = None
        else:
            strategy = {'fast2': 'fast', 'ghost2': 'ghost'}.get(method, strategy)
            self.hooks = BookKeeping(model, operator, strategy, max_grad_norm=max_grad_norm)

    def aggregate(self, x, y, profiler=None, loss_fn=None):
        if self.hooks is not None:
            return self.hooks.aggregate(x, y, self.execution_method, profiler, loss_fn)
        if loss_fn is None:
            loss_fn = lambda output, target: torch.nn.functional.cross_entropy(output, target, reduction='none')
        stats = {k: 0. for k in PHASES}
        stats.update(self.routes.metadata())
        stats.update(bk_cache_bytes=0, ghost_layer_count=0, fast_layer_count=0,
                     bk_ghost_layers=[], bk_fast_layers=[], layer_strategies={}, backward_calls=1)
        with timed(profiler, 'first_pass_seconds', x.device):
            self.wrapper.zero_grad(set_to_none=True)
            loss = loss_fn(self.wrapper(x), y).sum()
            loss.backward()
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        stats['first_pass_parameter_grad_count'] = sum(p.grad is not None for _, p in params)
        stats['temporary_per_sample_grad_bytes'] = sum(p.grad_sample.numel()*p.grad_sample.element_size() for _, p in params)
        with timed(profiler, 'norm_seconds', x.device), torch.no_grad():
            grads = {p: p.grad_sample for _, p in params}
            for _, p in params:
                p.grad_sample = None
            transform_samples(self.model, self.operator, grads)
            layer_sq = {n: grads[p].flatten(1).square().sum(1) for n, p in params}
            norms = sum(layer_sq.values()).clamp_min(0).sqrt()
            factors = (self.max_grad_norm/(norms+1e-6)).clamp(max=1)
            for _, p in params:
                p.grad = torch.einsum('b,bp->p', factors, grads[p].flatten(1)).reshape_as(p)
        return loss.detach(), norms, factors, layer_sq, stats

    def step(self, optimizer, sigma, batch_size, generator):
        """Use the exact same C for clipping and Gaussian noise."""
        noise_and_step(self.model, optimizer, sigma, batch_size, generator, bound=self.max_grad_norm)

    def remove(self):
        if self.hooks is not None:
            self.hooks.remove()
        else:
            self.wrapper.remove_hooks()


class HybridBKClipper(Clipper):
    """Default hybrid BK interface; registered fallback is whole-step FGC."""
    pass

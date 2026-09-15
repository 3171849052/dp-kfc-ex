"""One-backward hybrid norm and direct preconditioned Book-Keeping sum."""
import torch
from torch import nn
from torch.nn import functional as F
from exp21.handlers import Record, kind, nbytes
from exp21.profiling import timed, PHASES


def square(g):
    return (g.coalesce().values() if g.is_sparse else g).square().sum()


def merge(target, contributions):
    for p, g in contributions.items():
        if p not in target:
            target[p] = g
        elif target[p].is_sparse and not g.is_sparse:
            target[p] = g + target[p]
        else:
            target[p] = target[p] + g


class BookKeeping:
    def __init__(self, model, operator=None, strategy='auto', tile=64):
        if strategy not in ('auto', 'ghost', 'fast') or tile < 1:
            raise ValueError('Invalid norm strategy/tile')
        self.model, self.operator, self.strategy, self.tile = model, operator, strategy, tile
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.trainable = {id(p) for p in self.params}
        self.modules = {}
        owners = {}
        for name, m in model.named_modules():
            ps = [p for p in m.parameters(recurse=False) if id(p) in self.trainable]
            if ps:
                adapter = kind(m)  # Fail before any mutation.
                allowed = {'weight'} if adapter == 'embedding' else {'weight', 'bias'}
                unsupported = [n for n, p in m.named_parameters(recurse=False)
                               if id(p) in self.trainable and n not in allowed]
                if unsupported:
                    raise NotImplementedError(f'Unsupported trainable parameters in {name}: {unsupported}')
                self.modules[name] = m
                for p in ps:
                    owners.setdefault(id(p), []).append(name)
        # Shared A operators would define conflicting maps on the same parameter.
        # Identity geometry is explicit for tied Transformer primitives in v1.
        if operator is not None and any(len(v) > 1 for v in owners.values()):
            raise NotImplementedError('Shared parameters currently require identity geometry (operator=None)')
        self.pending, self.records, self.handles = [], [], []
        self.enabled, self.gd = False, False
        for name, m in self.modules.items():
            self.handles.append(m.register_forward_hook(self.capture(name)))

    def capture(self, name):
        def hook(module, args, output):
            if not self.enabled:
                return
            if self.gd and isinstance(module, nn.Embedding):
                output.requires_grad_(True)
            x = args[0].detach()
            def backward(b):
                self.pending.append((name, module, x, b.detach()))
            output.register_hook(backward)
        return hook

    def clear(self):
        self.pending.clear()
        self.records.clear()
        self.enabled = False

    def remove(self):
        self.clear()
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @torch.no_grad()
    def norms(self, stats):
        self.records = [Record(n, m, x, b, self.operator, self.trainable)
                        for n, m, x, b in self.pending]
        self.pending.clear()
        stats['bk_cache_bytes'] = sum(r.bytes() for r in self.records)
        counts = {}
        for r in self.records:
            for p in r.params.values():
                counts[id(p)] = counts.get(id(p), 0)+1
        shared = [r for r in self.records if any(counts[id(p)] > 1 for p in r.params.values())]
        # Include all parameters of affected records exactly once, with cross terms.
        total = self.records[0].b.new_zeros(len(self.records[0].b))
        choices, layer_sq = {}, {}
        for r in self.records:
            if r in shared:
                choices[r.name] = 'fast_shared'
                continue
            if r.kind == 'embedding':
                values = []
                for i in range(len(total)):
                    sparse = r.sample(i)
                    values.append(sum(square(g) for g in sparse.values()))
                    stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'],
                        sum(nbytes(g) for g in sparse.values()))
                    del sparse
                sq = torch.stack(values)
                choice = 'fast_sparse'
            else:
                choice = self.strategy
                complete = len(r.params) == len(list(r.module.parameters(recurse=False)))
                if r.kind == 'norm' or not complete:
                    choice = 'fast'
                elif choice == 'auto':
                    choice = 'ghost' if 2*r.b.shape[1]**2 <= r.z.shape[-1]*r.b.shape[-1] else 'fast'
                if choice == 'ghost':
                    sq = r.ghost(self.tile)
                else:
                    grads = r.fast()
                    stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'], sum(nbytes(g) for g in grads.values()))
                    sq = sum(g.flatten(1).square().sum(1) for g in grads.values())
                    del grads
            total.add_(sq)
            layer_sq[r.name] = sq
            choices[r.name] = choice
        if shared:
            vals = []
            for i in range(len(total)):
                merged = {}
                for r in shared:
                    merge(merged, r.sample(i))
                vals.append(sum(square(g) for g in merged.values()))
                stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'], sum(nbytes(g) for g in merged.values()))
                del merged
            sq = torch.stack(vals)
            total.add_(sq)
            layer_sq['shared_parameters_combined'] = sq
        stats['layer_strategies'] = choices
        stats['bk_ghost_layers'] = [n for n, c in choices.items() if c == 'ghost']
        stats['bk_fast_layers'] = [n for n, c in choices.items() if c != 'ghost']
        stats['ghost_layer_count'] = len(stats['bk_ghost_layers'])
        stats['fast_layer_count'] = len(stats['bk_fast_layers'])
        return total.clamp_min(0).sqrt(), layer_sq

    @torch.no_grad()
    def reconstruct(self, factors):
        for p in self.params:
            p.grad = torch.zeros_like(p)
        for r in self.records:
            for p, g in r.aggregate(factors).items():
                p.grad.add_(g)

    def aggregate(self, x, y, method='bk', profiler=None, loss_fn=None):
        if method not in ('bk', 'bk_gd', 'fast2', 'ghost2'):
            raise ValueError(method)
        if loss_fn is None:
            loss_fn = lambda output, target: F.cross_entropy(output, target, reduction='none')
        stats = {key: 0. for key in PHASES}
        stats.update(bk_cache_bytes=0, temporary_per_sample_grad_bytes=0,
                     fallback_layers=0, fallback_layer_names=[], fallback_layer_count=0,
                     requires_second_backward=method in ('fast2', 'ghost2'),
                     backward_calls=0, first_pass_parameter_grad_count=0)
        self.clear()
        self.gd = method == 'bk_gd'
        try:
            with timed(profiler, 'first_pass_seconds', x.device):
                self.model.zero_grad(set_to_none=True)
                if self.gd:
                    for p in self.params:
                        p.requires_grad_(False)
                    if x.is_floating_point():
                        x = x.detach().requires_grad_(True)
                self.enabled = True
                losses = loss_fn(self.model(x), y)
                losses.sum().backward()
                stats['backward_calls'] += 1
                stats['first_pass_parameter_grad_count'] = sum(p.grad is not None for p in self.params)
                self.enabled = False
                if self.gd:
                    assert stats['first_pass_parameter_grad_count'] == 0
                    for p in self.params:
                        p.requires_grad_(True)
            with timed(profiler, 'norm_seconds', x.device):
                norms, layer_sq = self.norms(stats)
                factors = (1/(norms+1e-6)).clamp(max=1).detach()
            if method in ('bk', 'bk_gd'):
                with timed(profiler, 'bk_reconstruction_seconds', x.device):
                    self.reconstruct(factors)
            else:
                self.clear()
                with timed(profiler, 'second_pass_seconds', x.device):
                    self.model.zero_grad(set_to_none=True)
                    (loss_fn(self.model(x.detach()), y)*factors).sum().backward()
                    stats['backward_calls'] += 1
                with timed(profiler, 'aggregate_transform_seconds', x.device):
                    transform_aggregate(self.model, self.operator)
            return losses.detach().sum(), norms, factors, layer_sq, stats
        finally:
            for p in self.params:
                p.requires_grad_(True)
            self.clear()
            self.gd = False


@torch.no_grad()
def transform_aggregate(model, operator):
    if operator is None:
        return
    for name, m in model.named_modules():
        if name not in operator.data:
            continue
        from exp21.handlers import LINEAR_LAYOUTS
        w = m.weight.grad.T if type(m) in LINEAR_LAYOUTS else m.weight.grad.flatten(1)
        g = torch.cat((w, m.bias.grad[:, None]), -1) if m.bias is not None else w
        g = operator.transform_matrix(name, g)
        weight = g[:, :-1] if m.bias is not None else g
        if type(m) in LINEAR_LAYOUTS:
            weight = weight.T
        m.weight.grad.copy_(weight.reshape_as(m.weight))
        if m.bias is not None:
            m.bias.grad.copy_(g[:, -1])

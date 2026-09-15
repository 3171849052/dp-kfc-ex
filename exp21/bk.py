"""One-backward hybrid norm and direct preconditioned Book-Keeping sum."""
import torch
from torch import nn
from torch.nn import functional as F
from exp21.handlers import Record, nbytes, retained_bytes, LINEAR_LAYOUTS
from exp21.routing import Routes
from exp21.fallback import whole_step, rng_state, replay_rng
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
    def __init__(self, model, operator=None, strategy='auto', tile=64, max_grad_norm=1.):
        if strategy not in ('auto', 'ghost', 'fast') or tile < 1 or max_grad_norm <= 0:
            raise ValueError('Invalid norm strategy/tile/max_grad_norm')
        self.model, self.operator, self.strategy, self.tile = model, operator, strategy, tile
        self.max_grad_norm = max_grad_norm
        self.routes = Routes(model, operator)
        self.params, self.trainable = self.routes.params, self.routes.trainable
        self.modules = self.routes.modules
        self.pending, self.records, self.handles = [], [], []
        self.enabled, self.gd = False, False
        self.batch_size, self.anchors = 0, []
        if not self.routes.fallback_layer_names:
            for name, m in self.modules.items():
                self.handles.append(m.register_forward_hook(self.capture(name)))

    def capture(self, name):
        def hook(module, args, output):
            if not self.enabled:
                return
            x = args[0].detach()
            # Typed HF adapters: position embeddings are evaluated once, then
            # broadcast across examples. Hook the expanded output BEFORE addition
            # so its backprop retains the example dimension and cross terms.
            if id(module) in self.routes.broadcast_embeddings:
                if x.ndim == 1:
                    x, output = x.unsqueeze(0), output.unsqueeze(0)
                x = x.expand(self.batch_size, *x.shape[1:])
                output = output.expand(self.batch_size, *output.shape[1:])
            if self.gd and not output.requires_grad:
                output.requires_grad_(True)
                self.anchors.append(name)
            # Store x in a disposable slot, not in a graph-lifetime closure.
            activation = [x]
            def backward(b):
                self.pending.append((name, module, activation.pop(), b.detach()))
            output.register_hook(backward)
            return output
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
        self.records = []
        self.pending.reverse()
        while self.pending:
            n, m, x, b = self.pending.pop()
            self.records.append(Record(n, m, x, b, self.operator, self.trainable))
            # Release each source immediately after its z is formed, rather than
            # retaining all source activations until the last layer is converted.
            del x, b
        stats['bk_cache_bytes'] = retained_bytes(self.records)
        seen = {id(p) for r in self.records for p in r.params.values()}
        if seen != self.trainable:
            missing = [n for n, p in self.model.named_parameters() if id(p) in self.trainable-seen]
            raise RuntimeError(f'Trainable parameters did not produce a supported output backprop: {missing}')
        counts = {}
        for r in self.records:
            for p in r.params.values():
                counts[id(p)] = counts.get(id(p), 0)+1
        shared = [r for r in self.records if any(counts[id(p)] > 1 for p in r.params.values())]
        # Include all parameters of affected records exactly once, with cross terms.
        total = self.records[0].b.new_zeros(len(self.records[0].b))
        choices, layer_sq = {}, {}
        # Common LM tie: sparse embedding rows + a dense output projection.
        # Never form even one full per-example vocabulary-by-hidden matrix.
        pairs, paired = [], set()
        for emb in shared:
            if emb.kind != 'embedding':
                continue
            weight = emb.params['weight']
            others = [r for r in shared if r is not emb and any(p is weight for p in r.params.values())]
            if len(others) == 1:
                head = others[0]
                if (head.kind == 'linear' and not isinstance(head.module, nn.Conv2d)
                        and head.params.get('weight') is weight
                        and all(counts[id(p)] == 1 for n, p in head.params.items() if n != 'weight')
                        and len(head.params) == len(list(head.module.parameters(recurse=False)))):
                    pairs.append((emb, head))
                    paired.update((emb, head))
        for emb, head in pairs:
            sq = head.ghost(self.tile)
            cross_values, embed_values = [], []
            for i in range(len(total)):
                g = emb.sample(i)[emb.params['weight']].coalesce()
                ids, values = g.indices()[0], g.values()
                if type(head.module) in LINEAR_LAYOUTS:
                    rows = head.z[i, :, ids].T @ head.b[i]
                else:
                    z = head.z[i, :, :-1] if head.module.bias is not None else head.z[i]
                    rows = head.b[i, :, ids].T @ z
                cross_values.append(2*(values*rows).sum())
                embed_values.append(values.square().sum())
                stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'], nbytes(g)+nbytes(rows))
            sq = sq+torch.stack(embed_values)+torch.stack(cross_values)
            total.add_(sq)
            layer_sq[emb.name+'<->'+head.name] = sq
            choices[emb.name], choices[head.name] = 'fast_sparse_tied', 'ghost_tied'
        shared = [r for r in shared if r not in paired]
        if any(r.kind == 'embedding' for r in shared):
            raise NotImplementedError('Repeated/multi-owner tied Embedding needs explicit whole-step FGC registration')
        for r in self.records:
            if r in paired:
                continue
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
        stats['bk_ghost_layers'] = [n for n, c in choices.items() if c.startswith('ghost')]
        stats['bk_fast_layers'] = [n for n, c in choices.items() if not c.startswith('ghost')]
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
        stats.update(self.routes.metadata(second=method in ('fast2', 'ghost2')))
        stats.update(bk_cache_bytes=0, temporary_per_sample_grad_bytes=0,
                     backward_calls=0, first_pass_parameter_grad_count=0)
        if self.routes.fallback_layer_names:
            return whole_step(self.model, self.routes, self.operator, x, y, loss_fn, self.max_grad_norm, profiler)
        self.clear()
        self.gd = method == 'bk_gd'
        self.batch_size, self.anchors = len(x), []
        state = rng_state(x.device) if method in ('fast2', 'ghost2') else None
        input_handle = (x.register_hook(lambda g: stats.update(input_gradient_computed=True))
                        if x.requires_grad else None)
        try:
            with timed(profiler, 'first_pass_seconds', x.device):
                self.model.zero_grad(set_to_none=True)
                if self.gd:
                    for p in self.params:
                        p.requires_grad_(False)
                    x = x.detach()
                self.enabled = True
                losses = loss_fn(self.model(x), y)
                if losses.shape != (len(x),):
                    raise ValueError('loss_fn must return one scalar per example')
                losses.sum().backward()
                stats['backward_calls'] += 1
                stats['first_pass_parameter_grad_count'] = sum(p.grad is not None for p in self.params)
                self.enabled = False
                stats['gd_applied'] = self.gd
                stats['gd_anchor_modules'] = list(self.anchors)
                stats['gd_anchor_module'] = self.anchors[0] if self.anchors else None
                if self.gd:
                    assert stats['first_pass_parameter_grad_count'] == 0
                    for p in self.params:
                        p.requires_grad_(True)
            with timed(profiler, 'norm_seconds', x.device):
                norms, layer_sq = self.norms(stats)
                factors = (self.max_grad_norm/(norms+1e-6)).clamp(max=1).detach()
            if method in ('bk', 'bk_gd'):
                with timed(profiler, 'bk_reconstruction_seconds', x.device):
                    self.reconstruct(factors)
            else:
                self.clear()
                with timed(profiler, 'second_pass_seconds', x.device), replay_rng(state, x.device):
                    self.model.zero_grad(set_to_none=True)
                    (loss_fn(self.model(x.detach()), y)*factors).sum().backward()
                    stats['backward_calls'] += 1
                with timed(profiler, 'aggregate_transform_seconds', x.device):
                    transform_aggregate(self.model, self.operator)
            return losses.detach().sum(), norms, factors, layer_sq, stats
        finally:
            if input_handle is not None:
                input_handle.remove()
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

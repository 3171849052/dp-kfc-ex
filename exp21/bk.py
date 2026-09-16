"""One-backward hybrid norm and direct preconditioned Book-Keeping sum."""
import torch
from torch import nn
from torch.nn import functional as F
from exp21.handlers import Record, nbytes, retained_bytes, LINEAR_LAYOUTS, sample_norm_squared
from exp21.routing import Routes, norm_route
from exp21.fallback import chunked_norms, rng_state, replay_rng
from torch.nn.attention import sdpa_kernel, SDPBackend
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
    def __init__(self, model, operator=None, strategy='auto', tile=64, max_grad_norm=1.,
                 max_fast_temp_bytes=256*2**20, fallback_vjp_chunk_size=32,
                 fallback_memory_budget_bytes=None,
                 max_shared_sample_bytes=64*2**20, tied_output_chunk_size=256):
        if strategy not in ('auto', 'ghost', 'fast') or tile < 1 or max_grad_norm <= 0:
            raise ValueError('Invalid norm strategy/tile/max_grad_norm')
        self.model, self.operator, self.strategy, self.tile = model, operator, strategy, tile
        if min(max_fast_temp_bytes, fallback_vjp_chunk_size, max_shared_sample_bytes, tied_output_chunk_size) < 1:
            raise ValueError('Memory caps and chunk sizes must be positive')
        fallback_memory_budget_bytes = (max_fast_temp_bytes if fallback_memory_budget_bytes is None
                                        else fallback_memory_budget_bytes)
        if fallback_memory_budget_bytes < 1:
            raise ValueError('Memory caps and chunk sizes must be positive')
        self.max_grad_norm = max_grad_norm
        self.max_fast_temp_bytes = max_fast_temp_bytes
        self.fallback_vjp_chunk_size = fallback_vjp_chunk_size
        self.fallback_memory_budget_bytes = fallback_memory_budget_bytes
        self.max_shared_sample_bytes = max_shared_sample_bytes
        self.tied_output_chunk_size = tied_output_chunk_size
        self.routes = Routes(model, operator)
        self.params, self.trainable = self.routes.params, self.routes.trainable
        self.modules = self.routes.modules
        self.pending, self.records, self.handles = [], [], []
        self.enabled, self.gd = False, False
        self.batch_size, self.anchors = 0, []
        # Parameter identity is the routing fact. Names are display metadata
        # only, since named_parameters() removes aliases by default.
        self.fallback_ids = set(self.routes.fallback_parameter_ids)
        self._fallback_names = list(self.routes.fallback_parameter_names)
        self._fallback_shared_groups(max_shared_sample_bytes)
        self.sync_fallback()
        self.fallback_parameter_bytes = sum(p.numel()*p.element_size() for p in self.fallback_params)
        self.fallback_vjp_max_chunk_size = fallback_vjp_chunk_size
        self.fallback_vjp_effective_chunk_size = 0
        for name, m in self.modules.items():
            self.handles.append(m.register_forward_hook(self.capture(name)))

    def sync_fallback(self):
        # Move complete affine records, preserving A transforms and shared cross terms.
        changed = True
        while changed:
            changed = False
            for name, module in self.modules.items():
                ids = {id(p) for p in module.parameters(recurse=False) if id(p) in self.trainable}
                if ids & self.fallback_ids and not ids <= self.fallback_ids:
                    self.fallback_ids.update(ids)
                    changed = True
        self.fallback_params = [p for p in self.params if id(p) in self.fallback_ids]
        self.bk_trainable = self.trainable-self.fallback_ids
        self.routes.fallback_parameter_ids = set(self.fallback_ids)
        self.routes.fallback_params = list(self.fallback_params)
        self.routes.fallback_parameter_names = Routes.parameter_names(self.model, self.fallback_ids)
        self.fallback_parameter_bytes = sum(p.numel()*p.element_size() for p in self.fallback_params)
        for n, m in self.modules.items():
            if any(id(p) in self.fallback_ids for p in m.parameters(recurse=False)) and n not in self.routes.fallback_layer_names:
                self.routes.fallback_layer_names.append(n)

    def _fallback_shared_groups(self, limit):
        """Fall back whole connected shared groups when their merged sample
        gradient would exceed the reference-path memory bound.
        """
        modules = list(self.modules.items())
        module_params = {
            name: {id(p): p for p in module.parameters(recurse=False)
                   if id(p) in self.trainable}
            for name, module in modules
        }
        unseen = set(name for name, _ in modules)
        while unseen:
            seed = unseen.pop()
            group_names, group_ids = {seed}, set(module_params[seed])
            changed = True
            while changed:
                changed = False
                for name in list(unseen):
                    if group_ids & set(module_params[name]):
                        unseen.remove(name)
                        group_names.add(name)
                        group_ids.update(module_params[name])
                        changed = True
            if len(group_names) < 2:
                continue
            group_modules = [self.modules[name] for name in group_names]
            tied = (len(group_names) == 2 and any(isinstance(m, nn.Embedding) for m in group_modules)
                    and any(isinstance(m, nn.Linear) or type(m) in LINEAR_LAYOUTS for m in group_modules))
            group_params = {pid: next(module_params[name][pid] for name in group_names
                                      if pid in module_params[name]) for pid in group_ids}
            bytes_ = sum(p.numel()*p.element_size() for p in group_params.values())
            if bytes_ > limit and not tied:
                self.fallback_ids.update(group_ids)

    def _set_effective_fallback_chunk(self):
        if self.fallback_parameter_bytes:
            memory_limited = max(1, self.fallback_memory_budget_bytes//self.fallback_parameter_bytes)
            self.fallback_vjp_effective_chunk_size = min(
                self.batch_size, self.fallback_vjp_max_chunk_size, memory_limited)
        else:
            self.fallback_vjp_effective_chunk_size = 0

    def shared_overflow(self):
        shared_ids = {pid for module in self.modules.values()
                      for pid, p in {id(p): p for p in module.parameters(recurse=False)
                                     if id(p) in self.bk_trainable}.items()
                      if self.use_counts.get(pid, 0) > 1}
        analytic = set()
        for pid in shared_ids:
            owners = [m for m in self.modules.values()
                      if any(id(p) == pid for p in m.parameters(recurse=False))]
            if (len(owners) == 2 and self.use_counts[pid] == 2
                    and any(isinstance(m, nn.Embedding) for m in owners)
                    and any(isinstance(m, nn.Linear) or type(m) in LINEAR_LAYOUTS for m in owners)):
                analytic.add(pid)
        generic = {}
        for module in self.modules.values():
            ps = {id(p): p for p in module.parameters(recurse=False)
                  if id(p) in self.bk_trainable}
            if any(self.use_counts.get(pid, 0) > 1 and pid not in analytic for pid in ps):
                generic.update(ps)
        size = sum(p.numel()*p.element_size() for p in generic.values())
        return set(generic) if size > self.max_shared_sample_bytes else set()

    def capture(self, name):
        def hook(module, args, output):
            if not self.enabled:
                return
            ids = [id(p) for p in module.parameters(recurse=False) if id(p) in self.bk_trainable]
            if not ids:
                return
            for pid in ids:
                self.use_counts[pid] = self.use_counts.get(pid, 0)+1
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
                self.anchor_tensors.append(output)
            # Store x in a disposable slot, not in a graph-lifetime closure.
            activation = [x]
            def backward(b):
                if not self.enabled:
                    return
                if self.streaming and all(self.use_counts[pid] == 1 for pid in ids):
                    with torch.no_grad():
                        record = Record(name, module, activation.pop(), b.detach(), self.operator, self.bk_trainable)
                        self.stream_sq[name] = self.record_norm(record, self.stats)
                else:
                    self.pending.append((name, module, activation.pop(), b.detach()))
            output.register_hook(backward)
            return output
        return hook

    def clear(self):
        self.pending.clear()
        self.records.clear()
        self.enabled = False
        self.anchor_tensors = []

    def remove(self):
        self.clear()
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def record_norm(self, r, stats):
        route = norm_route(r, self.strategy, self.max_fast_temp_bytes, self.tile)
        choice = route['strategy']
        stats['layer_routing'][r.name] = route
        stats['layer_strategies'][r.name] = choice
        if choice == 'fast_sparse':
            samples, ids, values = r.embedding_rows()
            stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'], nbytes(values))
            sq = r.b.new_zeros(len(r.b))
            sq.index_add_(0, samples, values.square().sum(-1))
            return sq
        if choice == 'ghost':
            return r.ghost(self.tile)
        if choice == 'chunked_fast':
            chunk = route['output_chunk_size']
            stats['temporary_per_sample_grad_bytes'] = max(
                stats['temporary_per_sample_grad_bytes'], r.affine_workspace(chunk))
            return r.chunked_norm(chunk)
        grads = r.fast()
        stats['temporary_per_sample_grad_bytes'] = max(
            stats['temporary_per_sample_grad_bytes'], sum(nbytes(g) for g in grads.values()))
        return sample_norm_squared(grads)

    @torch.no_grad()
    def norms(self, stats):
        for name in self.routes.fallback_layer_names:
            stats['layer_routing'][name] = dict(strategy='fallback',
                estimated_fast_bytes=self.fallback_vjp_effective_chunk_size*self.fallback_parameter_bytes,
                estimated_full_fast_bytes=0, estimated_chunked_fast_bytes=0,
                output_chunk_size=self.fallback_vjp_effective_chunk_size,
                estimated_fast_cost=None, estimated_ghost_cost=None, routing_reason='bounded_fallback')
        self.pending.reverse()
        while self.pending:
            name, module, activation, backprop = self.pending.pop()
            self.records.append(Record(name, module, activation, backprop, self.operator, self.bk_trainable))
            del activation, backprop
        stats['bk_cache_bytes'] = retained_bytes(self.records)
        seen = {id(p) for r in self.records for p in r.params.values()}
        if not self.streaming and seen != self.bk_trainable:
            missing = [n for n, p in self.model.named_parameters() if id(p) in self.bk_trainable-seen]
            raise RuntimeError(f'Trainable parameters did not produce a supported output backprop: {missing}')
        counts = {}
        for r in self.records:
            for p in r.params.values():
                counts[id(p)] = counts.get(id(p), 0)+1
        shared = [r for r in self.records if any(counts[id(p)] > 1 for p in r.params.values())]
        # Include all parameters of affected records exactly once, with cross terms.
        total = self.norm_template.clone()
        choices, layer_sq = stats['layer_strategies'], dict(self.stream_sq)
        for sq in layer_sq.values():
            total.add_(sq)
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
            route = norm_route(head, self.strategy, self.max_fast_temp_bytes, self.tile)
            width = head.b.shape[-1]
            row_bytes = head.affine_workspace(1)
            chunk = min(self.tied_output_chunk_size, max(1, width//2),
                        width,
                        self.max_fast_temp_bytes//row_bytes if row_bytes else 0)
            chunk_feasible = chunk >= 1
            use_chunk = chunk_feasible and self.strategy != 'ghost'
            if self.strategy == 'auto' and use_chunk:
                use_chunk = (1.25*route['estimated_fast_macs']
                             < route['estimated_ghost_macs'])
            choice = 'chunked_fast_tied' if use_chunk else 'ghost_tied'
            if self.strategy == 'fast' and not use_chunk:
                raise RuntimeError('Forced Fast strategy exceeds the tied head workspace cap')
            if use_chunk:
                sq = head.chunked_norm(chunk)
                stats['temporary_per_sample_grad_bytes'] = max(
                    stats['temporary_per_sample_grad_bytes'], row_bytes*chunk)
            else:
                sq = head.ghost(self.tile)
            route.update(strategy=choice,
                         routing_reason='chunked_fast_compute' if use_chunk else 'ghost_compute',
                         output_chunk_size=chunk if use_chunk else 0,
                         candidate_output_chunk_size=chunk if chunk_feasible else 0,
                         estimated_chunked_fast_bytes=row_bytes*chunk if use_chunk else 0)
            stats['layer_routing'][head.name] = route
            stats['layer_routing'][emb.name] = dict(strategy='fast_sparse_tied',
                estimated_fast_bytes=0, estimated_fast_cost=0, estimated_ghost_cost=0,
                routing_reason='fast_sparse')
            samples, ids, values = emb.embedding_rows()
            stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'], nbytes(values))
            sq.index_add_(0, samples, values.square().sum(-1))
            # Only token rows, bounded in chunks, including the exact tie cross term.
            for start in range(0, len(ids), self.tied_output_chunk_size):
                ss, vv = samples[start:start+self.tied_output_chunk_size], ids[start:start+self.tied_output_chunk_size]
                if type(head.module) in LINEAR_LAYOUTS:
                    rows = (head.z[ss, :, vv].unsqueeze(-1)*head.b[ss]).sum(1)
                else:
                    z = head.z[..., :-1] if head.module.bias is not None else head.z
                    rows = (head.b[ss, :, vv].unsqueeze(-1)*z[ss]).sum(1)
                sq.index_add_(0, ss, 2*(values[start:start+self.tied_output_chunk_size]*rows).sum(-1))
                # ``values`` is the already-bounded sparse embedding-row
                # buffer; report the dominant affine chunk separately.
                stats['temporary_per_sample_grad_bytes'] = max(
                    stats['temporary_per_sample_grad_bytes'], nbytes(rows))
                del rows
            total.add_(sq)
            layer_sq[emb.name+'<->'+head.name] = sq
            choices[emb.name], choices[head.name] = 'fast_sparse_tied', choice
        shared = [r for r in shared if r not in paired]
        if any(r.kind == 'embedding' for r in shared):
            raise NotImplementedError('Repeated/multi-owner Embedding requires explicit bounded Fast fallback registration')
        for r in self.records:
            if r in paired:
                continue
            if r in shared:
                shared_choice = 'ghost_shared' if self.strategy == 'ghost' else 'fast_shared'
                choices[r.name] = shared_choice
                route = norm_route(r, 'ghost' if self.strategy == 'ghost' else 'fast',
                                   self.max_fast_temp_bytes, self.tile)
                route.update(strategy=shared_choice, routing_reason=shared_choice,
                             estimated_fast_bytes=sum(p.numel()*p.element_size()
                                                      for p in r.params.values()))
                stats['layer_routing'][r.name] = route
                continue
            sq = self.record_norm(r, stats)
            total.add_(sq)
            layer_sq[r.name] = sq
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
        if self.streaming:
            self.records.clear()
            stats['bk_cache_bytes'] = 0
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
        self.clear()
        self.streaming = method in ('fast2', 'ghost2')
        self.gd = method == 'bk_gd' or self.streaming or bool(self.fallback_params)
        self.batch_size, self.anchors, self.anchor_tensors = len(x), [], []
        self._set_effective_fallback_chunk()
        self.use_counts, self.stream_sq = {}, {}
        self.norm_template = self.params[0].new_zeros(len(x))
        stats = {key: 0. for key in PHASES}
        stats.update(self.routes.metadata(second=self.streaming))
        stats.update(bk_cache_bytes=0, temporary_per_sample_grad_bytes=0,
                     backward_calls=0, first_pass_parameter_grad_count=0,
                     first_pass_param_grad_disabled=self.gd, layer_routing={}, layer_strategies={},
                     first_pass_includes_streamed_norm=self.streaming,
                     fallback_vjp_chunk_size=self.fallback_vjp_chunk_size,
                     fallback_vjp_max_chunk_size=self.fallback_vjp_max_chunk_size,
                     fallback_vjp_effective_chunk_size=self.fallback_vjp_effective_chunk_size,
                     fallback_parameter_bytes=self.fallback_parameter_bytes,
                     fallback_memory_budget_bytes=self.fallback_memory_budget_bytes,
                     fallback_temporary_grad_bytes=0,
                     fallback_parameter_count=sum(p.numel() for p in self.fallback_params))
        self.stats = stats
        state = rng_state(x.device)
        input_handle = (x.register_hook(lambda g: stats.update(input_gradient_computed=True))
                        if x.requires_grad else None)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                with timed(profiler, 'first_pass_seconds', x.device):
                    self.model.zero_grad(set_to_none=True)
                    if self.gd:
                        for p in self.params:
                            if id(p) not in self.fallback_ids:
                                p.requires_grad_(False)
                        x = x.detach()
                    self.enabled = True
                    losses = loss_fn(self.model(x), y)
                    if losses.shape != (len(x),):
                        raise ValueError('loss_fn must return one scalar per example')
                    oversized = self.shared_overflow()
                    if oversized:
                        # Rebuild before any sample gradient allocation; the outer
                        # first-pass timer includes discovery and the replay forward.
                        self.fallback_ids.update(oversized)
                        self.sync_fallback()
                        self._set_effective_fallback_chunk()
                        del losses
                        self.clear()
                        self.gd, self.enabled = True, True
                        self.use_counts, self.stream_sq, self.anchors = {}, {}, []
                        stats.update(self.routes.metadata(second=self.streaming))
                        stats.update(first_pass_param_grad_disabled=True,
                                     fallback_parameter_count=sum(p.numel() for p in self.fallback_params),
                                     fallback_vjp_effective_chunk_size=self.fallback_vjp_effective_chunk_size,
                                     fallback_parameter_bytes=self.fallback_parameter_bytes)
                        for p in self.params:
                            p.requires_grad_(id(p) in self.fallback_ids)
                        x = x.detach()
                        with replay_rng(state, x.device):
                            losses = loss_fn(self.model(x), y)
                    if self.fallback_params:
                        torch.autograd.grad(losses.sum(), self.fallback_params+self.anchor_tensors, retain_graph=True)
                    elif self.gd:
                        torch.autograd.grad(losses.sum(), self.anchor_tensors)
                    else:
                        losses.sum().backward()
                    self.anchor_tensors.clear()
                    if not self.fallback_params:
                        losses = losses.detach()
                    stats['backward_calls'] += 1
                    stats['first_pass_parameter_grad_count'] = sum(p.grad is not None for p in self.params)
                    self.enabled = False
                    stats['gd_applied'] = method == 'bk_gd'
                    stats['gd_anchor_modules'] = list(self.anchors)
                    stats['gd_anchor_module'] = self.anchors[0] if self.anchors else None
                    for p in self.params:
                        p.requires_grad_(True)
                    if self.streaming:
                        # Shared/tied cross terms complete when all reverse hooks
                        # have arrived; release those records before pass one ends.
                        stream_result = self.norms(stats)
                with timed(profiler, 'norm_seconds', x.device):
                    norms, layer_sq = stream_result if self.streaming else self.norms(stats)
                    if self.fallback_params:
                        fallback_sq = chunked_norms(losses, self.fallback_params, self.model,
                                                    self.operator, self.fallback_vjp_effective_chunk_size, stats)
                        norms = (norms.square()+fallback_sq).clamp_min(0).sqrt()
                        layer_sq['fallback'] = fallback_sq
                        losses = losses.detach()
                    factors = (self.max_grad_norm/(norms+1e-6)).clamp(max=1).detach()
                if not self.streaming:
                    with timed(profiler, 'bk_reconstruction_seconds', x.device):
                        self.reconstruct(factors)
                if self.streaming or self.fallback_params:
                    self.clear()
                    with timed(profiler, 'second_pass_seconds', x.device), replay_rng(state, x.device):
                        weighted = (loss_fn(self.model(x.detach()), y)*factors).sum()
                        targets = self.params if self.streaming else self.fallback_params
                        grads = torch.autograd.grad(weighted, targets)
                        for p, g in zip(targets, grads):
                            p.grad = g
                        stats['backward_calls'] += 1
                    with timed(profiler, 'aggregate_transform_seconds', x.device):
                        transform_aggregate(self.model, self.operator, None if self.streaming else self.fallback_ids)
                return losses.detach().sum(), norms, factors, layer_sq, stats
        finally:
            if input_handle is not None:
                input_handle.remove()
            for p in self.params:
                p.requires_grad_(True)
            self.clear()
            self.stream_sq.clear()
            self.gd = False


@torch.no_grad()
def transform_aggregate(model, operator, parameter_ids=None):
    if operator is None:
        return
    for name, m in model.named_modules():
        if name not in operator.data or (parameter_ids is not None and id(m.weight) not in parameter_ids):
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

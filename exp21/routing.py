"""Explicit BK / bounded local Fast / unsupported routing, never name heuristics."""
from torch import nn
from dp_kfac.models import TinyViT
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from exp21.handlers import kind

# These modules are sample-independent but have packed or functional parameters.
FALLBACK_MODULE_TYPES = {nn.MultiheadAttention}
FALLBACK_DIRECT_TYPES = {TinyViT}


def register_fallback(cls, *, direct_parameters_only=False):
    """Opt in a known sample-independent module to batched-VJP FGC.

    The caller guarantees no cross-example operations or forward state mutation.
    Stateful normalization and mutating embeddings remain explicitly rejected.
    """
    (FALLBACK_DIRECT_TYPES if direct_parameters_only else FALLBACK_MODULE_TYPES).add(cls)


class Routes:
    def __init__(self, model, operator):
        self.modules, self.owners = {}, {}
        self.fallback_layer_names, self.fallback_parameter_names = [], []
        self.fallback_parameter_ids = set()
        self.fallback_params = []
        self.broadcast_embeddings = set()
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.trainable = {id(p) for p in self.params}
        covered = set()
        subtree = []
        all_modules = dict(model.named_modules())
        for name, m in all_modules.items():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                raise NotImplementedError('unsupported-dangerous: BatchNorm couples examples / mutates state')
            if isinstance(m, nn.Embedding) and (m.max_norm is not None or m.scale_grad_by_freq):
                raise NotImplementedError('unsupported-dangerous: Embedding max_norm/scale_grad_by_freq')
            if isinstance(m, BertEmbeddings):
                self.broadcast_embeddings.add(id(m.position_embeddings))
            if isinstance(m, GPT2Model):
                self.broadcast_embeddings.add(id(m.wpe))
            if any(name == prefix or name.startswith(prefix+'.') or prefix == '' for prefix in subtree):
                continue
            direct = {n: p for n, p in m.named_parameters(recurse=False, remove_duplicate=False)
                      if id(p) in self.trainable}
            if type(m) in FALLBACK_MODULE_TYPES:
                ps = {n: p for n, p in m.named_parameters(remove_duplicate=False)
                      if id(p) in self.trainable}
                if ps:
                    self.fallback_layer_names.append(name)
                    self.fallback_parameter_names.extend((name+'.' if name else '')+n for n in ps)
                    covered.update(id(p) for p in ps.values())
                    self.fallback_parameter_ids.update(id(p) for p in ps.values())
                    subtree.append(name)
                continue
            if not direct:
                continue
            if type(m) in FALLBACK_DIRECT_TYPES:
                self.fallback_layer_names.append(name)
                self.fallback_parameter_names.extend((name+'.' if name else '')+n for n in direct)
                covered.update(id(p) for p in direct.values())
                self.fallback_parameter_ids.update(id(p) for p in direct.values())
                continue
            adapter = kind(m)
            allowed = {'weight'} if adapter == 'embedding' else {'weight', 'bias'}
            if direct.keys()-allowed:
                raise NotImplementedError(f'Unsupported trainable parameters in {name}: {direct.keys()-allowed}')
            self.modules[name] = m
            for p in direct.values():
                self.owners.setdefault(id(p), []).append(name)
                covered.add(id(p))
        if covered != self.trainable:
            raise NotImplementedError('Unsupported trainable parameters were not assigned a route')
        selected = set() if operator is None else set(operator.data)
        for name in selected:
            m = all_modules[name]
            if kind(m) != 'linear':
                raise NotImplementedError(f'A geometry only applies to affine BK modules: {name}')
            if not all(p.requires_grad for p in m.parameters(recurse=False)):
                raise NotImplementedError(f'Partial freezing within preconditioned layer: {name}')
        # One shared Parameter cannot use two different augmented module maps.
        all_owners = {}
        for n, m in all_modules.items():
            for p in {id(p): p for p in m.parameters(recurse=False)}.values():
                if id(p) in self.trainable:
                    all_owners.setdefault(id(p), []).append(n)
        shared_names = {n for names in all_owners.values() if len(names) > 1 for n in names}
        if selected & shared_names:
            raise NotImplementedError('Shared weights require identity geometry; A may cover other layers')
        self.preconditioned_layers = sorted(selected)
        self.identity_geometry_layers = [n for n, m in all_modules.items()
            if n not in selected and any(id(p) in self.trainable for p in m.parameters(recurse=False))]
        self.fallback_params = [p for p in self.params if id(p) in self.fallback_parameter_ids]

    @staticmethod
    def parameter_names(model, parameter_ids):
        """Display metadata, retaining aliases while never using them for routing."""
        return [n for n, p in model.named_parameters(remove_duplicate=False)
                if id(p) in parameter_ids]

    def metadata(self, second=False):
        count = len(self.fallback_layer_names)
        return dict(fallback_layers=count, fallback_layer_count=count,
            fallback_layer_names=list(self.fallback_layer_names),
            fallback_parameter_names=list(self.fallback_parameter_names),
            fallback_parameter_ids=sorted(self.fallback_parameter_ids),
            requires_second_backward=bool(count) or second,
            preconditioned_layers=self.preconditioned_layers,
            identity_geometry_layers=self.identity_geometry_layers,
            gd_anchor_module=None, gd_anchor_modules=[],
            gd_applied=False, input_gradient_computed=False)


# RTX 3080 Ti synthetic sweep calibration, expressed in equivalent MACs.
# Fast writes/reads the sample matrix for reduction; Ghost has skinny GEMMs and
# one Python launch group per row tile. These are deliberately simple constants,
# not a fitted lookup table. Re-run benchmark_router.py on a different device.
FAST_REDUCTION_EQUIVALENT_TOKENS = 8
GHOST_COMPUTE_FACTOR = 8
LAUNCH_EQUIVALENT_MACS = 440_000_000


def norm_route(record, strategy, max_fast_temp_bytes, tile=64):
    """Choose full affine Fast, chunked affine Fast, or row-tiled Ghost."""
    if strategy not in ('auto', 'ghost', 'fast'):
        raise ValueError(f'Invalid norm strategy: {strategy}')
    b = len(record.b)
    if record.kind != 'linear':
        choice = 'fast_sparse' if record.kind == 'embedding' else 'fast'
        return dict(strategy=choice, estimated_fast_bytes=0,
                    estimated_full_fast_bytes=0, estimated_chunked_fast_bytes=0,
                    output_chunk_size=0, candidate_output_chunk_size=0,
                    estimated_full_fast_cost=0, estimated_chunked_fast_cost=0,
                    estimated_fast_cost=0, estimated_ghost_cost=0,
                    routing_reason='fast_sparse' if choice == 'fast_sparse' else 'fast_norm')
    t, d, o = record.b.shape[1], record.affine_z().shape[-1], record.b.shape[-1]
    element_size = record.b.element_size()
    full_bytes = b*d*o*element_size
    chunk_base_bytes = b*d*element_size
    full_feasible = full_bytes <= max_fast_temp_bytes
    max_chunk = max_fast_temp_bytes//chunk_base_bytes if chunk_base_bytes else 0
    chunk = min(o, max_chunk)
    chunk_feasible = chunk >= 1
    chunk_bytes = b*d*chunk*element_size if chunk_feasible else 0
    fast_macs, ghost_macs = b*t*d*o, b*t*t*(d+o)
    rows = min(tile, t)
    tiles = (t+rows-1)//rows
    full_cost = fast_macs + b*d*o*FAST_REDUCTION_EQUIVALENT_TOKENS + LAUNCH_EQUIVALENT_MACS
    chunk_cost = (fast_macs + b*d*o*FAST_REDUCTION_EQUIVALENT_TOKENS
                  + (o+chunk-1)//chunk*LAUNCH_EQUIVALENT_MACS) if chunk_feasible else None
    ghost_cost = GHOST_COMPUTE_FACTOR*ghost_macs + tiles*LAUNCH_EQUIVALENT_MACS
    if strategy == 'ghost':
        choice, reason = 'ghost', 'ghost_only_feasible'
    elif strategy == 'fast':
        if full_feasible:
            choice, reason = 'full_fast', 'fast_compute'
        elif chunk_feasible:
            choice, reason = 'chunked_fast', 'chunked_fast_compute'
        else:
            raise RuntimeError('Forced Fast strategy exceeds the affine workspace cap')
    else:
        candidates = [('ghost', ghost_cost)]
        if full_feasible:
            candidates.append(('full_fast', full_cost))
        if chunk_feasible:
            candidates.append(('chunked_fast', chunk_cost))
        choice = min(candidates, key=lambda item: item[1])[0]
        reason = ({'full_fast': 'fast_compute', 'chunked_fast': 'chunked_fast_compute',
                   'ghost': 'ghost_compute'}[choice]
                  if choice != 'ghost' or full_feasible or chunk_feasible
                  else 'ghost_only_feasible')
    return dict(strategy=choice,
                estimated_fast_bytes=full_bytes,
                estimated_full_fast_bytes=full_bytes,
                estimated_chunked_fast_bytes=chunk_bytes,
                output_chunk_size=chunk if choice == 'chunked_fast' else 0,
                candidate_output_chunk_size=chunk if chunk_feasible else 0,
                estimated_fast_cost=full_cost,
                estimated_full_fast_cost=full_cost,
                estimated_chunked_fast_cost=chunk_cost,
                estimated_ghost_cost=ghost_cost,
                estimated_fast_macs=fast_macs,
                estimated_ghost_macs=ghost_macs,
                routing_reason=reason)

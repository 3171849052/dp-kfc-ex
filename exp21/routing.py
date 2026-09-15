"""Explicit BK / whole-step FGC / unsupported routing, never name heuristics."""
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
            direct = {n: p for n, p in m.named_parameters(recurse=False) if id(p) in self.trainable}
            if type(m) in FALLBACK_MODULE_TYPES:
                ps = {n: p for n, p in m.named_parameters() if id(p) in self.trainable}
                if ps:
                    self.fallback_layer_names.append(name)
                    self.fallback_parameter_names.extend((name+'.' if name else '')+n for n in ps)
                    covered.update(id(p) for p in ps.values())
                    subtree.append(name)
                continue
            if not direct:
                continue
            if type(m) in FALLBACK_DIRECT_TYPES:
                self.fallback_layer_names.append(name)
                self.fallback_parameter_names.extend((name+'.' if name else '')+n for n in direct)
                covered.update(id(p) for p in direct.values())
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
            for p in m.parameters(recurse=False):
                if id(p) in self.trainable:
                    all_owners.setdefault(id(p), []).append(n)
        shared_names = {n for names in all_owners.values() if len(names) > 1 for n in names}
        if selected & shared_names:
            raise NotImplementedError('Shared weights require identity geometry; A may cover other layers')
        self.preconditioned_layers = sorted(selected)
        self.identity_geometry_layers = [n for n, m in all_modules.items()
            if n not in selected and any(id(p) in self.trainable for p in m.parameters(recurse=False))]

    def metadata(self, second=False):
        count = len(self.fallback_layer_names)
        return dict(fallback_layers=count, fallback_layer_count=count,
            fallback_layer_names=list(self.fallback_layer_names),
            fallback_parameter_names=list(self.fallback_parameter_names),
            requires_second_backward=bool(count) or second,
            preconditioned_layers=self.preconditioned_layers,
            identity_geometry_layers=self.identity_geometry_layers,
            gd_anchor_module=None, gd_anchor_modules=[],
            gd_applied=False, input_gradient_computed=False)

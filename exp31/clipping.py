"""Exp22 BK engine with frozen modules excluded from hooks and aggregates."""
from torch import nn
from exp22.methods import Clipper as BaseClipper

class Clipper(BaseClipper):
    def __init__(
        self,
        model: nn.Module,
        operator=None,
        method: str = "bk",
        max_grad_norm: float = 1.0,
    ):
        if method not in ("bk", "ghost", "bk_gd", "exact", "dp_sgd", "dp_kfc_a_bk", "dp_kfc"):
            raise ValueError(method)
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.model = model
        self.operator = operator
        self.method = method
        self.max_grad_norm = max_grad_norm
        self.linear_modules = dict((n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and m.weight.requires_grad)
        self.norm_modules = dict((n, m) for n, m in model.named_modules() if isinstance(m, nn.LayerNorm) and m.weight.requires_grad)
        self.pos_embed = None
        self.cls_token = None
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.parameter_ids = {id(p) for p in self.parameters}
        handled = {id(p) for m in self.linear_modules.values() for p in m.parameters(recurse=False)}
        handled.update(id(p) for m in self.norm_modules.values() for p in m.parameters(recurse=False))
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            handled.add(id(self.pos_embed))
        if self.cls_token is not None and self.cls_token.requires_grad:
            handled.add(id(self.cls_token))
        if handled != self.parameter_ids:
            missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in handled]
            raise NotImplementedError(f"unsupported trainable parameters: {missing}")
        selected = set() if operator is None else set(operator.data)
        if not selected <= set(self.linear_modules):
            raise ValueError(f"operator layers not in model: {sorted(selected - set(self.linear_modules))}")
        self.preconditioned_layers = sorted(selected)
        self.identity_geometry_layers = sorted(set(self.linear_modules) - selected)
        self.identity_geometry_layers += sorted(self.norm_modules)
        if self.pos_embed is not None and self.pos_embed.requires_grad:
            self.identity_geometry_layers.append("pos_embed")
        if self.cls_token is not None and self.cls_token.requires_grad:
            self.identity_geometry_layers.append("cls_token")
        self.identity_geometry_layers.sort()
        self.pending = []
        self.records = []
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.noise_events = 0
        self._last_stats = {}


"""Explicit layer adapters. Non-affine primitives use identity preconditioning."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.pytorch_utils import Conv1D as HFConv1D
from transformers.models.llama.modeling_llama import LlamaRMSNorm

LINEAR_LAYOUTS = {HFConv1D}
RMS_LAYOUTS = {nn.RMSNorm: 'eps', LlamaRMSNorm: 'variance_epsilon'}


def register_conv1d(cls):
    """Register an HF Conv1D-compatible [input, output] weight layout."""
    LINEAR_LAYOUTS.add(cls)


def register_rmsnorm(cls, eps_attribute='eps'):
    RMS_LAYOUTS[cls] = eps_attribute


def kind(module):
    if isinstance(module, nn.Linear) or type(module) in LINEAR_LAYOUTS:
        return 'linear'
    if isinstance(module, nn.Conv2d):
        if module.groups != 1 or module.padding_mode != 'zeros':
            raise NotImplementedError('Conv2d requires groups=1 and zero padding')
        return 'linear'
    if isinstance(module, nn.Embedding):
        if module.scale_grad_by_freq or module.max_norm is not None:
            raise NotImplementedError('Embedding scale_grad_by_freq/max_norm are unsupported')
        return 'embedding'
    if isinstance(module, nn.LayerNorm) or type(module) in RMS_LAYOUTS:
        return 'norm'
    raise NotImplementedError(f'Unsupported trainable module: {type(module).__name__}')


def nbytes(t):
    if t.is_sparse:
        t = t.coalesce()
        return nbytes(t.indices()) + nbytes(t.values())
    return t.numel()*t.element_size()


def retained_bytes(records):
    """Actual distinct storage retained by records, including view bases."""
    storages = {}
    for record in records:
        for tensor in (record.x, record.z, record.b):
            if tensor is not None:
                storage = tensor.untyped_storage()
                storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class Record:
    def __init__(self, name, module, activation, backprop, operator, trainable):
        self.name, self.module = name, module
        self.kind = kind(module)
        self.params = {n: p for n, p in module.named_parameters(recurse=False) if id(p) in trainable}
        self.x, self.b = None, backprop
        self.z = None
        if self.kind == 'linear':
            if isinstance(module, nn.Conv2d):
                a = F.unfold(activation, module.kernel_size, dilation=module.dilation,
                             padding=module.padding, stride=module.stride).transpose(1, 2)
                self.b = backprop.flatten(2).transpose(1, 2)
            else:
                a = activation.reshape(len(activation), -1, activation.shape[-1])
                self.b = backprop.reshape(len(backprop), -1, backprop.shape[-1])
            if module.bias is not None:
                a = torch.cat((a, torch.ones_like(a[..., :1])), -1)
            self.z = (operator.transform_activation(name, a.transpose(1, 2)).transpose(1, 2)
                      if operator is not None and name in operator.data else a)
        elif self.kind == 'norm':
            dims = tuple(range(-module.weight.ndim, 0))
            if isinstance(module, nn.LayerNorm):
                mean = activation.mean(dims, keepdim=True)
                var = activation.var(dims, unbiased=False, keepdim=True)
                self.z = (activation-mean)*torch.rsqrt(var+module.eps)
            else:
                eps = getattr(module, RMS_LAYOUTS[type(module)])
                if eps is None:
                    eps = torch.finfo(activation.dtype).eps
                self.z = activation*torch.rsqrt(activation.square().mean(dims, keepdim=True)+eps)
        else:
            self.x = activation

    def bytes(self):
        return sum(nbytes(t) for t in (self.x, self.b, self.z) if t is not None)

    def split(self, g):
        m = self.module
        weight = g[..., :-1] if m.bias is not None else g
        if type(m) in LINEAR_LAYOUTS:
            weight = weight.transpose(-1, -2)
        result = {}
        if 'weight' in self.params:
            result[self.params['weight']] = weight.reshape(*g.shape[:-2], *m.weight.shape)
        if 'bias' in self.params:
            result[self.params['bias']] = g[..., -1]
        return result

    def sample(self, i):
        """One sample only; embedding remains sparse even for huge vocabularies."""
        if self.kind == 'linear':
            return self.split(self.b[i].T @ self.z[i])
        if self.kind == 'embedding':
            ids, b = self.x[i].reshape(-1), self.b[i].reshape(-1, self.b.shape[-1])
            if self.module.padding_idx is not None:
                mask = ids != self.module.padding_idx
                ids, b = ids[mask], b[mask]
            g = torch.sparse_coo_tensor(ids[None], b, self.module.weight.shape, check_invariants=False).coalesce()
            return {self.params['weight']: g}
        m = self.module
        dims = tuple(range(self.b[i].ndim-m.weight.ndim))
        reduce = lambda t: t.sum(dims) if dims else t
        result = {}
        if 'weight' in self.params:
            result[self.params['weight']] = reduce(self.b[i]*self.z[i])
        if 'bias' in self.params:
            result[self.params['bias']] = reduce(self.b[i])
        return result

    def fast(self):
        if self.kind == 'linear':
            return self.split(self.b.transpose(1, 2) @ self.z)
        if self.kind == 'norm':
            dims = tuple(range(1, self.b.ndim-self.module.weight.ndim))
            reduce = lambda t: t.sum(dims) if dims else t
            result = {}
            if 'weight' in self.params:
                result[self.params['weight']] = reduce(self.b*self.z)
            if 'bias' in self.params:
                result[self.params['bias']] = reduce(self.b)
            return result
        raise ValueError('Embedding uses sparse sample aggregation, never [B,V,d]')

    def ghost(self, tile):
        total = self.b.new_zeros(len(self.b))
        # Gram identity covers both trainable weight and augmented bias. Frozen
        # affine subsets use Fast, since P may mix the augmented coordinates.
        for j in range(0, self.b.shape[1], tile):
            bj, zj = self.b[:, j:j+tile], self.z[:, j:j+tile]
            for k in range(0, self.b.shape[1], tile):
                bg = bj @ self.b[:, k:k+tile].transpose(1, 2)
                zg = zj @ self.z[:, k:k+tile].transpose(1, 2)
                total.add_((bg*zg).sum((1, 2)))
        return total.clamp_min(0)

    def aggregate(self, factors):
        if self.kind == 'linear':
            b = (self.b*factors[:, None, None]).reshape(-1, self.b.shape[-1])
            return self.split(b.T @ self.z.reshape(-1, self.z.shape[-1]))
        if self.kind == 'embedding':
            ids = self.x.reshape(-1)
            b = (self.b*factors.reshape(-1, *([1]*(self.b.ndim-1)))).reshape(-1, self.b.shape[-1])
            if self.module.padding_idx is not None:
                mask = ids != self.module.padding_idx
                ids, b = ids[mask], b[mask]
            g = torch.zeros_like(self.module.weight)
            g.index_add_(0, ids, b)
            return {self.params['weight']: g}
        return {p: torch.einsum('b,bp->p', factors, g.flatten(1)).reshape_as(p)
                for p, g in self.fast().items()}

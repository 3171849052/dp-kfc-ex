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


def sample_norm_squared(grads):
    # Reduce strided weight views directly: flattening an augmented weight view
    # copies B*O*D elements, and square().sum() creates another full-sized tensor.
    return sum(torch.linalg.vector_norm(g, dim=tuple(range(1, g.ndim))).square()
               for g in grads.values())


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
        # Gram identity covers trainable weight and augmented bias coordinates.
        # Exclude frozen coordinates after A transformation, just as split does.
        z = self.z
        if 'weight' not in self.params:
            z = z[..., -1:]
        elif self.module.bias is not None and 'bias' not in self.params:
            z = z[..., :-1]
        bt, zt = self.b.transpose(1, 2), z.transpose(1, 2)
        # Even tile >= T must not allocate a full [B,T,T] Gram (except T=1).
        rows = min(tile, max(1, self.b.shape[1]-1))
        for j in range(0, self.b.shape[1], rows):
            bg = self.b[:, j:j+rows] @ bt
            zg = z[:, j:j+rows] @ zt
            bg.mul_(zg)
            total.add_(bg.sum((1, 2)))
            del bg, zg
        return total.clamp_min(0)

    def embedding_rows(self):
        """Merge repeated (example, token) rows without dense vocabulary storage."""
        batch = len(self.b)
        ids = self.x.reshape(batch, -1)
        samples = torch.arange(batch, device=ids.device)[:, None].expand_as(ids)
        keys = (samples*self.module.num_embeddings+ids).reshape(-1)
        values = self.b.reshape(-1, self.b.shape[-1])
        if self.module.padding_idx is not None:
            mask = ids.reshape(-1) != self.module.padding_idx
            keys, values = keys[mask], values[mask]
        unique, inverse = torch.unique(keys, return_inverse=True)
        merged = values.new_zeros((unique.numel(), values.shape[-1]))
        merged.index_add_(0, inverse, values)
        return unique//self.module.num_embeddings, unique % self.module.num_embeddings, merged

    def embedding_norm(self):
        samples, ids, values = self.embedding_rows()
        total = self.b.new_zeros(len(self.b))
        total.index_add_(0, samples, values.square().sum(-1))
        return total

    def chunked_norm(self, chunk):
        total = self.b.new_zeros(len(self.b))
        for v in range(0, self.b.shape[-1], chunk):
            g = self.b[..., v:v+chunk].transpose(1, 2) @ self.z
            total.add_(g.square().sum((1, 2)))
            del g
        return total

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
        weighted = self.b*factors.reshape(-1, *([1]*(self.b.ndim-1)))
        dims = tuple(range(weighted.ndim-self.module.weight.ndim))
        result = {}
        if 'weight' in self.params:
            result[self.params['weight']] = (weighted*self.z).sum(dims)
        if 'bias' in self.params:
            result[self.params['bias']] = weighted.sum(dims)
        return result

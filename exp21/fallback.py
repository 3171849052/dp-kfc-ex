"""Whole-step Fast Clipping: one batched VJP, then weighted backward.

This explicit fallback is sample-independent FP32, using PyTorch math attention.
It retains full per-sample gradients during the first pass, so it is a correctness
route for packed/custom parameters, not a scalable replacement for BK embeddings.
"""
from contextlib import contextmanager
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from exp21.profiling import timed, PHASES
from exp21.handlers import nbytes, LINEAR_LAYOUTS


def rng_state(device):
    return torch.random.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None


@contextmanager
def replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(state[0])
        if devices:
            torch.cuda.set_rng_state(state[1], device)
        yield


@torch.no_grad()
def transform_samples(model, operator, grads):
    if operator is None:
        return
    for name, m in model.named_modules():
        if name not in operator.data:
            continue
        w = grads[m.weight].transpose(-1, -2) if type(m) in LINEAR_LAYOUTS else grads[m.weight].flatten(2)
        g = torch.cat((w, grads[m.bias].unsqueeze(-1)), -1) if m.bias is not None else w
        g = operator.transform_matrix(name, g)
        w = g[..., :-1] if m.bias is not None else g
        if type(m) in LINEAR_LAYOUTS:
            w = w.transpose(-1, -2)
        grads[m.weight] = w.reshape_as(grads[m.weight]).contiguous()
        if m.bias is not None:
            grads[m.bias] = g[..., -1].contiguous()


def whole_step(model, routes, operator, x, y, loss_fn, bound, profiler):
    from exp21.bk import transform_aggregate
    stats = {key: 0. for key in PHASES}
    stats.update(routes.metadata(second=True))
    stats.update(bk_cache_bytes=0, temporary_per_sample_grad_bytes=0,
        bk_ghost_layers=[], bk_fast_layers=[], ghost_layer_count=0, fast_layer_count=0,
        layer_strategies={n: 'whole_step_fgc' for n in routes.identity_geometry_layers+routes.preconditioned_layers},
        backward_calls=0, first_pass_parameter_grad_count=0, first_pass_reverse_vectors=len(x))
    model.zero_grad(set_to_none=True)
    state = rng_state(x.device)
    with sdpa_kernel(SDPBackend.MATH):
        with timed(profiler, 'first_pass_seconds', x.device):
            losses = loss_fn(model(x.detach()), y)
            if losses.shape != (len(x),):
                raise ValueError('loss_fn must return one scalar per example')
            values = torch.autograd.grad(losses, routes.params,
                grad_outputs=torch.eye(len(x), device=x.device, dtype=losses.dtype), is_grads_batched=True)
            stats['backward_calls'] += 1
            stats['first_pass_parameter_grad_count'] = sum(p.grad is not None for p in routes.params)
        with timed(profiler, 'norm_seconds', x.device), torch.no_grad():
            grads = dict(zip(routes.params, values))
            del values
            stats['temporary_per_sample_grad_bytes'] = sum(nbytes(g) for g in grads.values())
            transform_samples(model, operator, grads)
            norms = sum(g.flatten(1).square().sum(1) for g in grads.values()).clamp_min(0).sqrt()
            factors = (bound/(norms+1e-6)).clamp(max=1).detach()
            del grads
        with timed(profiler, 'second_pass_seconds', x.device), replay_rng(state, x.device):
            model.zero_grad(set_to_none=True)
            (loss_fn(model(x.detach()), y)*factors).sum().backward()
            stats['backward_calls'] += 1
        with timed(profiler, 'aggregate_transform_seconds', x.device):
            transform_aggregate(model, operator)
    return losses.detach().sum(), norms, factors, {}, stats

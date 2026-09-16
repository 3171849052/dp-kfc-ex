"""Bounded local Fast VJP norm with exact RNG replay for reconstruction."""
from contextlib import contextmanager
import torch
from exp21.handlers import nbytes, LINEAR_LAYOUTS, sample_norm_squared


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
        if name not in operator.data or m.weight not in grads:
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


def chunked_norms(losses, params, model, operator, chunk_size, stats):
    total = torch.zeros_like(losses)
    for start in range(0, len(losses), chunk_size):
        stop = min(start+chunk_size, len(losses))
        vectors = losses.new_zeros((stop-start, len(losses)))
        vectors[:, start:stop] = torch.eye(stop-start, device=losses.device, dtype=losses.dtype)
        values = torch.autograd.grad(losses, params, grad_outputs=vectors,
                                    is_grads_batched=True, retain_graph=stop < len(losses))
        stats['backward_calls'] += 1
        with torch.no_grad():
            grads = dict(zip(params, values))
            del values
            size = sum(nbytes(g) for g in grads.values())
            stats['fallback_temporary_grad_bytes'] = max(stats['fallback_temporary_grad_bytes'], size)
            stats['temporary_per_sample_grad_bytes'] = max(stats['temporary_per_sample_grad_bytes'], size)
            transform_samples(model, operator, grads)
            total[start:stop] = sample_norm_squared(grads)
            del grads, vectors
    return total

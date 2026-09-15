"""Forward-only A geometry and Exp19 Structured Ghost production path."""
import math
import torch
from exp12.curvature import layers, activation_sum
from exp13.operator import Operator as FullOperator
from dp_kfac.optimizer import generate_pink_noise
from exp19.methods import GhostNorm, ghost_aggregate, activation_forward_only, noise_and_normalize, noise_and_step
from exp20.config import DAMPING, POWERS
from exp20.profiling import timed

class AOperator(FullOperator):
    def __init__(self, factors, power):
        self.data = {}
        spectra, gains = [], []
        raw = torch.zeros((), device=next(iter(factors.values()))['A'].device, dtype=torch.float64)
        ref = torch.zeros_like(raw)
        for n, f in factors.items():
            a = f['A'].double()
            e, q = torch.linalg.eigh((a+a.T)/2)
            e = e.clamp_min(0)
            raw.add_(f['output_dimension'] * (e*(e+DAMPING).pow(-2*power)).sum())
            ref.add_(f['output_dimension'] * (e*(e+DAMPING).pow(-1)).sum())
            self.data[n] = (torch.eye(len(e), device=e.device) if power == 0 else ((q*(e+DAMPING).pow(-power))@q.T).float())
            spectra.append(e*(e+DAMPING).pow(-2*power))
            gains.append((e+DAMPING).pow(-power))
        raw, ref = torch.stack((raw, ref)).cpu().tolist()
        self.scale = 1. if power == .5 else math.sqrt(ref/raw)
        self.moments = dict(m_p=raw, m_reference=ref, scale_match=self.scale)

        assert math.isclose(self.scale**2*raw, ref, rel_tol=1e-12)
        eig = torch.cat(spectra)*self.scale**2
        gain = torch.cat(gains)*self.scale
        quantiles = torch.tensor([.1, .5, .9, .99], device=eig.device, dtype=eig.dtype)
        values = torch.cat((gain.quantile(quantiles), eig.quantile(quantiles))).cpu().tolist()
        self.diagnostics = dict(zip([f'{prefix}_p{p}' for prefix in ('operator_gain', 'transformed_eig') for p in (10, 50, 90, 99)], values))
        # p90/p10 over every augmented activation eigenvalue, including zeros.
        self.diagnostics['transformed_condition_proxy'] = values[6]/values[4] if values[4] > 0 else float('inf')

    def transform_activation(self, name, a):
        return self.scale * (self.data[name] @ a)

    def transform_backprop(self, name, b):
        return b

    def transform_matrix(self, name, g):
        return self.scale * (g @ self.data[name])


@torch.no_grad()
def build_from_cache(model, power, cache, seed, epoch, profiler=None):
    assert power in POWERS
    device = cache[0].device
    stats = dict(builder_forward_calls=0, builder_vjp_calls=0,
                 builder_reverse_vectors=0, builder_samples=sum(map(len, cache)),
                 curvature_backward_seconds=0.)
    factors, counts = {}, {}
    for x in cache:
        with timed(profiler, 'activation_forward_seconds', device):
            acts = activation_forward_only(model, x)
        stats['builder_forward_calls'] += 1
        with timed(profiler, 'factor_accumulation_seconds', device):
            for n, module in layers(model).items():
                a, count = activation_sum(acts[n], module)
                if n not in factors:
                    factors[n] = dict(A=torch.zeros_like(a), output_dimension=module.weight.shape[0])
                    counts[n] = 0
                factors[n]['A'].add_(a)
                counts[n] += count
        del acts, a
    with timed(profiler, 'factor_accumulation_seconds', device):
        for n, f in factors.items():
            f['A'].div_(counts[n])
    with timed(profiler, 'matrix_function_seconds', device):
        operator = AOperator(factors, power)
    stats.update(operator.moments)
    stats.update(operator.diagnostics)
    stats['operator_state_bytes'] = sum(t.numel()*t.element_size() for t in operator.data.values())+8
    return operator, stats


def synthetic_cache(seed, epoch, device, batches, batch_size):
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.random.default_generator.manual_seed(seed+10000+epoch)
        torch.cuda.default_generators[index].manual_seed(seed+10000+epoch)
        return [generate_pink_noise(batch_size, (1, 28, 28), device) for _ in range(batches)]



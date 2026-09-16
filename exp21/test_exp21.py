"""Independent sample-backward golden references plus real CUDA/MNIST checks."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(ROOT/'exp21/.cache/hf')
import copy
import json
import weakref
import pytest
import torch
from torch import nn
from exp12.runtime import runtime
from exp21.bk import BookKeeping, transform_aggregate
from exp21.handlers import Record, register_conv1d, register_rmsnorm
from exp21.methods import Clipper, build_from_cache, synthetic_cache, noise_and_step
from exp21.run_one import initialize, load_data
from exp21.config import METHODS


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(4)
    torch.manual_seed(21)
    with runtime('cuda:0'):
        yield


def loss(output, target):
    return (output-target).square().reshape(len(output), -1).sum(1)/2


def golden(model, x, y, op=None, loss_fn=loss, max_grad_norm=1.):
    sums = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    norms, factors = [], []
    for xx, yy in zip(x, y):
        model.zero_grad(set_to_none=True)
        loss_fn(model(xx[None]), yy[None]).sum().backward()
        transform_aggregate(model, op)
        norm = sum(p.grad.square().sum() for p in model.parameters() if p.requires_grad).sqrt()
        factor = (max_grad_norm/(norm+1e-6)).clamp(max=1)
        norms.append(norm)
        factors.append(factor)
        for n, p in model.named_parameters():
            if p.requires_grad:
                sums[n].add_(p.grad*factor)
    return torch.stack(norms), torch.stack(factors), sums


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d))
        self.eps = 1e-6

    def forward(self, x):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps)*self.weight


class Conv1D(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d, o))
        self.bias = nn.Parameter(torch.randn(o))

    def forward(self, x):
        return x @ self.weight + self.bias


register_rmsnorm(RMS)
register_conv1d(Conv1D)


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, 4, padding_idx=0)
        self.out = nn.Linear(4, 9, bias=False)
        self.out.weight = self.emb.weight

    def forward(self, x):
        return self.out(self.emb(x))


class Reused(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.a(x)+self.b(2*x)+self.a(x.sin())


def case(name):
    if name == 'linear2':
        return nn.Linear(4, 5), torch.randn(3, 4)
    if name == 'linear3':
        return nn.Linear(4, 5), torch.randn(3, 7, 4)
    if name == 'linear4':
        return nn.Linear(4, 5, bias=False), torch.randn(3, 2, 7, 4)
    if name == 'conv':
        return nn.Conv2d(2, 3, 3, padding=1, stride=2), torch.randn(3, 2, 9, 9)
    if name == 'embedding':
        return nn.Embedding(100003, 4, padding_idx=0), torch.tensor([[0, 2, 2, 8], [3, 3, 3, 3], [5, 4, 5, 4]])
    if name == 'layernorm':
        return nn.LayerNorm((2, 4)), torch.randn(3, 5, 2, 4)
    if name == 'rms':
        return RMS(4), torch.randn(3, 7, 4)
    if name == 'torch_rms':
        return nn.RMSNorm(4), torch.randn(3, 7, 4)
    if name == 'conv1d':
        return Conv1D(4, 5), torch.randn(3, 7, 4)
    if name == 'tied':
        return Tied(), torch.tensor([[0, 2, 2, 8], [3, 3, 3, 3], [5, 4, 5, 4]])
    if name == 'reused':
        return Reused(), torch.randn(3, 2, 4)
    raise ValueError(name)


@pytest.mark.parametrize('name', ['linear2', 'linear3', 'linear4', 'conv', 'embedding', 'layernorm', 'rms', 'torch_rms', 'conv1d', 'tied', 'reused'])
@pytest.mark.parametrize('method,strategy', [('bk','ghost'), ('bk','fast'), ('bk','auto'), ('bk_gd','auto'), ('fast2','fast'), ('ghost2','ghost')])
def test_primitive(name, method, strategy):
    model, x = case(name)
    model, x = model.cuda(), x.cuda()
    y = torch.randn_like(model(x))
    ref = copy.deepcopy(model)
    norms, factors, grads = golden(ref, x, y)
    hooks = BookKeeping(model, strategy=strategy, tile=3)
    touched = []
    handles = [p.register_hook(lambda g: touched.append(True)) for p in model.parameters()]
    _, actual, clips, _, stats = hooks.aggregate(x, y, method, loss_fn=loss)
    if method == 'bk_gd':
        assert touched == [] and stats['first_pass_parameter_grad_count'] == 0
    torch.testing.assert_close(actual, norms, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(clips, factors, rtol=1e-4, atol=2e-5)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[n], rtol=1e-4, atol=3e-6)
        assert 'grad_sample' not in p.__dict__
    assert stats['backward_calls'] == (2 if method.endswith('2') else 1)
    assert stats['fallback_layer_count'] == 0
    assert stats['fallback_layers'] == 0 and stats['fallback_layer_names'] == []
    if method in ('bk', 'bk_gd'):
        assert not stats['requires_second_backward'] and stats['second_pass_seconds'] == 0
    assert not hooks.pending and not hooks.records
    old = {n: p.detach().clone() for n, p in model.named_parameters()}
    noise_and_step(model, torch.optim.SGD(model.parameters(), lr=.5), 0, len(x), torch.Generator(device='cuda').manual_seed(7))
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, old[n]-.5*grads[n]/len(x), rtol=1e-4, atol=3e-6)
    hooks.remove()
    for h in handles:
        h.remove()
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('method', ['bk', 'bk_gd', 'fast2', 'ghost2'])
def test_nonidentity_operator(method):
    model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).cuda()
    x, y = torch.randn(5, 4, device='cuda'), torch.randn(5, 3, device='cuda')
    op, _ = build_from_cache(model, .25, [x], 42, 1)
    norms, factors, grads = golden(copy.deepcopy(model), x, y, op)
    strategy = {'fast2': 'fast', 'ghost2': 'ghost'}.get(method, 'auto')
    hooks = BookKeeping(model, op, strategy)
    _, n, c, _, _ = hooks.aggregate(x, y, method, loss_fn=loss)
    torch.testing.assert_close(n, norms, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(c, factors, rtol=1e-4, atol=1e-5)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], rtol=1e-4, atol=2e-6)
    hooks.remove()


def test_ghost_never_materializes(monkeypatch):
    model = nn.Linear(4, 5).cuda()
    x = torch.randn(3, 8, 4, device='cuda')
    hooks = BookKeeping(model, strategy='ghost', tile=3)
    def forbidden(*args, **kwargs):
        pytest.fail('Ghost materialized per-example parameter gradient')
    monkeypatch.setattr(Record, 'fast', forbidden)
    monkeypatch.setattr(Record, 'sample', forbidden)
    _, _, _, _, stats = hooks.aggregate(x, torch.randn(3, 8, 5, device='cuda'), loss_fn=loss)
    assert stats['temporary_per_sample_grad_bytes'] == 0
    hooks.remove()


def test_cache_release_and_failure_restore():
    model = nn.Linear(4, 5).cuda()
    hooks = BookKeeping(model)
    x = torch.randn(3, 4, device='cuda')
    refs = []
    original = hooks.reconstruct
    def capture(c):
        refs.extend(weakref.ref(t) for r in hooks.records for t in (r.x, r.b, r.z) if t is not None)
        original(c)
    hooks.reconstruct = capture
    hooks.aggregate(x, torch.randn(3, 5, device='cuda'), 'bk_gd', loss_fn=loss)
    assert all(r() is None for r in refs)
    with pytest.raises(RuntimeError):
        hooks.aggregate(x, torch.randn(3, 8, device='cuda'), 'bk_gd', loss_fn=loss)
    assert all(p.requires_grad for p in model.parameters())
    assert not hooks.pending and not hooks.records and not hooks.enabled
    hooks.remove()


def test_unsupported():
    with pytest.raises(NotImplementedError, match='Unsupported trainable'):
        BookKeeping(nn.PReLU())
    with pytest.raises(NotImplementedError, match='groups'):
        BookKeeping(nn.Conv2d(4, 4, 3, groups=2))


def test_real_mnist_five_methods():
    train, _ = load_data()
    # A complete production-size batch, plus independent sample-backward oracle.
    x = torch.stack([train[i][0] for i in range(256)]).cuda()
    y = torch.tensor([train[i][1] for i in range(256)], device='cuda')
    base = initialize(42, 'cuda')
    op, _ = build_from_cache(base, .25, synthetic_cache(42, 1, 'cuda:0', 1, 256), 42, 1)
    ce = lambda z, target: torch.nn.functional.cross_entropy(z, target, reduction='none')
    norms, factors, grads = golden(copy.deepcopy(base), x, y, op, ce)
    errors = {}
    for method in METHODS:
        model = copy.deepcopy(base)
        clipper = Clipper(model, op, method)
        _, n, c, _, stats = clipper.aggregate(x, y)
        torch.testing.assert_close(n, norms, rtol=5e-4, atol=1e-4)
        torch.testing.assert_close(c, factors, rtol=5e-4, atol=1e-5)
        max_error = 0.
        delta_sq = norm_sq = 0.
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, grads[name], rtol=5e-4, atol=3e-5)
            delta = p.grad-grads[name]
            max_error = max(max_error, delta.abs().max().item())
            delta_sq += delta.square().sum().item()
            norm_sq += grads[name].square().sum().item()
        errors[method] = dict(gradient_max_abs_error=max_error,
            gradient_relative_l2_error=(delta_sq/norm_sq)**.5,
            norm_max_abs_error=(n-norms).abs().max().item(),
            clip_max_abs_error=(c-factors).abs().max().item())
        if method in ('bk', 'bk_gd'):
            assert stats['fallback_layer_count'] == 0 and not stats['requires_second_backward']
            assert stats['second_pass_seconds'] == 0
        clipper.remove()
    (ROOT/'exp21/results/correctness.json').write_text(json.dumps(errors, indent=2)+'\n')


def test_event_no_sync(monkeypatch):
    from exp21.profiling import EventProfiler
    model = nn.Linear(4, 5).cuda()
    hooks = BookKeeping(model)
    prof = EventProfiler()
    def forbidden(*a, **k):
        pytest.fail('Synchronize inside timed step')
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'synchronize', forbidden)
        hooks.aggregate(torch.randn(3, 4, device='cuda'), torch.randn(3, 5, device='cuda'), 'bk_gd', prof, loss)
    torch.cuda.synchronize()
    values = prof.resolve()
    assert values['bk_reconstruction_seconds'] > 0
    assert 'second_pass_seconds' not in values
    hooks.remove()


def test_real_huggingface_conv1d():
    from transformers.pytorch_utils import Conv1D as HFConv1D
    register_conv1d(HFConv1D)
    model = HFConv1D(5, 4).cuda()
    x, y = torch.randn(3, 7, 4, device='cuda'), torch.randn(3, 7, 5, device='cuda')
    norms, factors, grads = golden(copy.deepcopy(model), x, y)
    hooks = BookKeeping(model, strategy='ghost', tile=3)
    _, n, c, _, _ = hooks.aggregate(x, y, 'bk_gd', loss_fn=loss)
    torch.testing.assert_close(n, norms, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(c, factors, rtol=1e-4, atol=1e-5)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], rtol=1e-4, atol=3e-6)
    hooks.remove()


def test_embedding_workspace_scales_with_tokens():
    model, x = case('embedding')
    model, x = model.cuda(), x.cuda()
    hooks = BookKeeping(model)
    _, _, _, _, stats = hooks.aggregate(x, torch.zeros(3, 4, 4, device='cuda'), 'bk_gd', loss_fn=loss)
    # Vectorization merges all examples together; workspace scales with B*T*d.
    assert stats['temporary_per_sample_grad_bytes'] <= x.numel()*model.embedding_dim*4
    assert stats['bk_cache_bytes'] < 4096
    hooks.remove()


def test_auto_strategy_and_repeated_step_memory():
    model = initialize(42, 'cuda')
    x = torch.randn(256, 1, 28, 28, device='cuda')
    y = torch.arange(256, device='cuda') % 10
    op, _ = build_from_cache(model, .25, [x], 42, 1)
    hooks = BookKeeping(model, op)
    allocated = []
    for i in range(6):
        result = hooks.aggregate(x, y, 'bk_gd')
        stats = result[-1]
        assert stats['layer_strategies'] == dict(conv1='full_fast', conv2='full_fast', fc1='ghost', fc2='ghost')
        assert not hooks.records and not hooks.pending
        del result
        model.zero_grad(set_to_none=True)
        allocated.append(torch.cuda.memory_allocated())
    assert max(allocated[1:])-min(allocated[1:]) <= 4096
    hooks.remove()


def test_rng_and_protocol():
    from exp21.run_one import private_loader, warmup
    from exp21.config import SEEDS, METHOD_ORDER, FORMAL_RUN_COUNT
    from torch.utils.data import TensorDataset
    assert FORMAL_RUN_COUNT == 25
    assert all(set(METHOD_ORDER[s]) == set(METHODS) for s in SEEDS)
    assert all(len({METHOD_ORDER[s].index(m) for s in SEEDS}) == 5 for m in METHODS)
    first = initialize(42, 'cuda')
    for method in METHODS:
        model = initialize(42, 'cuda')
        assert all(torch.equal(a, b) for a, b in zip(first.parameters(), model.parameters()))
    data = TensorDataset(torch.randn(256, 1, 28, 28), torch.arange(256)%10)
    cpu, cuda = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    for method in METHODS:
        warmup(method, data, torch.device('cuda:0'))
        assert torch.equal(cpu, torch.random.get_rng_state())
        assert torch.equal(cuda, torch.cuda.get_rng_state())
    a = torch.cat(list(private_loader(torch.arange(1024), 42)))
    b = torch.cat(list(private_loader(torch.arange(1024), 42)))
    assert torch.equal(a, b)
    generator = torch.Generator(device='cuda').manual_seed(40042)
    noise_state = generator.get_state().clone()
    one = synthetic_cache(42, 1, 'cuda:0', 1, 4)
    two = synthetic_cache(42, 1, 'cuda:0', 1, 4)
    assert torch.equal(one[0], two[0])
    assert torch.equal(generator.get_state(), noise_state)
    assert torch.equal(cpu, torch.random.get_rng_state())
    assert torch.equal(cuda, torch.cuda.get_rng_state())


def test_extra_trainable_parameter_rejected():
    model = nn.Linear(3, 4)
    model.extra = nn.Parameter(torch.ones(4))
    with pytest.raises(NotImplementedError, match='Unsupported trainable parameters'):
        BookKeeping(model)


from exp21.test_transformer_cases import (
    test_corrected_nonidentity_geometry, test_partial_geometry_and_nonunit_bound,
    test_cnn_builder_matches_exp20, test_output_anchor_and_minimal_cache,
    test_tied_analytic_cross_term, test_tiny_transformer_integration,
    test_tinyvit_whole_step_fallback, test_dangerous_not_silent_fallback)


from exp21.test_optimized_cases import (test_row_ghost, test_memory_compute_router,
    test_local_fallback, test_streaming_baseline, test_vector_embedding,
    test_tied_workspace, test_shared_guard, test_optimized_repeated_memory,
    test_dropout_fallback_replay, test_frozen_affine_memory_cap,
    test_ghost_conv_cap, test_norm_allocation_shapes,
    test_shared_parameter_across_bk_fallback_boundary,
    test_forced_fast_uses_chunked_affine, test_chunked_affine_matches_fast,
    test_fallback_chunk_uses_memory_budget)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

"""Real operator, Opacus, Ghost, RNG and CUDA integration tests."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import os
import tempfile
TMP = ROOT/'exp19/.cache/tmp'
TMP.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TMP)
tempfile.tempdir = str(TMP)
import copy
import weakref
import pytest
import torch
from torch import nn
from exp12.curvature import estimate, layers
from exp12.runtime import runtime
from exp13.operator import Operator
from exp13.run_exp13 import initialize
from exp19 import methods as m
from exp19.config import METHODS
from exp19.run_one import load_data, private_loader
from exp19.profiling import phase_start, phase_end


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(4)
    with runtime('cuda:0'):
        yield


def tiny():
    torch.manual_seed(19)
    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3)).cuda()


@pytest.mark.parametrize('method', METHODS[:2])
def test_original_operator_regression(method):
    model = tiny()
    cache = [torch.randn(8, 4, device='cuda') for _ in range(2)]
    factors, budget = estimate(model, cache, 'KFAC-U', seed=42+20000+1)
    ref = Operator(factors)
    actual, stats = m.build_from_cache(model, method, cache, 42, 1)
    for n in ref.data:
        for a, b in zip(ref.data[n], actual.data[n]):
            torch.testing.assert_close(a, b)
    assert stats['builder_vjp_calls'] == budget['vjp_calls'] == 2
    assert stats['builder_reverse_vectors'] == 16


@pytest.mark.parametrize('method', METHODS[2:])
def test_forward_only_no_labels_no_reverse(method, monkeypatch):
    model = tiny()
    cache = [torch.randn(8, 4, device='cuda')]
    def forbidden(*args, **kwargs):
        pytest.fail('Forward-only builder attempted labels or reverse differentiation')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(m, 'uniform_labels', forbidden)
    monkeypatch.setattr(torch, 'multinomial', forbidden)
    op, stats = m.build_from_cache(model, method, cache, 42, 1)
    assert stats['builder_vjp_calls'] == stats['builder_reverse_vectors'] == 0
    assert stats['curvature_backward_seconds'] == 0
    assert stats['builder_forward_calls'] == 1 and stats['builder_samples'] == 8
    assert set(op.data) == set(layers(model))


def test_a_operator_power_and_scale():
    model = tiny()
    cache = [torch.randn(8, 4, device='cuda')]
    a, _ = m.build_from_cache(model, METHODS[2], cache, 42, 1)
    b, stats = m.build_from_cache(model, METHODS[3], cache, 42, 1)
    assert stats['scale_match']**2*stats['m_raw'] == pytest.approx(stats['m_reference'], rel=1e-12)
    assert a.scale == 1
    for n in a.data:
        torch.testing.assert_close(b.data[n]@b.data[n], a.data[n], atol=2e-5, rtol=2e-5)
        v = torch.randn(2, layers(model)[n].out_features, 1, device='cuda')
        assert b.transform_backprop(n, v) is v


def equivalence(model, x, y, op):
    other = copy.deepcopy(model)
    wrapper = m.GradSampleModule(model, loss_reduction='sum')
    hooks = m.GhostNorm(other, op)
    exact = m.exact_aggregate(wrapper, op, x, y)
    ghost = m.ghost_aggregate(other, hooks, x, y)
    for a, b in zip(exact[:3], ghost[:3]):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    for n in exact[3]:
        torch.testing.assert_close(exact[3][n], ghost[3][n], rtol=3e-4, atol=1e-3)
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-5)
    assert exact[4]['grad_sample_bytes'] > 0
    assert ghost[4]['grad_sample_bytes'] == 0
    assert all(not hasattr(p, 'grad_sample') for p in other.parameters())
    for net in (model, other):
        opt = torch.optim.SGD(net.parameters(), lr=.5)
        m.noise_and_step(net, opt, 0, len(x), torch.Generator(device='cuda').manual_seed(42))
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    hooks.remove()
    wrapper.to_standard_module()


def test_exact_ghost_norm_clip_aggregate_update():
    model = tiny()
    x = torch.randn(8, 4, device='cuda')
    op, _ = m.build_from_cache(model, METHODS[0], [x], 42, 1)
    equivalence(model, x, torch.arange(8, device='cuda')%3, op)


@pytest.mark.parametrize('method', METHODS)
def test_real_mnist_batch(method):
    train, _ = load_data()
    x = torch.stack([train[i][0] for i in range(4)]).cuda()
    y = torch.tensor([train[i][1] for i in range(4)], device='cuda')
    model = initialize(42, 'cuda')
    cache = m.synthetic_cache(42, 1, 'cuda', 1, 4)
    op, _ = m.build_from_cache(model, method, cache, 42, 1)
    if method == METHODS[0]:
        equivalence(model, x, y, op)
    else:
        hooks = m.GhostNorm(model, op)
        result = m.ghost_aggregate(model, hooks, x, y)
        assert torch.isfinite(result[1]).all()
        assert all(not hasattr(p, 'grad_sample') for p in model.parameters())
        assert hooks.activations == {}
        hooks.remove()


def test_paired_initialization_shuffle_noise_synthetic_labels():
    models = [initialize(42, 'cuda') for _ in METHODS]
    for net in models[1:]:
        for a,b in zip(models[0].parameters(), net.parameters()):
            assert torch.equal(a,b)
    schedules = []
    for method in METHODS:
        loader = private_loader(torch.arange(1024), 42)
        schedules.append([torch.cat(list(loader)) for epoch in range(2)])
    for schedule in schedules[1:]:
        assert all(torch.equal(a,b) for a,b in zip(schedules[0], schedule))
    caches = [m.synthetic_cache(42, 1, 'cuda', 2, 4) for _ in METHODS]
    for cache in caches[1:]:
        assert all(torch.equal(a,b) for a,b in zip(caches[0], cache))
    gen1, gen2 = [torch.Generator(device='cuda').manual_seed(20043) for _ in range(2)]
    for epoch in range(2):
        p = torch.ones(256, 10, device='cuda')
        assert torch.equal(m.uniform_labels(p, gen1), m.uniform_labels(p, gen2))
    noise_states = []
    for method, net in zip(METHODS, models):
        gen = torch.Generator(device='cuda').manual_seed(40042)
        before = gen.get_state().clone()
        cpu = torch.random.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state().clone()
        m.build_from_cache(net, method, caches[0], 42, 1)
        assert torch.equal(gen.get_state(), before)
        assert torch.equal(torch.random.get_rng_state(), cpu)
        assert torch.equal(torch.cuda.get_rng_state(), cuda)
        draws = [torch.randn(p.numel(), device='cuda', generator=gen) for p in net.parameters()]
        noise_states.append(draws)
    for draws in noise_states[1:]:
        assert all(torch.equal(a,b) for a,b in zip(noise_states[0], draws))


def test_noise_divide_sgd_order():
    net = tiny()
    old = [p.detach().clone() for p in net.parameters()]
    gen = torch.Generator(device='cuda').manual_seed(7)
    refgen = torch.Generator(device='cuda').manual_seed(7)
    expected = []
    for p, v in zip(net.parameters(), old):
        p.grad = torch.full_like(p, 3.)
        noise = torch.randn(p.numel(), device=p.device, generator=refgen).reshape_as(p)
        expected.append(v-.5*(3.+.8*noise)/8)
    m.noise_and_step(net, torch.optim.SGD(net.parameters(), lr=.5), .8, 8, gen)
    for a,b in zip(net.parameters(), expected):
        torch.testing.assert_close(a,b)
    assert torch.equal(gen.get_state(), refgen.get_state())


def test_cuda_operator_lifetime_and_peak_accounting():
    model = tiny()
    x = torch.randn(8, 4, device='cuda')
    for method in METHODS:
        start = phase_start('cuda')
        op, stats = m.build_from_cache(model, method, [x], 42, 1)
        ref = weakref.ref(op)
        tensors = list(op.data.values()) if method in METHODS[2:] else [t for pair in op.data.values() for t in pair]
        refs = [weakref.ref(t) for t in tensors]
        del tensors
        hooks = m.GhostNorm(model, op)
        m.ghost_aggregate(model, hooks, x, torch.arange(8, device='cuda')%3)
        hooks.remove()
        model.zero_grad(set_to_none=True)
        memory = phase_end('cuda', 'build', start)
        assert memory['build_incremental_peak_bytes'] == memory['build_peak_cuda_allocated_bytes']-start[0]
        del hooks, op
        assert ref() is None and all(r() is None for r in refs)
        assert all(not isinstance(v, torch.Tensor) for v in stats.values())


@pytest.mark.parametrize('method', METHODS)
def test_formal_builder_budget(method):
    model = tiny()
    cache = [torch.randn(256, 4, device='cuda') for _ in range(10)]
    _, stats = m.build_from_cache(model, method, cache, 42, 1)
    assert stats['builder_forward_calls'] == 10
    assert stats['builder_samples'] == 2560
    assert stats['builder_vjp_calls'] == (10 if method in METHODS[:2] else 0)
    assert stats['builder_reverse_vectors'] == (2560 if method in METHODS[:2] else 0)


def test_scale_moment_independent_formula():
    factors = {'layer': {'A': torch.diag(torch.tensor([0., 1., 4.], device='cuda')), 'output_dimension': 2}}
    op = m.AOperator(factors, .25)
    raw = 2*(1/(1.001**.5)+4/(4.001**.5))
    ref = 2*(1/1.001+4/4.001)
    assert op.scale == pytest.approx((ref/raw)**.5, rel=1e-12)


def test_synthetic_rng_isolation():
    initialize(42, 'cuda')
    cpu = torch.random.get_rng_state().clone()
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
    gen = torch.Generator(device='cuda').manual_seed(40042)
    noise = gen.get_state().clone()
    m.synthetic_cache(42, 2, 'cuda', 2, 4)
    assert torch.equal(cpu, torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda, torch.cuda.get_rng_state_all()))
    assert torch.equal(noise, gen.get_state())


def test_event_timer_has_no_internal_synchronization(monkeypatch):
    from exp19.profiling import EventProfiler, timed, timestamp
    model = tiny()
    x = torch.randn(8, 4, device='cuda')
    op, _ = m.build_from_cache(model, METHODS[1], [x], 42, 1)
    profiler = EventProfiler()
    sync = torch.cuda.synchronize
    def forbidden(*a, **k):
        pytest.fail('Internal instrumentation synchronized CUDA')
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'synchronize', forbidden)
        hooks = m.GhostNorm(model, op)
        m.ghost_aggregate(model, hooks, x, torch.arange(8, device='cuda')%3, profiler)
        hooks.remove()
        wrapper = m.GradSampleModule(model, loss_reduction='sum')
        m.exact_aggregate(wrapper, op, x, torch.arange(8, device='cuda')%3, profiler)
        wrapper.to_standard_module()
        m.build_from_cache(model, METHODS[2], [x], 42, 1, profiler)
    sync()
    result = profiler.resolve()
    assert all(v >= 0 for v in result.values()) and not profiler.events
    calls = []
    def record(*a, **k):
        calls.append(1)
        return sync(*a, **k)
    monkeypatch.setattr(torch.cuda, 'synchronize', record)
    timestamp('cuda')
    with timed(profiler, 'work', 'cuda'):
        x.square()
    assert len(calls) == 1
    timestamp('cuda')
    profiler.resolve()
    assert len(calls) == 2


@pytest.mark.parametrize('method', METHODS)
def test_disposable_action_warmup(method, monkeypatch):
    from exp19 import run_one as run
    from opacus.accountants import RDPAccountant
    train, _ = load_data()
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    noise = torch.Generator(device='cuda').manual_seed(40042)
    noise_state = noise.get_state().clone()
    references = []
    calls = []
    for name in ('initialize', 'GhostNorm', 'GradSampleModule'):
        original = getattr(run, name)
        def tracked(*args, _original=original, _name=name, **kwargs):
            value = _original(*args, **kwargs)
            references.append(weakref.ref(value))
            calls.append(_name)
            return value
        monkeypatch.setattr(run, name, tracked)
    original_build = run.build_from_cache
    def build(*a, **k):
        operator, stats = original_build(*a, **k)
        references.append(weakref.ref(operator))
        tensors = list(operator.data.values()) if method in METHODS[2:] else [t for pair in operator.data.values() for t in pair]
        references.extend(weakref.ref(t) for t in tensors)
        assert stats['builder_samples'] == 256
        return operator, stats
    monkeypatch.setattr(run, 'build_from_cache', build)
    def forbidden(*a, **k):
        pytest.fail('Warmup executed a step')
    monkeypatch.setattr(torch.optim.SGD, 'step', forbidden)
    monkeypatch.setattr(RDPAccountant, 'step', forbidden)
    assert run.disposable_warmup(method, train, torch.device('cuda:0')) is None
    assert all(ref() is None for ref in references)
    assert ('GradSampleModule' in calls) == (method == METHODS[0])
    assert ('GhostNorm' in calls) == (method != METHODS[0])
    assert torch.equal(cpu, torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda, torch.cuda.get_rng_state_all()))
    assert torch.equal(noise_state, noise.get_state())


def test_balanced_order():
    from exp19.config import ORDER, SEEDS
    assert tuple(ORDER) == SEEDS
    assert all(len(v) == 4 and set(v) == set(METHODS) for v in ORDER.values())
    assert all(len({v.index(method) for v in ORDER.values()}) > 1 for method in METHODS)


def test_activation_only_forward_regression():
    from exp12.curvature import forward, activation_sum
    model = initialize(42, 'cuda')
    x = torch.randn(4, 1, 28, 28, device='cuda')
    outputs = []
    def check(module, inputs, output):
        assert not torch.is_grad_enabled()
        assert not inputs[0].requires_grad
        assert not output.requires_grad
        outputs.append(weakref.ref(output))
    handles = [module.register_forward_hook(check) for module in layers(model).values()]
    acts = m.activation_forward_only(model, x)
    for h in handles:
        h.remove()
    assert all(ref() is None for ref in outputs)
    assert all(not module._forward_hooks for module in layers(model).values())
    with torch.no_grad():
        _, old, _ = forward(model, x)
    assert set(acts) == set(old)
    for name, module in layers(model).items():
        a, count = activation_sum(acts[name], module)
        b, old_count = activation_sum(old[name], module)
        assert count == old_count
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_a_moments_no_item_per_layer(monkeypatch):
    factors = {str(i): {'A': torch.eye(i+2, device='cuda'), 'output_dimension': 3} for i in range(4)}
    def forbidden(*a, **k):
        pytest.fail('AOperator called Tensor.item')
    monkeypatch.setattr(torch.Tensor, 'item', forbidden)
    for power in (.5, .25):
        op = m.AOperator(factors, power)
        assert all(isinstance(v, float) for v in op.moments.values())
        assert op.scale**2*op.moments['m_raw'] == pytest.approx(op.moments['m_reference'])


def test_production_diagnostics_after_train_timer(monkeypatch, tmp_path):
    from exp19 import run_one as run
    phase = {'name': None}
    original_start, original_end = run.phase_start, run.phase_end
    starts = []
    def start(device):
        phase['name'] = 'build' if not starts else 'train'
        starts.append(phase['name'])
        return original_start(device)
    def end(device, name, allocation):
        phase['name'] = None
        return original_end(device, name, allocation)
    monkeypatch.setattr(run, 'phase_start', start)
    monkeypatch.setattr(run, 'phase_end', end)
    original_cpu = torch.Tensor.cpu
    transfers = []
    def cpu(tensor, *a, **k):
        assert phase['name'] != 'train'
        transfers.append(phase['name'])
        return original_cpu(tensor, *a, **k)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    run.run(METHODS[1], 42, True, tmp_path)
    import pandas as pd
    row = pd.read_csv(tmp_path/'runs'/f'{METHODS[1]}_42'/'metrics.csv').iloc[0]
    assert transfers and row.diagnostic_postprocess_seconds > 0
    assert row.algorithm_epoch_seconds == pytest.approx(row.preconditioner_build_seconds+row.private_train_seconds)

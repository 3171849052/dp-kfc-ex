"""Real operator, Opacus, Ghost, RNG and CUDA integration tests."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import os
import tempfile
TMP = ROOT/'exp20/.cache/tmp'
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
from exp20.run_one import initialize
from exp20 import methods as m
from exp20.config import METHODS
from exp20.run_one import load_data, private_loader
from exp20.profiling import phase_start, phase_end


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(4)
    with runtime('cuda:0'):
        yield

def tiny():
    torch.manual_seed(19)
    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3)).cuda()

@pytest.mark.parametrize('method', METHODS)
def test_forward_only_no_labels_no_reverse(method, monkeypatch):
    model = tiny()
    cache = [torch.randn(8, 4, device='cuda')]
    def forbidden(*args, **kwargs):
        pytest.fail('Forward-only builder attempted labels or reverse differentiation')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch, 'multinomial', forbidden)
    op, stats = m.build_from_cache(model, method, cache, 42, 1)
    assert stats['builder_vjp_calls'] == stats['builder_reverse_vectors'] == 0
    assert stats['curvature_backward_seconds'] == 0
    assert stats['builder_forward_calls'] == 1 and stats['builder_samples'] == 8
    assert set(op.data) == set(layers(model))

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

@pytest.mark.parametrize('method', METHODS)
def test_formal_builder_budget(method):
    model = tiny()
    cache = [torch.randn(256, 4, device='cuda') for _ in range(10)]
    _, stats = m.build_from_cache(model, method, cache, 42, 1)
    assert stats['builder_forward_calls'] == 10
    assert stats['builder_samples'] == 2560
    assert stats['builder_vjp_calls'] == 0
    assert stats['builder_reverse_vectors'] == 0

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
    from exp20.profiling import EventProfiler, timed, timestamp
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

def test_production_diagnostics_after_train_timer(monkeypatch, tmp_path):
    from exp20 import run_one as run
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

@pytest.mark.parametrize('power', METHODS)
def test_power_scale_independent_reference(power):
    q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64, device='cuda'))
    a = q @ torch.diag(torch.tensor([.001, .02, .3, 1., 4.], device='cuda', dtype=torch.float64)) @ q.T
    factors = {'a': dict(A=a, output_dimension=3), 'b': dict(A=a*2, output_dimension=7)}
    op = m.AOperator(factors, power)
    raw = ref = 0.
    for name, factor in factors.items():
        eig, vec = torch.linalg.eigh(factor['A'])
        raw += factor['output_dimension']*(eig*(eig+.001)**(-2*power)).sum().item()
        ref += factor['output_dimension']*(eig/(eig+.001)).sum().item()
        expected = (vec*(eig+.001)**(-power))@vec.T
        torch.testing.assert_close(op.data[name].double(), expected, atol=1e-5, rtol=1e-6)
        if power == 0:
            torch.testing.assert_close(op.data[name], torch.eye(5, device='cuda'), atol=0, rtol=0)
    assert op.scale**2*raw == pytest.approx(ref, rel=1e-12)
    if power == .5:
        assert op.scale == 1.


@pytest.mark.parametrize('power', METHODS)
def test_real_mnist_ghost_against_independent_per_sample(power):
    train, _ = load_data()
    x = torch.stack([train[i][0] for i in range(4)]).cuda()
    y = torch.tensor([train[i][1] for i in range(4)], device='cuda')
    model = initialize(42, 'cuda')
    op, _ = m.build_from_cache(model, power, m.synthetic_cache(42, 1, 'cuda', 1, 4), 42, 1)
    ref = copy.deepcopy(model)
    sums = {n: torch.zeros_like(p) for n, p in ref.named_parameters()}
    norms = []
    for xx, yy in zip(x, y):
        ref.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(ref(xx[None]), yy[None]).backward()
        op.transform_aggregate_gradient(ref)
        norm = sum(p.grad.square().sum() for p in ref.parameters()).sqrt()
        norms.append(norm)
        factor = (1/(norm+1e-6)).clamp(max=1)
        for n, p in ref.named_parameters():
            sums[n].add_(p.grad*factor)
    hooks = m.GhostNorm(model, op)
    _, actual, _, _, stats = m.ghost_aggregate(model, hooks, x, y)
    torch.testing.assert_close(actual, torch.stack(norms), rtol=5e-4, atol=1e-4)
    for n, p in model.named_parameters():
        assert not hasattr(p, 'grad_sample')
        torch.testing.assert_close(p.grad, sums[n], rtol=8e-4, atol=3e-5)
    assert stats['grad_sample_bytes'] == 0
    hooks.remove()


def test_synthetic_forks_only_selected_device(monkeypatch):
    original = torch.random.fork_rng
    calls = []
    def tracked(*args, **kwargs):
        calls.append(kwargs['devices'])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.random, 'fork_rng', tracked)
    m.synthetic_cache(42, 1, 'cuda:0', 1, 4)
    assert calls == [[0]]

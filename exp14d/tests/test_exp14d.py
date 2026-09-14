import inspect
import math
from unittest.mock import patch
import pytest
import torch
from exp14d import config as cfg
from exp14d import run_exp14d as runner
from exp14d.whitening import energy_metrics, coordinates, synthetic_probe, diagnose
from exp14b.builders import build_from_cache, synthetic_cache
from exp14b.operator import Operator
from exp13.ghost import ghost_aggregate, noise_and_step
from exp12.runtime import runtime


def test_protocol_and_reuse():
    assert cfg.BETAS == (0., .125, .25, .375, .5, .75, 1.)
    assert cfg.SEEDS == (42, 7)
    assert cfg.PRIVATE_STEPS == 1170
    assert cfg.EPOCHS == 5 and cfg.BATCH_SIZE == 256
    assert (cfg.EPSILON, cfg.DELTA, cfg.MAX_GRAD_NORM) == (1., 1e-5, 1.)
    assert (cfg.LEARNING_RATE, cfg.MOMENTUM, cfg.WEIGHT_DECAY) == (.5, 0, 0)
    assert (cfg.SYNTHETIC_BATCHES, cfg.SYNTHETIC_BATCH_SIZE) == (10, 256)
    assert runner.build_from_cache is build_from_cache
    assert runner.training.ghost_aggregate is ghost_aggregate
    assert runner.training.noise_and_step is noise_and_step
    assert runner.training.synthetic_cache is synthetic_cache
    assert build_from_cache.__globals__['Operator'] is Operator
    source = inspect.getsource(runner.run)
    assert 'training.run(' in source and 'optimizer.step' not in source


@pytest.mark.parametrize('d', [10, 11, 100])
def test_isotropic(d):
    m = energy_metrics(torch.ones(d))
    assert m['W'] == pytest.approx(1)
    assert m['effective_dimension'] == pytest.approx(d)
    assert m['top10_energy_share'] == pytest.approx(math.ceil(.1*d)/d)
    assert m['directions_for_90pct'] == pytest.approx(math.ceil(.9*d)/d)


def test_anisotropic():
    m = energy_metrics(torch.tensor([3.]+[0.]*99))
    assert m['W'] == pytest.approx(.01)
    assert m['top10_energy_share'] == pytest.approx(1)
    assert m['directions_for_90pct'] == pytest.approx(.01)


@pytest.mark.parametrize('beta', cfg.BETAS)
def test_coordinates(beta):
    a = torch.tensor([[2., .3], [.3, .5]], dtype=torch.float64)
    c = torch.tensor([[.4, .1], [.1, 1.]], dtype=torch.float64)
    operator = Operator({'toy': {'A': a, 'C': c}}, beta)
    ae, ua = torch.linalg.eigh(a)
    ce, uc = torch.linalg.eigh(c)
    g = torch.tensor([[[.1, -.3], [.7, .4]], [[.2, .5], [-.1, .8]]])
    left = uc @ torch.diag((ce+cfg.DAMPING)**(-beta)) @ uc.T
    right = ua @ torch.diag((ae+cfg.DAMPING)**(-beta)) @ ua.T
    expected = uc.T @ (operator.diagnostics['scale_match']*left @ g.double() @ right) @ ua
    torch.testing.assert_close(coordinates(operator, 'toy', g, ua, uc), expected, rtol=3e-6, atol=3e-7)


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with runtime(device):
        model = runner.training.initialize(42, device)
        cache = synthetic_cache(42, 1, device, 1, 4)
        operator, _, factors = build_from_cache(model, .5, cache, 42, 1)
        yield model, operator, factors, device


def test_probe_rng(setup):
    model, _, _, device = setup
    cpu, cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
    reference = synthetic_probe(42, 1, device, 8)
    for beta in cfg.BETAS:
        cache = synthetic_cache(42, 1, device, 1, 4)
        build_from_cache(model, beta, cache, 42, 1)
        actual = synthetic_probe(42, 1, device, 8)
        assert all(torch.equal(a, b) for a, b in zip(reference, actual))
        assert not torch.equal(cache[0], actual[0][:4])
    assert torch.equal(cpu, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    assert not torch.equal(reference[0], synthetic_probe(42, 2, device, 8)[0])


def test_diagnostic_finite_streaming_and_no_mutation(setup):
    model, operator, factors, device = setup
    x, y = synthetic_probe(42, 1, device, 8)
    before = {n: p.clone() for n, p in model.state_dict().items()}
    grads = [p.grad for p in model.parameters()]
    with patch.object(runner.training.datasets, 'MNIST', side_effect=AssertionError('private data access')):
        aggregate, rows, profile = diagnose(model, factors, operator, x, y, 2)
        other, _, _ = diagnose(model, factors, operator, x, y, 4)
    assert aggregate == pytest.approx(other, rel=1e-5)
    assert all(math.isfinite(v) for v in aggregate.values())
    assert all(math.isfinite(v) for row in rows for k, v in row.items() if k != 'layer')
    assert all(0 < row['W'] <= 1 for row in rows)
    assert all(math.isfinite(v) for v in profile)
    assert sum(row['d'] for row in rows) == sum(p.numel() for p in model.parameters())
    assert aggregate['W_global'] == pytest.approx(sum(r['d']*r['W'] for r in rows)/sum(r['d'] for r in rows))
    for n, p in model.state_dict().items():
        assert torch.equal(p, before[n])
    assert all(p.grad is g for p, g in zip(model.parameters(), grads))
    assert list(inspect.signature(diagnose).parameters) == ['model', 'factors', 'operator', 'probe_x', 'probe_y', 'batch_size']


def test_streamed_second_moments_match_explicit_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    factors = {'0': {'A': torch.diag(torch.tensor([1., 2., 3.])),
                     'C': torch.diag(torch.tensor([2., 4.]))}}
    operator = Operator(factors, .25)
    x = torch.tensor([[1., 2.], [2., -.5], [-1., .3]])
    y = torch.tensor([0, 1, 0])
    aggregate, rows, profile = diagnose(model, factors, operator, x, y, 2)
    energies = []
    for xi, yi in zip(x, y):
        gs = torch.autograd.grad(torch.nn.functional.cross_entropy(model(xi[None]), yi[None]),
                                 tuple(model.parameters()))
        g = torch.cat((gs[0], gs[1][:, None]), 1)
        energies.append(operator.transform_matrix('0', g).double().square().flatten())
    e = torch.stack(energies).mean(0)
    expected = energy_metrics(e)
    for key, value in expected.items():
        assert rows[0][key] == pytest.approx(value, rel=1e-6)
    assert aggregate['W_global'] == pytest.approx(expected['W'])
    q = torch.linspace(0, 1, 1001, dtype=torch.float64)
    torch.testing.assert_close(torch.tensor(profile, dtype=torch.float64), torch.quantile(e/e.mean(), 1-q))

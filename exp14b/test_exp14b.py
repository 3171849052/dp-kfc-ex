"""Core mathematical invariants only; explicit sample gradients are test-only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import copy
import pytest
import torch
from torch.nn import functional as F
from exp12.curvature import estimate
from exp12.runtime import runtime
from exp14.operator import Operator as RawOperator
from exp13.ghost import GhostNorm, ghost_aggregate
from exp14b import config as cfg
from exp14b.builders import synthetic_cache, build_preconditioner
from exp14b.operator import Operator
from exp14b.run_exp14b import initialize


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with runtime(device):
        model = initialize(42, device)
        cache = synthetic_cache(42, 1, device, 1, 4)
        factors, _ = estimate(model, cache, 'KFAC-U', seed=20043)
        yield model, factors, device


def test_half_exp14_regression(setup):
    _, factors, _ = setup
    actual, expected = Operator(factors, .5), RawOperator(factors, .5)
    assert actual.diagnostics['scale_match'] == pytest.approx(1.)
    for name in factors:
        for a, b in zip(actual.data[name], expected.data[name]):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize('beta', cfg.BETAS)
def test_ghost_clipped_aggregate(setup, beta):
    model, factors, device = setup
    model = copy.deepcopy(model)
    operator = Operator(factors, beta)
    x = synthetic_cache(123, 1, device, 1, 4)[0]
    y = torch.arange(4, device=device)
    expected = torch.zeros(sum(p.numel() for p in model.parameters()), device=device)
    norms = []
    for xi, yi in zip(x, y):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xi[None]), yi[None]).backward()
        operator.transform_aggregate_gradient(model)
        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        norm = grad.norm()
        norms.append(norm)
        expected += grad * (cfg.MAX_GRAD_NORM / (norm + 1e-6)).clamp(max=1)
    hooks = GhostNorm(model, operator)
    _, actual_norms, _ = ghost_aggregate(model, hooks, x, y)
    actual = torch.cat([p.grad.flatten() for p in model.parameters()])
    hooks.remove()
    torch.testing.assert_close(actual_norms, torch.stack(norms), rtol=3e-4, atol=2e-5)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)


def test_builder_rng_isolation(setup):
    model, _, device = setup
    cpu, cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
    for beta in cfg.BETAS:
        build_preconditioner(model, beta, 42, 1, device, batches=1, batch_size=4)
        assert torch.equal(torch.random.get_rng_state(), cpu)
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda))


def test_same_initialization(setup):
    _, _, device = setup
    reference = initialize(42, device)
    for beta in cfg.BETAS:
        model = initialize(42, device)
        for p, q in zip(model.parameters(), reference.parameters()):
            assert torch.equal(p, q)


@pytest.mark.parametrize('beta', cfg.BETAS)
def test_matched_small_matrix_reference(beta):
    # Two distinct blocks verify a global sum before taking the scale ratio.
    a = torch.tensor([[2., .3], [.3, .5]], dtype=torch.float64)
    c = torch.tensor([[.4, .1], [.1, 1.]], dtype=torch.float64)
    factors = {'one': {'A': a, 'C': c}, 'two': {'A': 3 * a, 'C': .2 * c}}
    operator = Operator(factors, beta)

    def power(factor, exponent):
        e, q = torch.linalg.eigh(factor)
        return q @ torch.diag((e + cfg.DAMPING).pow(-exponent)) @ q.T

    def moment(exponent):
        return sum(torch.trace(f['C'] @ power(f['C'], 2 * exponent)) *
                   torch.trace(f['A'] @ power(f['A'], 2 * exponent))
                   for f in factors.values()).item()

    raw, reference = moment(beta), moment(.5)
    scale = (reference / raw) ** .5
    stats = operator.diagnostics
    assert stats['predicted_second_moment_raw'] == pytest.approx(raw)
    assert stats['predicted_second_moment_ref'] == pytest.approx(reference)
    assert stats['scale_match'] ** 2 * raw == pytest.approx(reference)
    assert stats['predicted_rms_matched'] == pytest.approx(reference ** .5)
    g = torch.tensor([[.1, -.3], [.7, .4]])
    for name, f in factors.items():
        expected = scale * power(f['C'], beta) @ g.double() @ power(f['A'], beta)
        torch.testing.assert_close(operator.transform_matrix(name, g).double(), expected,
                                   rtol=2e-6, atol=2e-7)

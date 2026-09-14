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
from exp13.operator import Operator as Exp13Operator
from exp13.ghost import GhostNorm, ghost_aggregate
from exp14 import config as cfg
from exp14.builders import synthetic_cache, build_preconditioner
from exp14.operator import Operator, inverse_power
from exp14.run_exp14 import initialize


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with runtime(device):
        model = initialize(42, device)
        cache = synthetic_cache(42, 1, device, 1, 4)
        factors, _ = estimate(model, cache, 'KFAC-U', seed=20043)
        yield model, factors, device


def test_half_exp13_regression(setup):
    _, factors, _ = setup
    actual, expected = Operator(factors, .5), Exp13Operator(factors)
    for name in factors:
        for a, b in zip(actual.data[name], expected.data[name]):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize('beta', cfg.BETAS)
def test_small_matrix_power(beta):
    factor = torch.tensor([[2., .3], [.3, .5]], dtype=torch.float64)
    e, q = torch.linalg.eigh(factor)
    reference = q @ torch.diag((e + cfg.DAMPING).pow(-beta)) @ q.T
    torch.testing.assert_close(inverse_power(factor, beta).double(), reference,
                               atol=1e-7, rtol=1e-6)


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
        if beta != 0:
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


def test_identity_transform_and_unclipped_aggregate(setup):
    model, factors, device = setup
    model = copy.deepcopy(model)
    operator = Operator(factors, 0)
    for name, (left, right) in operator.data.items():
        g = torch.randn(len(left), len(right), device=device)
        torch.testing.assert_close(operator.transform_matrix(name, g), g)
    x = synthetic_cache(123, 1, device, 1, 4)[0]
    y = torch.arange(4, device=device)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y, reduction='sum').backward()
    expected = [p.grad.clone() for p in model.parameters()]
    hooks = GhostNorm(model, operator)
    _, _, clips = ghost_aggregate(model, hooks, x, y, bound=float('inf'))
    hooks.remove()
    assert (clips == 1).all()
    for p, reference in zip(model.parameters(), expected):
        torch.testing.assert_close(p.grad, reference)


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

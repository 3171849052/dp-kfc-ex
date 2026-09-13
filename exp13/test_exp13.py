"""Only invariants that affect the curvature/DP comparison; smoke is the CLI run."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import copy
import pytest
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from exp12.curvature import estimate
from exp12.runtime import runtime
from exp13 import config as cfg
from exp13.builders import ESTIMATORS, synthetic_cache, build_preconditioner
from exp13.operator import Operator
from exp13.ghost import GhostNorm, ghost_aggregate
from exp13.run_exp13 import initialize


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with runtime(device):
        model = initialize(42, device)
        cache = synthetic_cache(42, 1, device, 1, 4)
        factors = {method: estimate(model, cache, ESTIMATORS[method], seed=20043)[0]
                   for method in cfg.METHODS}
        yield model, factors, device


def test_common_activation_factors(setup):
    _, factors, _ = setup
    for method in cfg.METHODS[1:]:
        for name in factors[method]:
            torch.testing.assert_close(factors[method][name]['A'], factors['DP-KFC'][name]['A'])


def test_damped_inverse_sqrt(setup):
    _, factors, device = setup
    for method in cfg.METHODS:
        operator = Operator(factors[method])
        for name, (left, right) in operator.data.items():
            for root, factor in ((left, factors[method][name]['C']), (right, factors[method][name]['A'])):
                eye = torch.eye(len(root), device=device, dtype=torch.float64)
                whitened = root.double() @ (factor.double() + cfg.DAMPING * eye) @ root.double().T
                torch.testing.assert_close(whitened, eye, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize('method', cfg.METHODS[1:])
def test_ghost_clipped_aggregate(setup, method):
    model, factors, device = setup
    model = copy.deepcopy(model)
    operator = Operator(factors[method])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    data = datasets.MNIST(ROOT / 'exp1/data', train=True, download=False, transform=transform)
    x = torch.stack([data[i][0] for i in range(4)]).to(device)
    y = torch.tensor([data[i][1] for i in range(4)], device=device)
    # Explicit one-example ordinary gradients exist only in this correctness test.
    expected = torch.zeros(sum(p.numel() for p in model.parameters()), device=device)
    norms = []
    for xi, yi in zip(x, y):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xi[None]), yi[None], reduction='sum').backward()
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
    assert not hooks.activations
    assert all(getattr(p, 'grad_sample', None) is None for p in model.parameters())


def test_builder_rng_isolation(setup):
    model, _, device = setup
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all()
    for method in cfg.METHODS:
        build_preconditioner(model, method, 42, 1, device, batches=1, batch_size=4)
        assert torch.equal(torch.random.get_rng_state(), cpu)
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda))

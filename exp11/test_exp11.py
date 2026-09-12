"""Core float32 equivalence, including mixed Conv/Linear and augmented biases."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
from unittest.mock import patch
import pytest
import torch
from exp11.operators import build
from exp11.ghost import GhostNorm, exact_aggregate, ghost_aggregate, noise_and_step
from exp10 import run_exp10 as base

ERRORS = {}


def compare(label, exact, ghost):
    error = (exact - ghost).abs()
    # Relative infinity-norm error is stable for near-zero gradient coordinates.
    relative = error.max() / exact.abs().max().clamp_min(1e-12)
    ERRORS[label] = dict(max_absolute=error.max().item(), relative_inf=relative.item())
    torch.testing.assert_close(ghost, exact, rtol=2e-4, atol=2e-6)
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    (output / 'correctness_errors.json').write_text(json.dumps(ERRORS, indent=2) + '\n')


@pytest.mark.parametrize('kind', ['DP-SGD', 'Factorized Equil', 'DP-KFC'])
def test_equivalence(kind):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    model = base.SimpleCNN().to(device)
    if kind == 'DP-KFC':
        with patch.object(base, 'GradSampleModule', side_effect=AssertionError('KFC builder must use plain model')):
            operator = build(model, kind, 42, 1, device)
    else:
        operator = build(model, kind, 42, 1, device)
    reference = base.GradSampleModule(copy.deepcopy(model), loss_reduction='sum')
    hooks = GhostNorm(model, operator)
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.tensor([0, 1, 4, 9], device=device)
    opt_e = torch.optim.SGD(reference.parameters(), lr=.5)
    opt_g = torch.optim.SGD(model.parameters(), lr=.5)
    _, ne, ce, _ = exact_aggregate(reference, operator, x, y)
    _, ng, cg, _ = ghost_aggregate(model, hooks, x, y)
    assert hooks.backends == dict(conv1='gradient_matrix', conv2='gradient_matrix',
                                  fc1='gram', fc2='gram')
    compare(kind + '/norm', ne, ng)
    compare(kind + '/clip', ce, cg)
    compare(kind + '/aggregate', torch.cat([p.grad.flatten() for p in reference.parameters()]),
            torch.cat([p.grad.flatten() for p in model.parameters()]))
    for (name, p), pe in zip(model.named_parameters(), reference.parameters()):
        if name.endswith('bias'):
            compare(kind + '/' + name, pe.grad, p.grad)
        assert getattr(p, 'grad_sample', None) is None
        assert getattr(p, '_current_grad_sample', None) is None
    assert hooks.activations == {}
    noise_and_step(reference, opt_e, 0., batch_size=4)
    noise_and_step(model, opt_g, 0., batch_size=4)
    compare(kind + '/update', torch.cat([p.flatten() for p in reference.parameters()]),
            torch.cat([p.flatten() for p in model.parameters()]))
    hooks.remove()
    reference.to_standard_module()

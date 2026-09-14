"""Only the mathematical and clipping-order invariants needed for exp15."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from dp_kfac.models import SimpleCNN
from dp_kfac.standalone.trainer import build_optimizer, private_step as baseline_step
from dp_kfac.privacy import _compute_clip_factors
from exp15.preconditioner import InverseLBFGS, SyntheticKLBFGS
from exp15.run_exp15 import configuration, private_step


def factor(n):
    f = InverseLBFGS(torch.device('cpu'), configuration(0, 0)['lbfgs'])
    curvature = torch.diag(torch.linspace(.5, 3, n, dtype=torch.float64))
    for _ in range(3):
        s = torch.randn(n, dtype=torch.float64)
        f.append(s, curvature @ s, torch.ones(n), damp=True)
    return f


def test_compact_powers():
    torch.manual_seed(0)
    f = factor(12)
    identity = torch.eye(12, dtype=torch.float64)
    h = f.hv(identity)
    ev, q = torch.linalg.eigh((h + h.T) / 2)
    v = torch.randn(5, 12, dtype=torch.float64)
    torch.testing.assert_close(f.power(v, 0), v, rtol=0, atol=0)
    torch.testing.assert_close(f.power(v, 1), f.hv(v.T).T, rtol=1e-9, atol=1e-9)
    for p in (.25, .5, .75):
        actual = f.power(v, p)
        assert torch.isfinite(actual).all() and (ev > 0).all()
        torch.testing.assert_close(actual, v @ (q * ev.pow(p)) @ q.T, rtol=1e-9, atol=1e-9)


def test_identity_matches_existing_dp_sgd():
    torch.manual_seed(9)
    c = configuration(0., 0)
    model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    baseline = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    baseline.load_state_dict(model.state_dict())
    state = SyntheticKLBFGS(c, torch.device('cpu'))
    x, y = torch.randn(2, 1, 28, 28), torch.tensor([2, 3])
    optimizer, baseline_optimizer = [build_optimizer(m, c['training']) for m in (model, baseline)]
    torch.manual_seed(20)
    private_step(model, optimizer, x, y, c, state, 1.3)
    c['algorithm'] = 'dp_sgd'
    torch.manual_seed(20)
    baseline_step(baseline, baseline_optimizer, RDPAccountant(), x, y, c, None, 1.3, .01)
    for actual, expected in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kronecker_preclip_and_noisy_update():
    torch.manual_seed(12)
    c = configuration(.5, 0)
    model = GradSampleModule(torch.nn.Sequential(torch.nn.Linear(2, 2)), loss_reduction='sum')
    state = SyntheticKLBFGS(c, torch.device('cpu'))
    ha, hg = factor(3), factor(2)
    state.factors['0'] = (ha, hg)
    x, y = torch.randn(3, 2), torch.tensor([0, 1, 0])
    torch.nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
    layer = model._module[0]
    matrix = torch.cat([layer.weight.grad_sample, layer.bias.grad_sample.unsqueeze(-1)], -1)
    def dense_half(f, n):
        h = f.hv(torch.eye(n, dtype=torch.float64))
        ev, q = torch.linalg.eigh((h + h.T) / 2)
        return ((q * ev.sqrt()) @ q.T).float()
    expected = dense_half(hg, 2) @ matrix @ dense_half(ha, 3)
    norms = expected.flatten(1).norm(dim=1)
    clips = _compute_clip_factors(norms.square(), 1.)
    grads = [(expected[..., :-1] * clips[:, None, None]).sum(0),
             (expected[..., -1] * clips[:, None]).sum(0)]
    before = [p.detach().clone() for p in model.parameters()]
    torch.manual_seed(99)
    expected_parameters = [old - .5 * (g + torch.randn_like(g.flatten()).reshape_as(g)*1.3)/3
                           for old, g in zip(before, grads)]
    torch.manual_seed(99)
    result = private_step(model, build_optimizer(model, c['training']), x, y, c, state, 1.3)
    assert abs(result['preclip_norm_mean'] - norms.mean().item()) < 1e-5
    for actual, expected_p in zip(model.parameters(), expected_parameters):
        torch.testing.assert_close(actual, expected_p, rtol=1e-5, atol=1e-6)

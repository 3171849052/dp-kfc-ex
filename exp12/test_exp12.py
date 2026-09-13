"""Core scientific invariants only."""
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from exp12 import curvature as c


def test_output_factor():
    p = torch.randn(7, 10, dtype=torch.double).softmax(-1)
    s = c.output_factor(p)
    torch.testing.assert_close(s @ s.mT, torch.diag_embed(p) - p.unsqueeze(-1)*p.unsqueeze(-2))


def test_fisher_ggn():
    torch.manual_seed(3)
    w = torch.randn(20, dtype=torch.double, requires_grad=True)
    x = torch.randn(2, dtype=torch.double)
    def logits(v):
        return v[8:].reshape(3, 4) @ (v[:8].reshape(4, 2) @ x).tanh()
    z = logits(w)
    p = z.softmax(0)
    scores = torch.stack([torch.autograd.grad(z.log_softmax(0)[i], w, retain_graph=True)[0].flatten() for i in range(3)])
    jac = torch.autograd.functional.jacobian(logits, w).reshape(3, -1)
    torch.testing.assert_close(scores.T @ (p[:, None]*scores), jac.T @ (torch.diag(p)-p[:,None]*p[None,:]) @ jac)


def test_mc_convergence():
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 3)).double()
    x = [torch.randn(4, 2, dtype=torch.double)]
    ref, _ = c.estimate(model, x, 'KFLR')
    mc, _ = c.estimate(model, x, 'KFAC-M', k=3000, seed=4)
    for name in ref:
        assert (mc[name]['C']-ref[name]['C']).norm()/ref[name]['C'].norm() < .08


def test_one_layer_kfra_and_shared_a():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(2, 3)).double()
    x = [torch.randn(5, 2, dtype=torch.double)]
    ref, _ = c.estimate(model, x, 'KFLR')
    for method in ['KFRA', 'KFAC-U', 'KFAC-M']:
        out, _ = c.estimate(model, x, method)
        torch.testing.assert_close(out['0']['A'], ref['0']['A'])
        if method == 'KFRA':
            torch.testing.assert_close(out['0']['C'], ref['0']['C'])


def test_oracle_isolation():
    assert 'oracle' not in inspect.getsource(c)
    from exp12 import probes
    assert 'MNIST' not in inspect.getsource(probes)


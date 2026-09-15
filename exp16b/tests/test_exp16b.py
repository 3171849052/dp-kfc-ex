"""Numerical invariants and isolation of synthetic refresh."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from copy import deepcopy
import pytest
import torch
from opacus import GradSampleModule
from exp15.preconditioner import InverseLBFGS, SyntheticKLBFGS
from exp15.run_exp15 import configuration as old_configuration
from exp16b.preconditioner import Preconditioner, FisherFactor
from exp16b.run_exp16b import configuration, private_step
from dp_kfac.models import SimpleCNN
from dp_kfac.standalone.trainer import build_optimizer

torch.set_num_threads(4)


def setup(mode, q):
    c = configuration(q, smoke=True, mode=mode)
    model = GradSampleModule(torch.nn.Sequential(torch.nn.Linear(2, 2)), loss_reduction='sum')
    state = Preconditioner(c, torch.device('cpu'))
    factors = []
    for n in (3, 2):
        factor = InverseLBFGS(torch.device('cpu'), c['lbfgs'])
        for _ in range(3):
            s = torch.randn(n, dtype=torch.float64)
            factor.append(s, torch.arange(1, n+1).double() * s)
        factors.append(factor)
    state.hessian.factors['0'] = tuple(factors)
    state.fisher['0'] = tuple(FisherFactor(
        torch.diag(torch.arange(1, n+1).double()), c['kfac']['damping'], q) for n in (3, 2))
    return c, model, state


@pytest.mark.parametrize('mode', ['hessian', 'fisher', 'nested'])
def test_identity(mode):
    _, model, state = setup(mode, 0)
    torch.nn.functional.cross_entropy(model(torch.randn(3, 2)), torch.tensor([0, 1, 0]),
                                      reduction='sum').backward()
    before = [p.grad_sample for p in model.parameters()]
    copies = [v.clone() for v in before]
    state.apply(model, 0)
    for p, ref, copied in zip(model.parameters(), before, copies):
        assert p.grad_sample is ref
        assert torch.equal(p.grad_sample, copied)


@pytest.mark.parametrize('q', [.25, .5])
def test_hessian_matches_exp15(q):
    c, model, state = setup('hessian', q)
    other = GradSampleModule(torch.nn.Sequential(torch.nn.Linear(2, 2)), loss_reduction='sum')
    other.load_state_dict(model.state_dict())
    x, y = torch.randn(3, 2), torch.tensor([0, 1, 0])
    for m in (model, other):
        torch.nn.functional.cross_entropy(m(x), y, reduction='sum').backward()
    reference = SyntheticKLBFGS(c, torch.device('cpu'))
    reference.factors = deepcopy(state.hessian.factors)
    state.apply(model, q)
    reference.apply(other, q)
    for a, b in zip(model.parameters(), other.parameters()):
        assert torch.equal(a.grad_sample, b.grad_sample)


@pytest.mark.parametrize('mode', ['hessian', 'fisher', 'nested'])
@pytest.mark.parametrize('q', [.25, .5])
def test_dense_formula_and_dp_order(mode, q):
    c, model, state = setup(mode, q)
    x, y = torch.randn(3, 2), torch.tensor([0, 1, 0])
    torch.nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
    layer = model._module[0]
    raw = torch.cat((layer.weight.grad_sample, layer.bias.grad_sample.unsqueeze(-1)), -1)
    def dense(f, n):
        h = f.hv(torch.eye(n, dtype=torch.float64))
        e, v = torch.linalg.eigh((h+h.T)/2)
        return ((v*e.pow(q)) @ v.T).float()
    z = raw
    if mode in ('hessian', 'nested'):
        ha, hg = state.hessian.factors['0']
        z = dense(hg, 2) @ raw @ dense(ha, 3)
    u = z
    if mode in ('fisher', 'nested'):
        a, g = state.fisher['0']
        u = g.inverse_power.float() @ z @ a.inverse_power.float()
    norms = u.flatten(1).norm(dim=1)
    from dp_kfac.privacy import _compute_clip_factors
    clips = _compute_clip_factors(norms.square(), c['privacy']['max_grad_norm'])
    grads = [(u[..., :-1] * clips[:, None, None]).sum(0).flatten(),
             (u[..., -1] * clips[:, None]).sum(0)]
    torch.manual_seed(99)
    expected = [p.detach() - .5 * ((g + 1.3*torch.randn_like(g))/3).view_as(p)
                for p, g in zip(model.parameters(), grads)]
    torch.manual_seed(99)
    result = private_step(model, build_optimizer(model, c['training']), x, y, c, state, 1.3)
    assert result['preclip_norm_mean'] == pytest.approx(norms.mean().item(), rel=1e-5)
    assert result['hessian_stage_norm_mean'] == pytest.approx(z.flatten(1).norm(dim=1).mean().item(), rel=1e-5)
    for actual, ref in zip(model.parameters(), expected):
        torch.testing.assert_close(actual, ref, rtol=1e-5, atol=1e-6)


def test_protocol():
    for mode in ('hessian', 'fisher', 'nested'):
        c = configuration(.5, mode=mode)
        old = old_configuration(.5, 42)
        old['lbfgs']['factor_damping'] = 0.1
        for section in ('training', 'privacy', 'synthetic', 'lbfgs', 'kfac', 'runtime'):
            assert c[section] == old[section]
        assert c['data']['batch_size'] == c['data']['eval_batch_size'] == 256
        assert c['data']['drop_last']
        assert c['seed'] == 42


@pytest.mark.parametrize('mode', ['fisher', 'nested'])
def test_refresh_frozen_independent_and_transformed(monkeypatch, mode):
    import exp16b.preconditioner as new
    import exp15.preconditioner as old
    c = configuration(.5, smoke=True, mode=mode)
    c['synthetic'].update(samples=8, batch_size=4)
    model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    state = Preconditioner(c, torch.device('cpu'))
    theta = deepcopy(model.state_dict())
    streams = {'a': [], 'b': []}
    original_batches = new.pink_batches
    def batches(stage):
        def generate(config, device):
            for x, y in original_batches(config, device):
                streams[stage].append(x.clone())
                yield x, y
        return generate
    monkeypatch.setattr(old, 'pink_batches', batches('a'))
    monkeypatch.setattr(new, 'pink_batches', batches('b'))
    frozen = {}
    original_stage_b = state.refresh_fisher
    def stage_b(m, epoch):
        for name, factors in state.hessian.factors.items():
            frozen[name] = [deepcopy(f.pairs) for f in factors]
        def forbidden(*args, **kwargs):
            pytest.fail('Stage B updated H')
        monkeypatch.setattr(InverseLBFGS, 'append', forbidden)
        original_stage_b(m, epoch)
    monkeypatch.setattr(state, 'refresh_fisher', stage_b)
    # The private model must never be forwarded by refresh.
    monkeypatch.setattr(model, 'forward', lambda *a: pytest.fail('private model used'))
    rng = torch.random.get_rng_state()
    state.refresh(model, 1)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key in theta:
        assert torch.equal(theta[key], model.state_dict()[key])
    assert all(p.grad is None and p.grad_sample is None for p in model.parameters())
    if mode == 'nested':
        assert len(streams['a']) == len(streams['b']) == 2
        assert not torch.equal(streams['a'][0], streams['b'][0])
        for name, factors in state.hessian.factors.items():
            for factor, snapshot in zip(factors, frozen[name]):
                for key in snapshot:
                    if torch.is_tensor(snapshot[key]):
                        assert torch.equal(factor.pairs[key], snapshot[key])
                    else:
                        assert factor.pairs[key] == snapshot[key]
    # Independently re-record Stage B and compare transformed factor covariances.
    totals = {}
    with torch.random.fork_rng():
        torch.manual_seed(c['seed'] + 20001)
        probe = SimpleCNN()
        probe.load_state_dict(model._module.state_dict())
        recorder = new.KFACRecorder(probe)
        recorder.enable()
        for x, y in original_batches(c, torch.device('cpu')):
            probe.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(probe(x), y, reduction='sum').backward()
            for name, module in probe.named_modules():
                if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    continue
                a = recorder.activations[name]
                if isinstance(module, torch.nn.Conv2d):
                    a = torch.nn.functional.unfold(a, module.kernel_size, padding=module.padding,
                                                   stride=module.stride).transpose(1, 2).flatten(0, 1)
                a = torch.cat((a, torch.ones_like(a[:, :1])), 1).double()
                g = new.rows(recorder.backprops[name]).double()
                covs = [a.T@a/len(a), g.T@g/len(g)]
                if mode == 'nested':
                    # Congruence transform of raw covariance, independent of transformed rows.
                    for i, f in enumerate(state.hessian.factors[name]):
                        hq = f.power(torch.eye(len(covs[i]), dtype=torch.float64), .5)
                        covs[i] = hq @ covs[i] @ hq
                if name not in totals:
                    totals[name] = covs
                else:
                    for total, cov in zip(totals[name], covs):
                        total.add_(cov)
        recorder.remove()
    for name, factors in state.fisher.items():
        for factor, ref in zip(factors, totals[name]):
            assert torch.equal(factor.covariance, factor.covariance.T)
            assert torch.isfinite(factor.covariance).all()
            assert torch.linalg.eigvalsh(factor.damped).min() > 0
            torch.testing.assert_close(factor.covariance, ref / 2, rtol=1e-8, atol=1e-9)


def test_exp16b_damping_reaches_hessian():
    c = configuration(.5, smoke=True, mode='hessian')
    assert c['lbfgs']['factor_damping'] == 0.1
    assert c['kfac']['damping'] == 0.001
    model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    state = Preconditioner(c, torch.device('cpu'))
    state.refresh(model, 1)
    assert state.hessian.factors
    for factors in state.hessian.factors.values():
        for factor in factors:
            assert isinstance(factor, InverseLBFGS)
            assert factor.params['Kron_BFGS_H_epsilon'] == 0.1

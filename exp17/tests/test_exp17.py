import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import copy
import numpy as np
import pytest
import torch
from opacus import GradSampleModule
from exp17 import run_exp17 as runner
from exp17.curvature import Diagnostic, TinyCNN, geometry, save_checkpoint
from exp15.preconditioner import SyntheticKLBFGS
from dp_kfac.recorder import KFACRecorder
import dp_kfac.privacy as privacy


@pytest.fixture
def setup():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    c = runner.configuration(True)
    device = torch.device('cuda:0')
    model = GradSampleModule(TinyCNN().to(device), loss_reduction='sum')
    return c, device, model


def test_private_identity_order_and_numeric_update(setup, monkeypatch):
    c, device, model = setup
    events = []
    def forbidden(*args):
        pytest.fail('Private training invoked curvature')
    monkeypatch.setattr(SyntheticKLBFGS, 'apply', forbidden)
    x, y = torch.randn(4,1,28,28,device=device), torch.arange(4,device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    original_norm = privacy._compute_per_sample_norms_squared
    def norm(*args):
        assert all(p.grad_sample is not None for p in model.parameters())
        events.append('norm')
        return original_norm(*args)
    monkeypatch.setattr(runner, '_compute_per_sample_norms_squared', norm)
    monkeypatch.setattr(privacy, '_compute_per_sample_norms_squared', norm)
    original_clip = privacy._compute_clip_factors
    def clip(*args):
        events.append('clipping')
        return original_clip(*args)
    monkeypatch.setattr(privacy, '_compute_clip_factors', clip)
    original_noise = torch.randn_like
    def noise(*args, **kwargs):
        events.append('noise')
        return original_noise(*args, **kwargs)
    monkeypatch.setattr(torch, 'randn_like', noise)
    original_step = optimizer.step
    def step():
        events.append('SGD')
        original_step()
    monkeypatch.setattr(optimizer, 'step', step)
    snapshots = []
    original_dp = runner.clip_and_noise_gradients
    def dp(model, sigma, bound, batch):
        samples = [p.grad_sample.clone() for p in model.parameters()]
        norms = sum(g.flatten(1).square().sum(1) for g in samples).sqrt()
        scales = (bound/(norms+1e-6)).clamp(max=1)
        snapshots.extend((p.detach().clone(), (g.flatten(1)*scales[:,None]).mean(0).reshape_as(p)) for p,g in zip(model.parameters(),samples))
        original_dp(model,sigma,bound,batch)
    monkeypatch.setattr(runner, 'clip_and_noise_gradients', dp)
    runner.private_step(model,optimizer,x,y,c,0.)
    assert events[:3] == ['norm','norm','clipping']
    assert set(events[3:-1]) == {'noise'} and events[-1] == 'SGD'
    for p,(old,g) in zip(model.parameters(),snapshots):
        torch.testing.assert_close(p,old-.5*g)


def test_five_checkpoints_shared_probes_memory_reset_rng(setup, monkeypatch):
    c, device, model = setup
    assert c['lbfgs']['factor_damping'] == .1 and c['kfac']['damping'] == .001
    state = Diagnostic(c, device, TinyCNN)
    reference = SyntheticKLBFGS(c, device)
    theta = copy.deepcopy(model._module.state_dict())
    base_pair = state._synthetic_pair
    collect = state.collect
    seen = []
    current = []
    def pair(probe,x,y):
        for n,v in probe.state_dict().items():
            assert torch.equal(v,theta[n])
        current[:] = [x.clone(),y.clone()]
        reference_probe = TinyCNN().to(device)
        reference_probe.load_state_dict(theta)
        reference._synthetic_pair(reference_probe,x,y)
        base_pair(probe,x,y)
    def capture(probe,recorder,x,y):
        assert torch.equal(x,current[0]) and torch.equal(y,current[1])
        seen.append(1)
        collect(probe,recorder,x,y)
    monkeypatch.setattr(state,'_synthetic_pair',pair)
    monkeypatch.setattr(state,'collect',capture)
    monkeypatch.setattr(runner,'load_data',lambda *args: pytest.fail('Diagnostic accessed private data'))
    identities = None
    for epoch in range(1,6):
        cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        state.refresh(model,epoch)
        assert torch.equal(cpu,torch.get_rng_state())
        assert torch.equal(gpu,torch.cuda.get_rng_state(device))
        for n,v in model._module.state_dict().items():
            assert torch.equal(v,theta[n])
        ids = [id(f) for pair in state.factors.values() for f in pair]
        if identities is not None:
            assert ids == identities
        identities = ids
        for name,pair in state.factors.items():
            for f,ref in zip(pair,reference.factors[name]):
                assert f.accepted == ref.accepted
                n = state.fisher[name][0 if f is pair[0] else 1].shape[0]
                eye = torch.eye(n,device=device,dtype=torch.float64)
                torch.testing.assert_close(f.hv(eye),ref.hv(eye))
    assert len(seen) == 15


@pytest.mark.parametrize('side_index',[0,1])
def test_dense_geometry(setup, side_index):
    c, device, model = setup
    state = Diagnostic(c,device,TinyCNN)
    state.refresh(model,1)
    for layer in state.factors:
        fisher, factor = state.fisher[layer][side_index], state.factors[layer][side_index]
        metrics, gv, ops, transformed, dense = geometry(fisher,factor,.001,123)
        Q,B,M = dense['Q'],dense['B'],dense['M']
        eye = torch.eye(len(Q),device=device,dtype=torch.float64)
        torch.testing.assert_close(Q,factor.hv(eye))
        torch.testing.assert_close(Q,Q.T)
        assert (torch.linalg.eigvalsh(Q)>0).all()
        torch.testing.assert_close(B@Q,eye,atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(M,M.T)
        assert (gv>0).all() and torch.isfinite(M).all()
        fvals, fvecs = torch.linalg.eigh(fisher+.001*eye)
        W = (fvecs*fvals.pow(-.5))@fvecs.T
        torch.testing.assert_close(M,W@B@W,atol=1e-7,rtol=1e-7)
        for q in (.25,.5):
            P = factor.power(eye,q)
            torch.testing.assert_close(P,dense[f'Q_power_{q}'],atol=1e-8,rtol=1e-7)
            torch.testing.assert_close(dense[f'F_transformed_{q}'],P.T@fisher@P,atol=1e-7,rtol=1e-7)
        assert np.isfinite(list(metrics.values())).all()
        assert all(np.isfinite(list(r.values())).all() for r in ops+transformed)


def test_five_checkpoint_csv_protocol(setup):
    import pandas as pd
    c, device, model = setup
    state = Diagnostic(c,device,TinyCNN)
    directory = ROOT/'exp17/results/test_five_checkpoints'
    directory.mkdir(parents=True,exist_ok=True)
    for p in directory.glob('*.csv'):
        p.unlink()
    for epoch in range(1,6):
        state.refresh(model,epoch)
        save_checkpoint(state,directory,epoch,(epoch-1)*2)
    for name, count in [('factors',40),('operator_metrics',80),('transformed_conditions',80)]:
        frame = pd.read_csv(directory/f'{name}.csv')
        assert len(frame) == count and set(frame.epoch) == set(range(1,6))
        assert np.isfinite(frame.select_dtypes('number')).all().all()
    frame = pd.read_csv(directory/'generalized_spectra.csv')
    assert np.isfinite(frame.select_dtypes('number')).all().all()

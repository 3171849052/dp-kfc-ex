"""Small dense references for invariants that affect the research conclusions."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import pytest
import torch
from unittest.mock import patch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from exp18.operators import Factor, Operator
from exp18.lowrank import Sketch
from exp18.builders import build, synthetic_cache
from exp18.config import METHODS, rebuild, DAMPING, BETA
from exp12.curvature import estimate
from exp13.ghost import GhostNorm, ghost_aggregate, noise_and_step
from exp13.run_exp13 import initialize

torch.set_num_threads(4)


def dense_power(f):
    e, q = torch.linalg.eigh(f.double())
    return ((q*(e.clamp_min(0)+DAMPING).pow(-BETA))@q.T).float()


@pytest.mark.parametrize('kind', ['identity','diagonal','dense','lowrank'])
def test_power_and_kronecker(kind):
    torch.manual_seed(7)
    def make(d):
        e = torch.linspace(.2, 2, d)
        q = torch.linalg.qr(torch.randn(d,d)).Q
        if kind == 'identity':
            return Factor(kind,d), torch.eye(d)
        if kind == 'diagonal':
            return Factor(kind,d,e), dense_power(torch.diag(e))
        if kind == 'dense':
            f = (q*e)@q.T
            return Factor(kind,d,f), dense_power(f)
        u, top, tau = q[:,:2], e[-2:], .1
        f = (u*top)@u.T + tau*(torch.eye(d)-u@u.T)
        return Factor(kind,d,top,u,tau), dense_power(f)
    a, pa = make(5); c, pc = make(3)
    x = torch.randn(2,5,7)
    torch.testing.assert_close(a.action(x),pa@x,atol=2e-5,rtol=2e-5)
    op = Operator({'layer':dict(A=a,C=c)})
    g = torch.randn(3,5)
    torch.testing.assert_close(op.transform_matrix('layer',g),op.scale*pc@g@pa,atol=2e-5,rtol=2e-5)
    # Explicit row-major Kronecker action.
    torch.testing.assert_close(op.transform_matrix('layer',g).flatten(),
        op.scale*torch.kron(pc.contiguous(),pa.contiguous())@g.flatten(),atol=2e-5,rtol=2e-5)
    d = op.diagnostics
    assert d['scale_match']**2*d['predicted_second_moment_raw'] == pytest.approx(d['predicted_second_moment_ref'])


def test_lowrank_stream_and_moment():
    torch.manual_seed(3)
    x = torch.randn(43,19,dtype=torch.float64)
    sketch = Sketch(19,4,'cpu',torch.Generator().manual_seed(9))
    omega = sketch.omega.clone()
    # Fail if the actual sketch operations produce any full-dimension matrix.
    class NoCovariance(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for t in result if isinstance(result,(tuple,list)) else [result]:
                if isinstance(t,torch.Tensor):
                    assert t.shape != (19,19)
            return result
    with NoCovariance():
        for chunk in x.split(7): sketch.first(chunk)
        sketch.prepare()
        for chunk in x.split(7): sketch.second(chunk)
        factor, diag = sketch.finish()
    cov = x.T@x/len(x)
    q = torch.linalg.qr(cov@omega).Q
    e, v = torch.linalg.eigh(q.T@cov@q)
    e, u = e[-4:], q@v[:,-4:]
    tau = (cov.trace()-e.sum())/15
    reconstructed = (u*e)@u.T+tau*(torch.eye(19)-u@u.T)
    probe = torch.randn(19,3)
    torch.testing.assert_close(factor.action(probe),dense_power(reconstructed)@probe,atol=2e-5,rtol=2e-5)
    for beta in [.25,.5]:
        spectrum = torch.linalg.eigvalsh(reconstructed)
        assert factor.moment(beta) == pytest.approx((spectrum*(spectrum+DAMPING).pow(-2*beta)).sum().item())
    assert diag['tau'] == pytest.approx(tau.item())
    exact = Sketch(3,4,'cpu',torch.Generator().manual_seed(9))
    exact.first(x[:,:3]); exact.prepare(); exact.second(x[:,:3])
    f, diag = exact.finish()
    torch.testing.assert_close(f.action(torch.eye(3)),dense_power(x[:,:3].T@x[:,:3]/len(x)))
    assert diag['rank'] == 3


@pytest.mark.parametrize('method', METHODS)
def test_builder_and_global_ghost(method):
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.ReLU(),torch.nn.Linear(7,3))
    cache = [torch.randn(4,5), torch.randn(4,5)]
    reference, _ = estimate(model,cache,'KFAC-U',seed=20043)
    op, stats, diag = build(model,method,cache,42,1)
    # Exact compressed factors versus the original estimator's statistics.
    if not method.startswith('rank'):
        kinds = {'diag':('diagonal','diagonal'),'a_only':('dense','identity'),
                 'c_only':('identity','dense'),'fullA_diagC':('dense','diagonal'),
                 'diagA_fullC':('diagonal','dense'),'refresh2':('dense','dense'),'frozen':('dense','dense')}
        fs = {n:{s:Factor(k,len(f[s]), f[s].diag() if k=='diagonal' else f[s] if k=='dense' else None)
                 for s,k in zip(('A','C'),kinds[method])} for n,f in reference.items()}
        refop = Operator(fs)
        for n,f in reference.items():
            g = torch.randn(f['C'].shape[0],f['A'].shape[0])
            torch.testing.assert_close(op.transform_matrix(n,g),refop.transform_matrix(n,g),atol=2e-5,rtol=2e-5)
    x, y = torch.randn(3,5), torch.tensor([0,1,2])
    explicit, norms = [], []
    for xi,yi in zip(x,y):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xi[None]),yi[None]).backward()
        op.transform_aggregate_gradient(model)
        g = torch.cat([p.grad.flatten() for p in model.parameters()])
        norms.append(g.norm()); explicit.append(g)
    norms = torch.stack(norms)
    expected = (torch.stack(explicit)*(1/(norms+1e-6)).clamp(max=1)[:,None]).sum(0)
    hooks = GhostNorm(model,op)
    _, actual_norms, factors = ghost_aggregate(model,hooks,x,y)
    hooks.remove()
    torch.testing.assert_close(actual_norms,norms,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(torch.cat([p.grad.flatten() for p in model.parameters()]),expected,atol=3e-5,rtol=3e-5)
    assert all(getattr(p,'grad_sample',None) is None for p in model.parameters())
    assert stats['builder_forward_calls'] == (4 if method.startswith('rank') else 2)


@pytest.mark.parametrize('sigma',[0.,1.1])
def test_noise_then_divide_then_sgd(sigma):
    model = torch.nn.Linear(4,3)
    optimizer = torch.optim.SGD(model.parameters(),lr=.5)
    old = [p.detach().clone() for p in model.parameters()]
    grads = [torch.randn_like(p) for p in model.parameters()]
    rng = torch.Generator().manual_seed(40042)
    expected_rng = torch.Generator().manual_seed(40042)
    expected = []
    for p,g,o in zip(model.parameters(),grads,old):
        p.grad = g.clone()
        noise = torch.randn(p.numel(),generator=expected_rng).reshape_as(p)
        expected.append(o-.5*(g+sigma*noise)/3)
    noise_and_step(model,optimizer,sigma,3,rng)
    for p,e in zip(model.parameters(),expected): torch.testing.assert_close(p,e)
    assert torch.equal(rng.get_state(),expected_rng.get_state())


def test_lazy_and_rng():
    from exp18.run_exp18 import Adapter
    device = torch.device('cuda:0')
    model = initialize(42,device)
    initial = [p.detach().clone() for p in model.parameters()]
    cpu = torch.random.get_rng_state(); cuda = torch.cuda.get_rng_state_all()
    noise = torch.Generator(device=device).manual_seed(40042)
    noise_state = noise.get_state()
    for method in METHODS:
        cache = synthetic_cache(42,1,device,1,2)
        build(model,method,cache,42,1)
        assert torch.equal(cpu,torch.random.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        assert torch.equal(noise_state,noise.get_state())
    for p,q in zip(initial,initialize(42,device).parameters()): torch.testing.assert_close(p,q)
    for method,expected in [('refresh2',[0,2,4]),('frozen',[0])]:
        assert [e for e in range(5) if rebuild(method,e)] == expected
        adapter = Adapter(method)
        seen, last = [], None
        for epoch in range(1,6):
            cache = adapter.cache(42,epoch,device,1,2)
            op, stats, _ = adapter.builder(model,method,cache,42,epoch)
            if stats['rebuilt']: seen.append(epoch-1)
            else:
                assert op is last
                assert not cache and stats['builder_vjp_calls'] == 0
            last = op
        assert seen == expected


def test_diag_never_factorizes():
    model = torch.nn.Sequential(torch.nn.Linear(5,3))
    with patch('torch.linalg.eigh', side_effect=AssertionError('diagonal eigendecomposition')), \
         patch('torch.linalg.eigvalsh', side_effect=AssertionError('diagonal eigendecomposition')):
        build(model,'diag',[torch.randn(4,5)],42,1)


@pytest.mark.parametrize('method',['diag','rank4'])
def test_convolution_ghost(method):
    from exp12.runtime import runtime
    device = torch.device('cuda:0')
    with runtime(device):
        model = initialize(42,device)
        cache = synthetic_cache(42,1,device,1,4)
        op, _, _ = build(model,method,cache,42,1)
        x, y = cache[0][:2], torch.tensor([1,2],device=device)
        grads, norms = [], []
        for xi,yi in zip(x,y):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(xi[None]),yi[None]).backward()
            op.transform_aggregate_gradient(model)
            g = torch.cat([p.grad.flatten() for p in model.parameters()])
            norms.append(g.norm()); grads.append(g)
        norms = torch.stack(norms)
        expected = (torch.stack(grads)*(1/(norms+1e-6)).clamp(max=1)[:,None]).sum(0)
        hooks = GhostNorm(model,op)
        _, actual, _ = ghost_aggregate(model,hooks,x,y)
        hooks.remove()
        torch.testing.assert_close(actual,norms,rtol=3e-4,atol=2e-5)
        torch.testing.assert_close(torch.cat([p.grad.flatten() for p in model.parameters()]),expected,rtol=3e-4,atol=3e-6)

import copy
import torch
import pytest
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from opacus.accountants import RDPAccountant
from exp25 import config as c
from exp25.model import make_model
from exp25.methods import aggregate, first_pass, update
from exp25.geometry import build
from exp25.data import rng, private_loader, auxiliary


torch.set_num_threads(2)


def toy():
    torch.manual_seed(99)
    model = nn.Module()
    model.backbone = nn.Identity()
    model.classifier = nn.Linear(4, 3).double()
    x = torch.randn(6, 4, dtype=torch.double)
    y = torch.tensor([0, 2, 1, 2, 0, 1])
    return model, x, y


def reference(model, x, y, operator):
    samples = []
    for xi, yi in zip(x, y):
        loss = F.cross_entropy(model.classifier(xi[None]), yi[None])
        w, b = torch.autograd.grad(loss, tuple(model.classifier.parameters()))
        gradient = torch.cat((w, b[:, None]), dim=1)
        if operator is not None:
            gradient = operator.transform_gradient('classifier', gradient)
        samples.append(gradient)
    samples = torch.stack(samples)
    norms = samples.flatten(1).norm(dim=1)
    factors = (c.CLIP/(norms+1e-6)).clamp(max=1)
    return norms, factors, (samples*factors[:, None, None]).sum(0)


def test_crossvit_freeze():
    model = make_model(42, pretrained=False)
    assert {n for n,p in model.named_parameters() if p.requires_grad} == {'classifier.weight','classifier.bias'}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 28900
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert model(torch.zeros(2,3,240,240)).shape == (2,100)
    other = make_model(42, pretrained=False)
    torch.testing.assert_close(model.classifier.weight, other.classifier.weight, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['identity','a_only','full'])
def test_reference_and_noiseless_adam(kind):
    model, x, y = toy()
    operator, diagnostics = build(model, x, y, kind)
    expected_norms, expected_factors, expected_sum = reference(model, x, y, operator)
    losses, summed, norms, factors, count = aggregate(model, x, y, operator)
    assert count == 0
    assert all(p.grad is None for p in model.classifier.parameters())
    torch.testing.assert_close(norms, expected_norms)
    torch.testing.assert_close(factors, expected_factors)
    torch.testing.assert_close(summed, expected_sum)
    assert not any(hasattr(p, 'grad_sample') for p in model.parameters())
    ref = copy.deepcopy(model)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=c.LR)
    refopt = torch.optim.Adam(ref.classifier.parameters(), lr=c.LR)
    ref.classifier.weight.grad = expected_sum[:,:-1]/len(y)
    ref.classifier.bias.grad = expected_sum[:,-1]/len(y)
    refopt.step()
    acc = RDPAccountant()
    update(model,opt,summed,0.,len(y),rng(42,'dp_noise'),acc)
    for p,q in zip(model.parameters(),ref.parameters()):
        torch.testing.assert_close(p,q)
    assert sum(h[2] for h in acc.history)==1


def test_anchor_disables_real_parameters():
    model,x,y=toy()
    seen=[]
    handle=model.classifier.register_forward_hook(
        lambda module,args,out: seen.append((all(not p.requires_grad for p in module.parameters()),out.requires_grad)))
    first_pass(model,x,y)
    handle.remove()
    assert seen==[(True,False)]
    assert all(p.requires_grad and p.grad is None for p in model.classifier.parameters())


def test_rms_and_full_factors():
    model,x,y=toy()
    aop,diag=build(model,x,None,'a_only')
    a=torch.cat((x,torch.ones_like(x[:,:1])),1)
    A=a.T@a/len(a)
    eig=torch.linalg.eigvalsh(A).clamp_min(0)
    scale=((eig/(eig+c.DAMPING)).sum()/(eig*(eig+c.DAMPING).pow(-.8)).sum()).sqrt()
    assert diag['p']==.4
    assert diag['rms_scale']==pytest.approx(scale.item())
    full,_=build(model,x,y,'full')
    b=model.classifier(x).softmax(1)-F.one_hot(y,3)
    torch.testing.assert_close(full.factors['classifier']['G'], b.T@b/len(b))
    torch.testing.assert_close(full.factors['classifier']['A'],A)


def test_privacy_schedule():
    assert c.TOTAL_STEPS==975 and c.STEPS_PER_EPOCH==195
    assert c.BATCH_SIZE==256 and c.SAMPLE_RATE==256/50000
    sigma=c.noise_multiplier()
    acc=RDPAccountant()
    for _ in range(c.TOTAL_STEPS):
        acc.step(noise_multiplier=sigma,sample_rate=c.SAMPLE_RATE)
    assert sum(h[2] for h in acc.history)==975
    assert 7.999 <= acc.get_epsilon(c.DELTA) <= 8.


def test_rng_pairing_and_auxiliary_labels():
    private=TensorDataset(torch.arange(600).float()[:,None],torch.arange(600)%100)
    public=TensorDataset(torch.arange(600).float()[:,None],torch.arange(600)%10)
    baseline=private_loader(private,42,1).dataset.indices
    for method in c.CONDITIONS:
        kind,source=c.condition(method)
        if source in ('public','oracle'):
            x,y=auxiliary(source,private,public,42,1,'cpu')
            assert len(x)==256
            assert torch.equal(y,x[:,0].long()%(10 if source=='public' else 100))
        assert private_loader(private,42,1).dataset.indices==baseline
    assert not torch.equal(torch.rand(10,generator=rng(42,'dp_noise')),torch.rand(10,generator=rng(42,'private')))
    x,y=auxiliary('pink',private,public,42,1,'cpu')
    x2,y2=auxiliary('pink',private,public,42,1,'cpu')
    torch.testing.assert_close(x,x2,rtol=0,atol=0)
    assert torch.equal(y,y2) and y.min()>=0 and y.max()<100
    assert x.shape==(256,3,240,240)


def test_seed_stream_independence():
    g=rng(7,'dp_noise')
    first=torch.randn(5,generator=g)
    torch.randn(100,generator=rng(7,'oracle'))
    torch.testing.assert_close(first,torch.randn(5,generator=rng(7,'dp_noise')))


def test_gaussian_noise_and_adam_reference():
    model,x,y=toy()
    ref=copy.deepcopy(model)
    _,summed,_,_,_=aggregate(model,x,y)
    sigma=.7
    noise=torch.randn(summed.shape,dtype=summed.dtype,generator=rng(23,'dp_noise'))*sigma*c.CLIP
    expected=(summed+noise)/len(y)
    ref.classifier.weight.grad=expected[:,:-1].contiguous()
    ref.classifier.bias.grad=expected[:,-1].contiguous()
    refopt=torch.optim.Adam(ref.classifier.parameters(),lr=c.LR)
    refopt.step()
    opt=torch.optim.Adam(model.classifier.parameters(),lr=c.LR)
    acc=RDPAccountant()
    update(model,opt,summed,sigma,len(y),rng(23,'dp_noise'),acc)
    for p,q in zip(model.parameters(),ref.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert acc.history==[(sigma,c.SAMPLE_RATE,1)]

"""Memory-route assertions and independent sample-backward oracle coverage."""
import copy
import pytest
import torch
from torch import nn
from exp21.bk import BookKeeping
from exp21.handlers import Record
from exp21.routing import norm_route, register_fallback
from exp21.test_transformer_cases import assert_result


@pytest.mark.parametrize('tokens,tile', [(11,4), (9,1), (7,32), (1025,64)])
def test_row_ghost(tokens, tile, monkeypatch):
    model = nn.Linear(13, 17).cuda()
    r = Record('', model, torch.randn(2,tokens,13,device='cuda'),
               torch.randn(2,tokens,17,device='cuda'), None, {id(p) for p in model.parameters()})
    expected = sum(g.flatten(1).square().sum(1) for g in r.fast().values())
    def forbidden(*a, **kw):
        pytest.fail('Ghost called dense gradient sampler')
    monkeypatch.setattr(Record, 'sample', forbidden)
    monkeypatch.setattr(Record, 'fast', forbidden)
    torch.testing.assert_close(r.ghost(tile), expected, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize('tokens,cap,expected,reason', [(3,1,'ghost','ghost_memory_cap'),
    (128,2**30,'fast','fast_compute'), (1,2**30,'ghost','ghost_compute')])
def test_memory_compute_router(tokens, cap, expected, reason):
    m = nn.Linear(32,48).cuda()
    r = Record('',m,torch.randn(4,tokens,32,device='cuda'),torch.randn(4,tokens,48,device='cuda'),None,{id(p) for p in m.parameters()})
    result = norm_route(r,'auto',cap)
    assert result['strategy'] == expected and result['routing_reason'] == reason
    assert result['estimated_fast_bytes'] == 4*33*48*4


class Packed(nn.Module):
    def __init__(self):
        super().__init__()
        self.packed = nn.Parameter(torch.randn(5,5)*.1)
    def forward(self, x):
        return x @ self.packed


register_fallback(Packed)


@pytest.mark.parametrize('chunk', [1,2,4])
@pytest.mark.parametrize('method', ['bk','bk_gd'])
def test_local_fallback(chunk, method, monkeypatch):
    from exp21.test_exp21 import golden, loss
    m = nn.Sequential(nn.Linear(4,5), Packed(), nn.Tanh(), nn.Linear(5,3)).cuda()
    x,y = torch.randn(5,4,device='cuda'),torch.randn(5,3,device='cuda')
    expected = golden(copy.deepcopy(m),x,y)
    h = BookKeeping(m,fallback_vjp_chunk_size=chunk)
    original = torch.autograd.grad
    chunks, final_targets = [], []
    def grad(outputs, inputs, **kw):
        if kw.get('is_grads_batched'):
            chunks.append(kw['grad_outputs'].shape[0])
            assert list(inputs) == [m[1].packed]
        elif not kw.get('retain_graph'):
            final_targets.extend(inputs)
        return original(outputs,inputs,**kw)
    monkeypatch.setattr(torch.autograd,'grad',grad)
    result = h.aggregate(x,y,method,loss_fn=loss)
    assert_result(m,result,expected)
    stats=result[-1]
    assert max(chunks)<=chunk and sum(chunks)==len(x)
    assert final_targets == [m[1].packed]
    assert stats['fallback_parameter_count']==25
    assert stats['fallback_temporary_grad_bytes']==min(chunk,len(x))*25*4
    assert set(stats['layer_strategies'])=={'0','3'}
    assert stats['first_pass_parameter_grad_count']==0
    h.remove()


@pytest.mark.parametrize('method',['fast2','ghost2'])
def test_streaming_baseline(method,monkeypatch):
    from exp21.test_exp21 import golden,loss
    m=nn.Sequential(nn.Linear(4,6),nn.Tanh(),nn.Linear(6,3)).cuda()
    x,y=torch.randn(3,9,4,device='cuda'),torch.randn(3,9,3,device='cuda')
    expected=golden(copy.deepcopy(m),x,y)
    h=BookKeeping(m,strategy='fast' if method=='fast2' else 'ghost')
    original=h.norms
    def norms(stats):
        assert not h.records and not h.pending
        assert all(p.grad is None for p in m.parameters())
        return original(stats)
    monkeypatch.setattr(h,'norms',norms)
    result=h.aggregate(x,y,method,loss_fn=loss)
    assert_result(m,result,expected)
    assert result[-1]['bk_cache_bytes']==0 and not result[-1]['gd_applied']
    assert result[-1]['first_pass_param_grad_disabled']
    h.remove()


@pytest.mark.parametrize('padding',[None,0])
def test_vector_embedding(padding,monkeypatch):
    from exp21.test_exp21 import golden,loss
    m=nn.Embedding(53,7,padding_idx=padding).cuda()
    x=torch.tensor([[0,2,2,4],[0,0,0,0],[5,5,3,5]],device='cuda')
    y=torch.randn(3,4,7,device='cuda')
    expected=golden(copy.deepcopy(m),x,y)
    def forbidden(*a,**kw):
        pytest.fail('Embedding used per-sample sparse tensor')
    monkeypatch.setattr(torch,'sparse_coo_tensor',forbidden)
    monkeypatch.setattr(Record,'sample',forbidden)
    h=BookKeeping(m)
    assert_result(m,h.aggregate(x,y,'bk_gd',loss_fn=loss),expected)
    h.remove()


class LargeTie(nn.Module):
    def __init__(self,vocab,dim):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim,padding_idx=0)
        self.head=nn.Linear(dim,vocab,bias=False)
        self.head.weight=self.emb.weight
    def forward(self,x):
        return self.head(self.emb(x))


@pytest.mark.parametrize('vocab,dim,tokens,cap,choice',[
    (53,13,3,4096,'ghost_tied'), (53,3,11,4096,'chunked_fast_tied'),
    (50003,3,9,4096,'chunked_fast_tied'), (50003,13,3,16,'ghost_tied')])
def test_tied_workspace(vocab,dim,tokens,cap,choice,monkeypatch):
    from exp21.test_exp21 import golden,loss
    m=LargeTie(vocab,dim).cuda()
    x=torch.randint(0,5,(2,tokens),device='cuda');x[:,0]=0
    y=torch.randn(2,tokens,vocab,device='cuda')*.01
    expected=golden(copy.deepcopy(m),x,y)
    def forbidden(*a,**kw):
        pytest.fail('Tied dense sample gradient or sparse per-example loop')
    monkeypatch.setattr(Record,'fast',forbidden)
    monkeypatch.setattr(Record,'sample',forbidden)
    h=BookKeeping(m,max_fast_temp_bytes=cap,tied_output_chunk_size=7,tile=2)
    result=h.aggregate(x,y,'bk_gd',loss_fn=loss)
    assert_result(m,result,expected)
    assert result[-1]['layer_strategies']['head']==choice
    assert result[-1]['temporary_per_sample_grad_bytes'] < 2*vocab*dim*4/4
    h.remove()


@pytest.mark.parametrize('reuse_only',[False,True])
def test_shared_guard(reuse_only,monkeypatch):
    from exp21.test_exp21 import golden,loss,Reused
    class Reuse(nn.Module):
        def __init__(self):
            super().__init__();self.a=nn.Linear(4,4)
        def forward(self,x):
            return self.a(x)+self.a(x.sin())
    m=(Reuse() if reuse_only else Reused()).cuda()
    x,y=torch.randn(3,2,4,device='cuda'),torch.randn(3,2,4,device='cuda')
    expected=golden(copy.deepcopy(m),x,y)
    def forbidden(*a,**kw):
        pytest.fail('Shared route exceeded sample memory guard')
    monkeypatch.setattr(Record,'sample',forbidden)
    h=BookKeeping(m,max_shared_sample_bytes=1,fallback_vjp_chunk_size=1)
    result=h.aggregate(x,y,'bk_gd',loss_fn=loss)
    assert_result(m,result,expected)
    assert result[-1]['fallback_parameter_count']==sum(p.numel() for p in m.parameters())
    assert result[-1]['fallback_temporary_grad_bytes']==sum(p.numel()*4 for p in m.parameters())
    h.remove()


@pytest.mark.parametrize('fallback',[False,True])
def test_optimized_repeated_memory(fallback):
    from exp21.test_exp21 import loss
    if fallback:
        m=nn.Sequential(nn.Linear(4,5),Packed(),nn.Linear(5,3)).cuda()
        x,y=torch.randn(5,129,4,device='cuda'),torch.randn(5,129,3,device='cuda')
    else:
        m=LargeTie(50003,3).cuda()
        x=torch.randint(0,9,(2,17),device='cuda')
        y=torch.zeros(2,17,50003,device='cuda')
    h=BookKeeping(m,tied_output_chunk_size=64,fallback_vjp_chunk_size=2)
    memory,temporary=[],[]
    for _ in range(7):
        result=h.aggregate(x,y,'bk_gd',loss_fn=loss)
        temporary.append(result[-1]['fallback_temporary_grad_bytes'])
        assert not h.records and not h.pending and not h.anchor_tensors
        del result
        m.zero_grad(set_to_none=True)
        memory.append(torch.cuda.memory_allocated())
    assert max(memory[1:])-min(memory[1:])<=4096
    assert len(set(temporary))==1
    h.remove()


def test_dropout_fallback_replay():
    from exp21.test_exp21 import loss
    from exp21.fallback import rng_state,replay_rng
    m=nn.Sequential(nn.Linear(4,5),nn.Dropout(.4),Packed(),nn.Linear(5,3)).cuda()
    ref=copy.deepcopy(m)
    x,y=torch.randn(5,4,device='cuda'),torch.randn(5,3,device='cuda')
    state=rng_state(x.device)
    # Independent scalar backward on the same stochastic batch graph preserves
    # each example's dropout mask; no batched VJP in the reference.
    losses=loss(ref(x),y)
    samples=[]
    for i in range(len(x)):
        ref.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=i+1<len(x))
        samples.append({n:p.grad.clone() for n,p in ref.named_parameters()})
    norms=torch.stack([sum(g.square().sum() for g in row.values()).sqrt() for row in samples])
    factors=(1/(norms+1e-6)).clamp(max=1)
    grads={n:sum(factors[i]*samples[i][n] for i in range(len(x))) for n,_ in ref.named_parameters()}
    h=BookKeeping(m,fallback_vjp_chunk_size=2)
    with replay_rng(state,x.device):
        result=h.aggregate(x,y,'bk_gd',loss_fn=loss)
    assert_result(m,result,(norms,factors,grads))
    h.remove()


@pytest.mark.parametrize('freeze',['weight','bias'])
def test_frozen_affine_memory_cap(freeze):
    from exp21.test_exp21 import golden,loss
    m=nn.Linear(4,5).cuda()
    getattr(m,freeze).requires_grad_(False)
    x,y=torch.randn(3,7,4,device='cuda'),torch.randn(3,7,5,device='cuda')
    expected=golden(copy.deepcopy(m),x,y)
    h=BookKeeping(m,max_fast_temp_bytes=1)
    result=h.aggregate(x,y,'bk_gd',loss_fn=loss)
    assert_result(m,result,expected)
    assert result[-1]['layer_routing']['']['routing_reason']=='ghost_memory_cap'
    assert not getattr(m,freeze).requires_grad
    h.remove()


def test_ghost_conv_cap():
    from exp21.test_exp21 import golden,loss
    m=nn.Conv2d(2,3,3,padding=1).cuda()
    x,y=torch.randn(3,2,5,5,device='cuda'),torch.randn(3,3,5,5,device='cuda')
    expected=golden(copy.deepcopy(m),x,y)
    h=BookKeeping(m,max_fast_temp_bytes=1,tile=4)
    result=h.aggregate(x,y,'bk_gd',loss_fn=loss)
    assert_result(m,result,expected)
    assert result[-1]['layer_routing']['']['estimated_fast_bytes']==3*(3*2*3*3+3)*4
    assert result[-1]['layer_strategies']['']=='ghost'
    h.remove()


def test_norm_allocation_shapes(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from exp21.test_exp21 import loss
    class AllocationGuard(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            tensors=out if isinstance(out,(tuple,list)) else [out]
            for tensor in tensors:
                if isinstance(tensor,torch.Tensor):
                    assert tuple(tensor.shape) not in ((2,50003,3),(50003,3),(2,17,17))
            return out
    m=LargeTie(50003,3).cuda()
    h=BookKeeping(m,tile=64,tied_output_chunk_size=7)
    original=h.norms
    def norms(stats):
        with AllocationGuard():
            return original(stats)
    monkeypatch.setattr(h,'norms',norms)
    h.aggregate(torch.randint(0,5,(2,17),device='cuda'),torch.zeros(2,17,50003,device='cuda'),'bk_gd',loss_fn=loss)
    h.remove()
    # Force row Ghost with tile > T under the same full-Gram guard.
    m=nn.Linear(3,9).cuda()
    r=Record('',m,torch.randn(2,17,3,device='cuda'),torch.randn(2,17,9,device='cuda'),None,{id(p) for p in m.parameters()})
    with AllocationGuard():
        r.ghost(64)

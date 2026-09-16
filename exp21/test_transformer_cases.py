"""Additional tests imported into test_exp21.py; no pretrained weights/network."""
# Helpers are provided by test_exp21's module namespace through explicit imports.
import copy
import json
import weakref
from pathlib import Path
import pytest
import torch
from torch import nn
from exp21.bk import BookKeeping
from exp21.handlers import Record
from exp21.methods import Clipper, HybridBKClipper, build_from_cache
from exp21.config import METHODS
from exp21.routing import register_fallback


def assert_result(model, result, expected):
    norms, factors, grads = expected
    torch.testing.assert_close(result[1], norms, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(result[2], factors, rtol=1e-4, atol=2e-5)
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, grads[name], rtol=1e-4, atol=3e-6, msg=lambda msg: name+': '+msg)


@pytest.mark.parametrize('method', ['bk', 'bk_gd', 'fast2', 'ghost2'])
@pytest.mark.parametrize('layer', ['linear', 'conv', 'hf'])
def test_corrected_nonidentity_geometry(method, layer):
    from exp21.test_exp21 import golden, loss
    from transformers.pytorch_utils import Conv1D
    if layer == 'linear':
        model, x = nn.Linear(4, 5, bias=False), torch.randn(3, 2, 4)
    elif layer == 'conv':
        model, x = nn.Conv2d(2, 3, 3, padding=1, bias=False), torch.randn(3, 2, 5, 5)
    else:
        model, x = Conv1D(5, 4), torch.randn(3, 2, 4)
    model, x = model.cuda(), x.cuda()
    y = torch.randn_like(model(x))
    op, _ = build_from_cache(model, .25, [x], 42, 1)
    expected_dim = 18 if layer == 'conv' else (5 if layer == 'hf' else 4)
    assert op.data[''].shape == (expected_dim, expected_dim)
    expected = golden(copy.deepcopy(model), x, y, op)
    clipper = Clipper(model, op, method)
    result = clipper.aggregate(x, y, loss_fn=loss)
    assert_result(model, result, expected)
    assert result[-1]['preconditioned_layers'] == ['']
    clipper.remove()


@pytest.mark.parametrize('method', METHODS)
def test_partial_geometry_and_nonunit_bound(method):
    from exp21.test_exp21 import golden, loss
    model = nn.Sequential(nn.Linear(4, 6, bias=False), nn.Tanh(), nn.Linear(6, 3)).cuda()
    x, y = torch.randn(5, 4, device='cuda'), torch.randn(5, 3, device='cuda')
    op, _ = build_from_cache(model, .25, [x], 42, 1, layer_names=['0'])
    expected = golden(copy.deepcopy(model), x, y, op, max_grad_norm=.37)
    clipper = Clipper(model, op, method, max_grad_norm=.37)
    result = clipper.aggregate(x, y, loss_fn=loss)
    assert_result(model, result, expected)
    assert result[-1]['preconditioned_layers'] == ['0']
    assert result[-1]['identity_geometry_layers'] == ['2']
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    generator = torch.Generator(device='cuda').manual_seed(987)
    reference_generator = torch.Generator(device='cuda').manual_seed(987)
    predicted = {}
    for name, p in model.named_parameters():
        noise = torch.randn(p.numel(), device=p.device, dtype=p.dtype, generator=reference_generator).reshape_as(p)
        predicted[name] = before[name]-.2*(expected[2][name]+.8*.37*noise)/len(x)
    clipper.step(torch.optim.SGD(model.parameters(), lr=.2), .8, len(x), generator)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, predicted[name], rtol=1e-4, atol=3e-6)
    assert torch.equal(generator.get_state(), reference_generator.get_state())
    clipper.remove()


def test_cnn_builder_matches_exp20():
    from exp21.run_one import initialize
    from exp20.methods import build_from_cache as old_builder
    model = initialize(42, 'cuda')
    x = torch.randn(4, 1, 28, 28, device='cuda')
    old, _ = old_builder(model, .25, [x], 42, 1)
    new, _ = build_from_cache(model, .25, [x], 42, 1)
    assert old.scale == new.scale
    for name in old.data:
        torch.testing.assert_close(new.data[name], old.data[name], rtol=0, atol=0)


@pytest.mark.parametrize('layer', ['linear', 'conv'])
def test_output_anchor_and_minimal_cache(layer):
    from exp21.test_exp21 import golden, loss
    model = (nn.Linear(4, 5) if layer == 'linear' else nn.Conv2d(2, 3, 3, padding=1)).cuda()
    x = torch.randn(*((3, 4) if layer == 'linear' else (3, 2, 5, 5)), device='cuda', requires_grad=True)
    y = torch.randn_like(model(x))
    op, _ = build_from_cache(model, .25, [x.detach()], 42, 1)
    expected = golden(copy.deepcopy(model), x.detach(), y, op)
    raw_refs, retained_refs = [], []
    touched, input_touched = [], []
    x.register_hook(lambda g: input_touched.append(True))
    handles = [p.register_hook(lambda g: touched.append(True)) for p in model.parameters()]
    capture = model.register_forward_pre_hook(lambda module, args: raw_refs.append(weakref.ref(args[0])))
    bk = BookKeeping(model, op)
    original = bk.reconstruct
    def reconstruction(factors):
        for record in bk.records:
            assert record.x is None
            assert record.z is not None and record.b is not None
            assert record.z.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            retained_refs.extend([weakref.ref(record.z), weakref.ref(record.b)])
        original(factors)
    bk.reconstruct = reconstruction
    result = bk.aggregate(x, y, 'bk_gd', loss_fn=loss)
    assert_result(model, result, expected)
    assert x.grad is None and input_touched == [] and touched == []
    assert result[-1]['gd_anchor_module'] == ''
    assert result[-1]['gd_anchor_modules'] == ['']
    assert result[-1]['first_pass_parameter_grad_count'] == 0
    assert all(r() is None for r in retained_refs)
    assert all(r() is None for r in raw_refs)
    for h in handles+[capture]:
        h.remove()
    bk.remove()


@pytest.mark.parametrize('method', ['bk', 'bk_gd'])
@pytest.mark.parametrize('layout', ['linear', 'hf'])
def test_tied_analytic_cross_term(method, layout, monkeypatch):
    from exp21.test_exp21 import golden, loss
    from transformers.pytorch_utils import Conv1D
    class TiedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            # HF [input,output] identity tie needs compatible square dimensions.
            vocab, hidden = (103, 7) if layout == 'linear' else (7, 7)
            self.emb = nn.Embedding(vocab, hidden, padding_idx=0)
            self.head = nn.Linear(hidden, vocab) if layout == 'linear' else Conv1D(vocab, hidden)
            self.head.weight = self.emb.weight
        def forward(self, x):
            return self.head(self.emb(x))
    model = TiedProjection().cuda()
    x = torch.tensor([[0,2,2,5],[3,3,3,3]], device='cuda')
    y = torch.randn_like(model(x))
    expected = golden(copy.deepcopy(model), x, y)
    sample, fast = Record.sample, Record.fast
    def guard_sample(record, i):
        assert record.kind == 'embedding', 'Tied head materialized [V,d] per sample'
        return sample(record, i)
    def guard_fast(record):
        assert record.kind != 'linear', 'Tied head materialized [B,V,d]'
        return fast(record)
    monkeypatch.setattr(Record, 'sample', guard_sample)
    monkeypatch.setattr(Record, 'fast', guard_fast)
    bk = BookKeeping(model, strategy='fast', tile=3)
    result = bk.aggregate(x, y, method, loss_fn=loss)
    assert_result(model, result, expected)
    assert result[-1]['layer_strategies']['head'] in ('ghost_tied', 'chunked_fast_tied')
    assert result[-1]['temporary_per_sample_grad_bytes'] <= 256*2**20
    assert not result[-1]['requires_second_backward']
    bk.remove()


def tiny_hf(name):
    from transformers import (BertConfig, BertForSequenceClassification, GPT2Config,
        GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM)
    if name == 'bert':
        config = BertConfig(vocab_size=23, hidden_size=12, num_hidden_layers=1,
            num_attention_heads=3, intermediate_size=20, max_position_embeddings=16,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0., num_labels=3)
        config._attn_implementation = 'eager'
        model = BertForSequenceClassification(config)
    elif name == 'gpt2':
        config = GPT2Config(vocab_size=23, n_embd=12, n_layer=1, n_head=3, n_positions=16,
            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., use_cache=False, tie_word_embeddings=True)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config)
    else:
        config = LlamaConfig(vocab_size=23, hidden_size=12, intermediate_size=20,
            num_hidden_layers=1, num_attention_heads=3, num_key_value_heads=3,
            max_position_embeddings=16, attention_dropout=0., use_cache=False,
            tie_word_embeddings=True)
        config._attn_implementation = 'eager'
        model = LlamaForCausalLM(config)
    return model.cuda()


def hf_loss(name):
    def per_example(output, target):
        if name == 'bert':
            return torch.nn.functional.cross_entropy(output.logits, target, reduction='none')
        logits = output.logits[:, :-1]
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), target[:, 1:], reduction='none').mean(1)
    return per_example


@pytest.mark.parametrize('name', ['bert', 'gpt2', 'llama'])
@pytest.mark.parametrize('method', ['bk', 'bk_gd'])
def test_tiny_transformer_integration(name, method):
    from exp21.test_exp21 import golden
    model = tiny_hf(name)
    x = torch.tensor([[2,3,3,5],[7,8,7,4]], device='cuda')
    y = torch.tensor([1,2], device='cuda') if name == 'bert' else x
    loss_fn = hf_loss(name)
    op, builder = build_from_cache(model, .25, [x], 42, 1)
    expected = golden(copy.deepcopy(model), x, y, op, loss_fn, max_grad_norm=.37)
    touched = []
    handles = [p.register_hook(lambda g: touched.append(True)) for p in model.parameters()]
    clipper = HybridBKClipper(model, op, method, max_grad_norm=.37)
    result = clipper.aggregate(x, y, loss_fn=loss_fn)
    assert_result(model, result, expected)
    stats = result[-1]
    assert stats['fallback_layer_count'] == 0 and stats['requires_second_backward'] is False
    assert stats['backward_calls'] == 1
    assert stats['preconditioned_layers'] and stats['identity_geometry_layers']
    if method == 'bk_gd':
        assert touched == [] and stats['first_pass_parameter_grad_count'] == 0
        assert stats['gd_anchor_module'].endswith('word_embeddings' if name == 'bert' else ('wte' if name == 'gpt2' else 'embed_tokens'))
    if name != 'bert':
        assert stats['layer_strategies']['lm_head'] == 'ghost_tied'
    error = max((p.grad-expected[2][n]).abs().max().item() for n, p in model.named_parameters())
    output = Path(__file__).parent/'results'/'integration'
    output.mkdir(exist_ok=True)
    (output/f'{name}_{method}.json').write_text(json.dumps(dict(model=name, method=method,
        gradient_max_abs_error=error, **stats), indent=2)+'\n')
    clipper.remove()
    for h in handles:
        h.remove()


@pytest.mark.parametrize('method', ['bk', 'bk_gd', 'exact'])
@pytest.mark.parametrize('geometry', ['identity', 'partial'])
def test_tinyvit_whole_step_fallback(method, geometry, monkeypatch):
    from exp21.test_exp21 import golden
    from dp_kfac.models import TinyViT
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from exp21.profiling import EventProfiler
    model = TinyViT(img_size=8, patch_size=4, embed_dim=8, num_heads=2, num_blocks=1, num_classes=3).cuda()
    x, y = torch.randn(2, 1, 8, 8, device='cuda'), torch.tensor([1,2], device='cuda')
    loss_fn = lambda z, target: torch.nn.functional.cross_entropy(z, target, reduction='none')
    op = None
    if geometry == 'partial':
        op, _ = build_from_cache(model, .25, [x], 42, 1, layer_names=['patch_embed', 'head'])
    with sdpa_kernel(SDPBackend.MATH):
        expected = golden(copy.deepcopy(model), x, y, op, loss_fn, max_grad_norm=.37)
    calls = []
    original_grad, original_backward = torch.autograd.grad, torch.Tensor.backward
    def grad(*a, **k):
        if k.get('is_grads_batched'):
            calls.append('batched_vjp')
            assert k['grad_outputs'].shape[0] <= clipper.hooks.fallback_vjp_chunk_size
            assert {id(p) for p in a[1]} == clipper.hooks.fallback_ids
        else:
            calls.append('reverse')
            if not k.get('retain_graph') and method != 'exact':
                assert {id(p) for p in a[1]} == clipper.hooks.fallback_ids
        return original_grad(*a, **k)
    def backward(tensor, *a, **k):
        calls.append('weighted_backward')
        return original_backward(tensor, *a, **k)
    clipper = HybridBKClipper(model, op, method, max_grad_norm=.37)
    prof = EventProfiler()
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, 'grad', grad)
        patch.setattr(torch.Tensor, 'backward', backward)
        result = clipper.aggregate(x, y, prof, loss_fn)
    assert calls == ['reverse', 'batched_vjp', 'reverse']
    assert_result(model, result, expected)
    torch.cuda.synchronize()
    timings = prof.resolve()
    stats = result[-1]
    assert timings['second_pass_seconds'] > 0
    assert stats['requires_second_backward'] and stats['fallback_layer_count'] > 0
    assert stats['backward_calls'] == 3
    if method != 'exact':
        assert timings['bk_reconstruction_seconds'] > 0
    assert stats['gd_applied'] == (method == 'bk_gd')
    assert stats['first_pass_parameter_grad_count'] == 0
    expected_fallback = {n for n, _ in model.named_parameters() if n == 'pos_embed' or n.startswith('blocks.1.')}
    assert set(stats['fallback_parameter_names']) == expected_fallback
    assert set(stats['fallback_layer_names']) == {'', 'blocks.1'}
    assert not clipper.hooks.pending and not clipper.hooks.records
    output = Path(__file__).parent/'results'/'integration'
    output.mkdir(exist_ok=True)
    (output/f'tinyvit_{method}_{geometry}.json').write_text(json.dumps(dict(**stats, timings=timings), indent=2)+'\n')
    clipper.remove()


def test_dangerous_not_silent_fallback():
    for module in [nn.BatchNorm1d(4), nn.Embedding(10, 4, max_norm=1.)]:
        with pytest.raises(NotImplementedError, match='unsupported-dangerous'):
            HybridBKClipper(module)

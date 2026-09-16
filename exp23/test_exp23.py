"""CUDA tests never launch the formal grid or smoke subprocesses."""
import math
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from exp12.runtime import runtime
from exp12.curvature import forward, covariance
from exp23 import config as cfg
from exp23.public_data import public_dataset, calibration
from exp23.geometry import build
from exp23.run_exp23 import initialize, private_loader, noise_generator
from exp19.methods import noise_and_normalize


@pytest.fixture(autouse=True)
def cuda_runtime():
    torch.set_num_threads(4)
    with runtime('cuda:0'):
        yield


@pytest.mark.parametrize('source', ['fashion', 'cifar10'])
def test_public_shape(source):
    data = public_dataset(source)
    x = torch.stack([data[i][0] for i in range(4)])
    assert x.shape == (4, 1, 28, 28)
    assert torch.isfinite(x).all()


def test_a_only_and_full_raw_a_and_rms():
    model = initialize(42, 'cuda:0')
    x = torch.randn(8, 1, 28, 28, device='cuda')
    y = torch.arange(8, device='cuda')
    with patch('torch.autograd.grad', side_effect=AssertionError('A-only VJP')), \
         patch('torch.Tensor.backward', side_effect=AssertionError('A-only backward')):
        op, a, counts = build(model, 'a_public_fashion', [(x, None)])
    full, f, full_counts = build(model, 'full_public_fashion', [(x, y)])
    for name in a:
        torch.testing.assert_close(a[name]['A'], f[name]['A'], rtol=0, atol=0)
        covariance_a = a[name]['A'].double()
        e, q = torch.linalg.eigh((covariance_a+covariance_a.T)/2)
        expected = ((q*(e.clamp_min(0)+.001).pow(-.4))@q.T).float()
        torch.testing.assert_close(op.data[name], expected)
        b = torch.randn(2, a[name]['output_dimension'], 3, device='cuda')
        assert op.transform_backprop(name, b) is b
        assert a[name]['A'].shape[0] == dict(model.named_modules())[name].weight[0].numel()+1
    assert op.power == .4
    assert math.isclose(op.scale**2*op.moments['m_p'], op.moments['m_reference'], rel_tol=1e-12)
    assert all(counts[k] == 0 for k in ('builder_backward_calls', 'builder_vjp_calls', 'builder_reverse_vectors'))
    assert counts['builder_forward_calls'] == 1
    assert full_counts['builder_vjp_calls'] == 1 and full_counts['builder_reverse_vectors'] == 8


@pytest.mark.parametrize('source', ['fashion', 'cifar10'])
def test_public_g_uses_true_labels(source):
    data = public_dataset(source)
    x = torch.stack([data[i][0] for i in range(8)]).cuda()
    y = torch.tensor([data[i][1] for i in range(8)], device='cuda')
    model = initialize(42, 'cuda:0')
    _, factors, _ = build(model, f'full_public_{source}', [(x, y)])
    logits, _, zs = forward(model, x)
    grads = torch.autograd.grad(F.cross_entropy(logits, y, reduction='sum'), tuple(zs.values()))
    for name, grad in zip(zs, grads):
        torch.testing.assert_close(factors[name]['C'], covariance(grad), atol=1e-7, rtol=1e-5)
    _, wrong, _ = build(model, f'full_public_{source}', [(x, (y+1)%10)])
    assert not torch.allclose(factors['fc2']['C'], wrong['fc2']['C'])


@pytest.mark.parametrize('seed', cfg.SEEDS)
def test_paired_rng_independent_of_method(seed):
    reference = None
    for method in cfg.METHODS:
        model = initialize(seed, 'cuda:0')
        initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        cpu, cuda = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        batches = [] if method == 'dp_sgd' else calibration(cfg.source(method), seed, 1, torch.device('cuda:0'))
        if batches:
            assert len(batches) == 10 and sum(len(x) for x, _ in batches) == 2560
        # Test actual builder RNG isolation on a small batch; smoke checks the full budget.
        build(model, method, [(x[:4], y[:4]) for x, y in batches[:1]])
        assert torch.equal(cpu, torch.random.get_rng_state())
        assert torch.equal(cuda, torch.cuda.get_rng_state())
        dataset = TensorDataset(torch.arange(768), torch.arange(768))
        loader = private_loader(dataset, seed)
        order = torch.cat([x for _ in range(2) for x, _ in loader])
        gen = noise_generator(seed, 'cuda:0')
        noise = []
        for _ in range(2):
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            noise_and_normalize(model, 1., 256, gen)
            noise.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone())
        actual = (initial, order, *noise)
        if reference is not None:
            for a, b in zip(actual, reference):
                assert torch.equal(a, b)
        reference = actual


def test_paired_contrast_signs():
    import pandas as pd
    from exp23.analyze import contrasts
    wide = pd.DataFrame({m: [i] for i, m in enumerate(cfg.METHODS)})
    result = contrasts(wide)
    assert result.Penalty_A.iloc[0] == -2
    assert result.Penalty_Full.iloc[0] == -2
    assert result.Interaction.iloc[0] == 0


def test_oracle_is_read_only():
    from exp23.geometry import alignment
    model = initialize(42, 'cuda:0')
    x = torch.randn(4, 1, 28, 28, device='cuda')
    y = torch.arange(4, device='cuda')
    _, factors, _ = build(model, 'full_pink', [(x, y)])
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    cpu, cuda = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    result = alignment(model, factors, [(x, y)], True)
    assert all(math.isclose(row['cosA'], 1., abs_tol=1e-12) for row in result)
    assert all(math.isclose(row['cosG'], 1., abs_tol=1e-12) for row in result)
    assert all(row['relative_frobenius_A'] == row['relative_frobenius_G'] == 0 for row in result)
    for name, p in model.named_parameters():
        assert torch.equal(p, weights[name]) and p.grad is None
    assert torch.equal(cpu, torch.random.get_rng_state())
    assert torch.equal(cuda, torch.cuda.get_rng_state())

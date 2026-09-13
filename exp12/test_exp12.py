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


def test_two_layer_kfra():
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 3)).double()
    x = torch.randn(5, 2, dtype=torch.double)
    with torch.no_grad():
        hidden = model[0](x)
        p = model(x).softmax(-1)
        last = sum(torch.diag(pi)-pi[:, None]*pi[None, :] for pi in p)/len(x)
        expected = torch.zeros(3, 3, dtype=torch.double)
        for hi in hidden:
            d = torch.diag((hi > 0).double())
            expected += d @ model[2].weight.T @ last @ model[2].weight @ d / len(x)
    out, stats = c.estimate(model, [x[:2], x[2:]], 'KFRA')
    torch.testing.assert_close(out['2']['C'], last)
    torch.testing.assert_close(out['0']['C'], expected)
    assert stats['forward_calls'] == 2
    assert stats['vjp_calls'] == stats['reverse_vectors'] == 0


def test_deeper_kfra():
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 3)).double()
    x = torch.randn(7, 2, dtype=torch.double)
    with torch.no_grad():
        h1 = model[0](x)
        h2 = model[2](h1.relu())
        p = model[4](h2.relu()).softmax(-1)
        reference = {'4': sum(torch.diag(pi)-torch.outer(pi, pi) for pi in p)/len(x)}
        for name, next_name, hidden in [('2', '4', h2), ('0', '2', h1)]:
            w = model[int(next_name)].weight
            reference[name] = sum(torch.diag((hi > 0).double()) @ w.T @ reference[next_name] @ w @ torch.diag((hi > 0).double()) for hi in hidden)/len(x)
    out, _ = c.estimate(model, [x[:3], x[3:]], 'KFRA')
    for n in reference:
        torch.testing.assert_close(out[n]['C'], reference[n])


def test_cnn_local_recursion():
    from dp_kfac.models import SimpleCNN
    torch.manual_seed(13)
    torch.set_num_threads(4)
    model = SimpleCNN().double()
    x = torch.randn(3, 1, 28, 28, dtype=torch.double)
    with torch.no_grad():
        a = model.pool1(model.conv1(x).relu())
        z2 = model.conv2(a)
        h = model.fc1(model.pool2(z2.relu()).flatten(1))
        p = model.fc2(h.relu()).softmax(-1)
        last = sum(torch.diag(pi)-torch.outer(pi, pi) for pi in p)/len(x)
        expected = sum(torch.diag((hi > 0).double()) @ model.fc2.weight.T @ last @ model.fc2.weight @ torch.diag((hi > 0).double()) for hi in h)/len(x)
        blocks = c.linear_to_blocks(model.fc1.weight, expected, (32, 7, 7))
        assert blocks.shape == (7, 7, 32, 32)
        for y in range(7):
            for xx in range(7):
                columns = torch.arange(32)*49+y*7+xx
                w = model.fc1.weight[:, columns]
                torch.testing.assert_close(blocks[y, xx], w.T @ expected @ w)
    out, stats = c.estimate(model, [x[:1], x[1:]], 'KFRA-block')
    torch.testing.assert_close(out['fc2']['C'], last)
    torch.testing.assert_close(out['fc1']['C'], expected)
    # Independent local argmax Jacobian, including channel-dependent pool routes.
    expected_conv2 = torch.zeros(32, 32, dtype=torch.double)
    for sample in z2:
        for y in range(7):
            for xx in range(7):
                patch = sample[:, 2*y:2*y+2, 2*xx:2*xx+2].reshape(32, 4)
                maxima, indices = patch.relu().max(-1)
                for pos in range(4):
                    d = torch.diag(((indices == pos) & (maxima > 0)).double())
                    expected_conv2 += d @ blocks[y, xx] @ d / (len(x)*14*14)
    torch.testing.assert_close(out['conv2']['C'], expected_conv2)
    assert stats['forward_calls'] == 2


def test_pool_and_conv_local_jacobians():
    torch.manual_seed(14)
    z = torch.randn(2, 2, 4, 4, dtype=torch.double)
    root = torch.randn(2, 2, 2, 2, dtype=torch.double)
    blocks = root @ root.mT
    actual = c.unpool_curvature(blocks, c.pool_gate_sum(z)/len(z))
    expected = torch.zeros_like(actual)
    for sample in z:
        jac = torch.autograd.functional.jacobian(lambda a: nn.functional.max_pool2d(a.relu(), 2), sample)
        for y in range(4):
            for x in range(4):
                for oy in range(2):
                    for ox in range(2):
                        j = jac[:, oy, ox, :, y, x]
                        expected[y, x] += j.T @ blocks[oy, ox] @ j / len(z)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn(2, 2, 3, 3, dtype=torch.double)
    root = torch.randn(3, 3, 2, 2, dtype=torch.double)
    blocks = root @ root.mT
    a = torch.randn(2, 3, 3, dtype=torch.double)
    jac = torch.autograd.functional.jacobian(lambda v: nn.functional.conv2d(v[None], weight, padding=1)[0], a)
    expected = torch.zeros_like(blocks)
    for y in range(3):
        for x in range(3):
            for oy in range(3):
                for ox in range(3):
                    j = jac[:, oy, ox, :, y, x]
                    expected[y, x] += j.T @ blocks[oy, ox] @ j
    torch.testing.assert_close(c.conv_to_blocks(weight, blocks), expected)


def test_u1_matches_existing_dp_kfc():
    from dp_kfac.models import SimpleCNN
    from dp_kfac.recorder import KFACRecorder
    from dp_kfac.covariance import compute_covariances
    torch.manual_seed(15)
    model = SimpleCNN()
    cache = [torch.randn(2, 1, 28, 28), torch.randn(2, 1, 28, 28)]
    labels = [torch.randint(10, (len(x),)) for x in cache]
    recorder = KFACRecorder(model)
    recorder.enable()
    original = []
    for x, y in zip(cache, labels):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
        original.append(compute_covariances(model, recorder.activations, recorder.backprops, eps=0))
        recorder.clear()
    recorder.remove()
    actual, _ = c.estimate_kfac_with_labels(model, cache, labels)
    for n in actual:
        torch.testing.assert_close(actual[n]['A'], sum(v.A[n] for v in original)/len(original), rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(actual[n]['C'], sum(v.G[n] for v in original)/len(original), rtol=2e-5, atol=1e-7)


def test_shared_cnn_activations_and_streaming():
    from dp_kfac.models import SimpleCNN
    torch.manual_seed(16)
    model = SimpleCNN().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    x = torch.randn(3, 1, 28, 28)
    ref, _ = c.estimate(model, [x], 'KFLR')
    for method, k in [('KFLR', 1), ('KFRA-block', 1)] + [(m, k) for m in ['KFAC-U', 'KFAC-M'] for k in [1, 3, 9]]:
        forward_calls = []
        handle = model.register_forward_hook(lambda *_: forward_calls.append(1))
        out, stats = c.estimate(model, [x[:1], x[1:]], method, k)
        handle.remove()
        assert len(forward_calls) == stats['forward_calls'] == 2
        for n in out:
            torch.testing.assert_close(out[n]['A'], ref[n]['A'])
        if method == 'KFLR':
            for n in out:
                torch.testing.assert_close(out[n]['C'], ref[n]['C'])
        if method == 'KFRA-block':
            full, _ = c.estimate(model, [x], method)
            for n in out:
                torch.testing.assert_close(out[n]['C'], full[n]['C'])
        assert all(p.grad is None for p in model.parameters())


def test_nonuniform_u_m_limits():
    model = nn.Sequential(nn.Linear(2, 3)).double()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.copy_(torch.tensor([2., 0., -1.]))
    x = [torch.zeros(4, 2, dtype=torch.double)]
    p = model(x[0]).softmax(-1).detach()[0]
    scores = p[None, :] - torch.eye(3, dtype=torch.double)
    uniform = scores.T @ scores/3
    exact, _ = c.estimate(model, x, 'KFLR')
    u, _ = c.estimate(model, x, 'KFAC-U', k=4000, seed=5)
    m, _ = c.estimate(model, x, 'KFAC-M', k=4000, seed=5)
    fisher = exact['0']['C']
    assert (m['0']['C']-fisher).norm()/fisher.norm() < .06
    assert (u['0']['C']-uniform).norm()/uniform.norm() < .04
    assert (u['0']['C']-fisher).norm()/fisher.norm() > 1

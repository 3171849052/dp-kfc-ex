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


def test_training_trajectory_checkpoints():
    """A short real-MNIST trajectory and the actual checkpoint CLI loading path."""
    import math
    import subprocess
    import sys
    import tempfile
    import pandas as pd
    from dp_kfac.models import SimpleCNN
    from exp12.train_trajectory import parse, train
    from exp12.run_trajectory_sweep import summarize_checkpoint
    root = Path(__file__).resolve().parents[1]
    parent = root/'exp12/checkpoints'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test_', dir=parent) as tmp:
        output = Path(tmp)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rows = train(parse(['--smoke', '--device', device, '--output', str(output)]))
        initial = torch.load(root/rows.iloc[0].checkpoint_path, map_location='cpu', weights_only=True)
        torch.manual_seed(42)
        expected = SimpleCNN().state_dict()
        assert initial.keys() == expected.keys()
        assert all(isinstance(v, torch.Tensor) for v in initial.values())
        for key in expected:
            assert torch.equal(initial[key], expected[key])
        final_path = root/rows.iloc[-1].checkpoint_path
        final = torch.load(final_path, map_location='cpu', weights_only=True)
        assert final.keys() == expected.keys()
        assert all(isinstance(v, torch.Tensor) for v in final.values())
        assert any(not torch.equal(initial[k], final[k]) for k in initial)
        table = pd.read_csv(output/'trajectory.csv')
        assert table.step.tolist() == [0, 2]
        assert torch.isfinite(torch.tensor(table.select_dtypes('number').to_numpy())).all()
        assert table.normalized_entropy.between(0, 1).all()
        assert table.mean_prediction_entropy.between(0, math.log(10)).all()
        assert table.mean_kl_to_uniform.between(0, math.log(10)).all()
        assert table.mean_max_probability.between(.1, 1).all()
        assert table.test_accuracy.between(0, 1).all()
        torch.testing.assert_close(torch.tensor(table.mean_kl_to_uniform.to_numpy()),
                                   math.log(10)-torch.tensor(table.mean_prediction_entropy.to_numpy()))
        diagnostic = output/'diagnostic'
        subprocess.run([sys.executable, str(root/'exp12/run_budget_sweep.py'), '--checkpoint',
                        str(final_path), '--output', str(diagnostic), '--device', device,
                        '--smoke', '--warmup', '1', '--repeats', '1'], check=True, capture_output=True, text=True)
        summary = summarize_checkpoint(diagnostic, next(table.tail(1).itertuples(index=False)))
        assert len(summary) == 8*4
        assert not summary.isna().any().any()
        assert (summary.loc[summary.estimator == 'KFLR', 'C_relative_error_vs_KFLR'] == 0).all()


def test_synthetic_state_metrics_use_supplied_cache():
    import math
    from exp12.state_metrics import synthetic_state_metrics
    model = nn.Linear(2, 10).double().eval()
    with torch.no_grad():
        model.weight.copy_(torch.arange(20, dtype=torch.double).reshape(10, 2)/3)
    cache = [torch.tensor([[0., 1.]], dtype=torch.double),
             torch.tensor([[1., 0.], [100., -2.]], dtype=torch.double)]
    seen = []
    handle = model.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
    actual = synthetic_state_metrics(model, cache)
    handle.remove()
    assert len(seen) == len(cache) and all(a is b for a, b in zip(seen, cache))
    logp = model(torch.cat(cache)).log_softmax(-1)
    p = logp.exp()
    entropy = (-(p*logp).sum(-1)).mean().item()
    assert abs(actual['synthetic_mean_prediction_entropy']-entropy) < 1e-12
    assert abs(actual['synthetic_mean_max_probability']-p.max(-1).values.mean().item()) < 1e-12
    assert abs(actual['synthetic_mean_kl_to_uniform']-(math.log(10)-entropy)) < 1e-12
    assert abs(actual['synthetic_normalized_entropy']-entropy/math.log(10)) < 1e-12
    assert all(math.isfinite(v) for v in actual.values())
    assert 0 <= actual['synthetic_normalized_entropy'] <= 1


def test_final_scheduled_step_is_not_duplicated(monkeypatch):
    import tempfile
    import pandas as pd
    from exp12 import train_trajectory as training
    parent = Path(__file__).parent/'checkpoints'
    parent.mkdir(exist_ok=True)
    # Exercise a scheduled positive step with only two updates.
    monkeypatch.setattr(training, 'SAVE_STEPS', (0, 2))
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        args = training.parse(['--smoke', '--device', 'cpu', '--output', tmp])
        training.train(args)
        rows = pd.read_csv(Path(tmp)/'trajectory.csv')
        assert rows.step.tolist() == [0, 2]
        assert rows.iloc[-1].checkpoint_path.endswith('step000002.pt')
        assert (Path(tmp)/'sgd_seed42_final.pt').is_file()
        assert not rows.duplicated(['seed', 'step']).any()


def test_trajectory_summary_domains_and_duplicate_guard(tmp_path):
    import pandas as pd
    import pytest
    from types import SimpleNamespace
    from exp12.run_trajectory_sweep import summarize_checkpoint, main
    identity = dict(estimator='KFAC-U-1', layer='fc2', label_seed=0)
    pd.DataFrame([dict(identity, kron_relative_error=.2, C_relative_error_vs_KFLR=.1)]).to_csv(tmp_path/'curvature_error.csv', index=False)
    pd.DataFrame([dict(identity, floored_condition_number=10.)]).to_csv(tmp_path/'whitening.csv', index=False)
    pd.DataFrame([dict(estimator='KFAC-U-1', label_seed=0, build_median_seconds=.01)]).to_csv(tmp_path/'compute_budget.csv', index=False)
    fields = ['mean_max_probability', 'mean_prediction_entropy', 'normalized_entropy', 'mean_kl_to_uniform']
    synthetic = {'synthetic_'+k: i+.25 for i, k in enumerate(fields)}
    (tmp_path/'state_metrics.json').write_text(json.dumps(synthetic))
    row = SimpleNamespace(checkpoint_path='model.pt', step=2, epoch=.1, **{k: i+.5 for i, k in enumerate(fields)})
    result = summarize_checkpoint(tmp_path, row).iloc[0]
    for key in fields:
        assert result['mnist_'+key] == getattr(row, key)
        assert result['synthetic_'+key] == synthetic['synthetic_'+key]
        assert key not in result.index
    assert result.epoch == .1
    pd.DataFrame([{'seed': 42, 'step': 2}]*2).to_csv(tmp_path/'duplicate.csv', index=False)
    with pytest.raises(AssertionError, match='must be unique'):
        main(['--trajectory', str(tmp_path/'duplicate.csv'), '--output', str(tmp_path/'unused')])
    assert not (tmp_path/'unused').exists()


def test_cuda_runtime_settings_and_cold_start():
    import subprocess
    import sys
    from exp12.runtime import runtime
    with runtime('cpu'):
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.autograd.is_multithreading_enabled()
    if torch.cuda.is_available():
        code = '''from exp12.runtime import runtime
import torch
from exp12.benchmark import measure
with runtime('cuda'):
    def build():
        a = torch.ones(3, 3, device='cuda', requires_grad=True)
        g, = torch.autograd.grad((a @ a).sum(), a)
        return {'linear': {'A': g, 'C': g}}, {}
    _, stats, rows = measure(build, 'cuda', warmup=2, repeats=2)
    assert len(rows) == 2 and stats['build_median_seconds'] > 0
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
        assert 'no current CUDA context' not in result.stderr
        assert 'cuBLAS' not in result.stderr

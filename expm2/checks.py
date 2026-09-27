"""CPU toy checks only; never initialize or train formal models."""
from __future__ import annotations

import ast
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm2 import ROOT, config as cfg
from expm2.bk import BKClipper, NUMERICAL_EPS
from expm2.geometry import FactorBatch, affine_modules, augmented_width, build_factors, output_width
from expm2.mechanism import Shape, add_noise_and_step

import torch
from torch import nn
from torch.nn import functional as F


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)).tanh())


def toy_checks():
    torch.manual_seed(42)
    model = Toy().double()
    factors = {}
    for name, module in affine_modules(model).items():
        a = torch.randn(augmented_width(module), augmented_width(module), dtype=torch.double)
        factors[name] = {'A': a @ a.T}
    x, y = torch.randn(5, 3, dtype=torch.double), torch.tensor([0, 1, 0, 1, 1])
    params = list(model.parameters())
    per_sample = {p: [] for p in params}
    for xx, yy in zip(x, y):
        grads = torch.autograd.grad(F.cross_entropy(model(xx[None]), yy[None]), params)
        for p, grad in zip(params, grads):
            per_sample[p].append(grad)
    per_sample = {p: torch.stack(values) for p, values in per_sample.items()}
    for method in cfg.METHODS:
        chosen = factors
        for beta, damping in ((b, cfg.DAMPING) for b in cfg.BETAS):
            shape = Shape(model, method, chosen, beta, damping)
            assert not hasattr(shape, 'tau')
            squared = torch.zeros(len(x), dtype=torch.double)
            for name, module in shape.modules.items():
                a = chosen[name]['A'] + damping * torch.eye(augmented_width(module), dtype=torch.double)
                va, ua = torch.linalg.eigh(a)
                ap = (ua * va.pow(-beta)) @ ua.T
                gp = torch.eye(output_width(module), dtype=torch.double)
                assert shape.data[name].noise_g is shape.data[name].metric_g is None
                block = shape.data[name]
                torch.testing.assert_close(block.metric_a, (ua * va.pow(beta / 2)) @ ua.T)
                torch.testing.assert_close(block.noise_a, (ua * va.pow(-beta / 2)) @ ua.T)
                torch.testing.assert_close(block.metric_a @ block.metric_a, (ua * va.pow(beta)) @ ua.T)
                assert math.isclose(block.raw_trace / output_width(module), float(va.pow(-beta).sum()))
                q = block.tau_layer * va.pow(-beta)
                assert 0 < block.diagnostics['q_min'] <= block.diagnostics['q_max']
                assert math.isclose(block.diagnostics['q_min'], float(q.min()))
                assert math.isclose(block.diagnostics['q_max'], float(q.max()))
                assert math.isclose(block.diagnostics['condition_S'], float(va.max()/va.min()) ** beta)
                covariance = torch.kron(gp.contiguous(), ap.contiguous()) * shape.data[name].tau_layer
                dim = covariance.shape[0]
                assert math.isclose(float(covariance.trace()), dim, rel_tol=1e-10)
                matrix = torch.cat((per_sample[module.weight], per_sample[module.bias][..., None]), dim=-1)
                flat = matrix.flatten(1)
                squared += torch.einsum('bi,ij,bj->b', flat, torch.linalg.inv(covariance), flat)
                # Actual sampling operators must have the specified covariance, not only reported trace.
                block = shape.data[name]
                ng = block.noise_g if block.noise_g is not None else torch.eye(output_width(module), dtype=torch.double)
                operator = math.sqrt(block.tau_layer) * torch.kron(ng.contiguous(), block.noise_a.contiguous())
                torch.testing.assert_close(operator @ operator.T, covariance)
                if beta == 0:
                    assert block.tau_layer == 1
                    assert torch.equal(block.noise_a, torch.eye(len(block.noise_a), dtype=torch.double))
            for p in shape.identity_parameters:
                squared += per_sample[p].flatten(1).square().sum(1)
            rows = shape.trace_rows()
            assert math.isclose(sum(r['trace_S'] for r in rows), shape.d_total)
            assert all(math.isclose(r['trace_S_per_dim'], 1, rel_tol=1e-12) for r in rows)
            identity = next(r for r in rows if r['layer'] == 'identity')
            assert identity['tau_layer'] == 1
            for name, module in shape.modules.items():
                row = next(r for r in rows if r['layer'] == name)
                assert math.isclose(row['tau_layer'] * row['trace_R'], augmented_width(module), rel_tol=1e-10)
            clipper = BKClipper(model, shape)
            result = clipper.aggregate_logical(x, y, physical_batch_size=2)
            clipper.remove()
            torch.testing.assert_close(result.matched_norms, squared.sqrt())
            expected_clips = (1 / (squared.sqrt() + NUMERICAL_EPS)).clamp(max=1)
            assert result.clip_factors.shape == (len(x),)
            torch.testing.assert_close(result.clip_factors, expected_clips)
            for p in params:
                expected = (per_sample[p] * expected_clips.reshape(-1, *([1] * p.ndim))).sum(0)
                torch.testing.assert_close(result.clipped_sum[p], expected)
                assert torch.equal(shape.transform_aggregate(result.clipped_sum)[p], result.clipped_sum[p])
            if beta == 0:
                torch.testing.assert_close(result.matched_norms, result.raw_norms, rtol=0, atol=0)
            base = shape._base_noise(torch.Generator().manual_seed(51))
            noise, energy = shape.sample_noise(2., 1., torch.Generator().manual_seed(51))
            for name, module in shape.modules.items():
                assert math.isclose(energy[name], 4 * sum(p.numel() for p in module.parameters()))
            for name, module in shape.modules.items():
                block = shape.data[name]
                raw = torch.cat((base[module.weight], base[module.bias][:, None]), dim=1)
                expected_noise = 2 * math.sqrt(block.tau_layer) * raw @ block.noise_a
                torch.testing.assert_close(noise[module.weight], expected_noise[:, :-1])
                torch.testing.assert_close(noise[module.bias], expected_noise[:, -1])
            for p in shape.identity_parameters:
                assert torch.equal(noise[p], base[p] * 2)
                assert energy[shape.parameter_names[p]] == 4 * p.numel()
            if beta == 0:
                assert all(torch.equal(noise[p], base[p] * 2) for p in params)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.)
            step_noise, _ = shape.sample_noise(2., 1., torch.Generator().manual_seed(9))
            def verify_step(*args, **kwargs):
                for p in params:
                    torch.testing.assert_close(p.grad, (result.clipped_sum[p] + step_noise[p]) / 256)
            optimizer.step = verify_step
            _, noise_rows = add_noise_and_step(shape, optimizer, result.raw_sum, result.clipped_sum,
                sigma=2., bound=1., expected_batch_size=256, generator=torch.Generator().manual_seed(9))
            dimensions = {}
            for row in rows:
                dimensions[row['group']] = dimensions.get(row['group'], 0) + row['layer_dimension']
            for row in noise_rows:
                if row['level'] == 'group':
                    assert math.isclose(row['noise_energy_share'], dimensions[row['group']] / shape.d_total)

    # A-only builder must neither backpropagate nor construct G, even with invalid labels.
    model = Toy()
    batches = [FactorBatch(torch.randn(256, 3), torch.full((256,), 99), 'public') for _ in range(10)]
    with patch.object(torch.Tensor, 'backward', side_effect=AssertionError('unexpected backward')):
        factors, stats = build_factors(model, batches, need_g=False, physical_batch_size=128)
    assert stats['builder_backward_calls'] == 0
    assert all('G' not in factor for factor in factors.values())


def protocol_checks():
    assert len(cfg.GRID) == len({s.name for s in cfg.GRID}) == 8
    assert {s.task for s in cfg.GRID} == {'vit'}
    assert {s.method for s in cfg.GRID} == {'inverse_dp_kfm_a'}
    assert {s.source for s in cfg.GRID} == {'public', 'pink'}
    assert {s.beta for s in cfg.GRID} == {.1, .2, .3, .4}
    assert all(s.beta > 0 for s in cfg.GRID)
    assert {s.damping for s in cfg.GRID} == {.001}
    assert {s.seed for s in cfg.GRID} == {42}
    assert {s.geometry_power for s in cfg.GRID} == {'inverse'}
    for gpu, beta in enumerate(cfg.BETAS):
        assert [(s.source, s.beta) for s in cfg.GRID if s.gpu == gpu] == [('public', beta), ('pink', beta)]
    assert cfg.VIT.accountant_steps == 975 and cfg.VIT.steps_per_epoch == 195
    assert cfg.VIT.physical_batch_size == 128 and cfg.VIT.logical_batch_size == 256
    assert cfg.DATA_ROOT == cfg.REPO_ROOT / 'data'
    for path in (cfg.RESULTS_ROOT, cfg.LOGS_ROOT, cfg.CACHE_ROOT, cfg.TMP_ROOT, *(s.directory for s in cfg.GRID)):
        assert path.is_relative_to(ROOT)
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    tree = ast.parse((ROOT / 'data.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'datasets']
    assert calls
    assert all(any(k.arg == 'download' and isinstance(k.value, ast.Constant) and k.value.value is False
                   for k in call.keywords) for call in calls)
    from expm2 import data
    with patch.object(torch, 'randint', side_effect=AssertionError('A-only pseudo-labels')), \
         patch.object(data, '_pink_input_shape', return_value=(3, 8, 8)):
        batch = next(data.pink_calibration('vit', 42, 1, 'cpu', need_g=False))
        assert torch.equal(batch.y, torch.zeros_like(batch.y))
    subprocess.run(['bash', '-n', str(ROOT / 'run_all.sh')], check=True)


def analysis_check():
    import pandas as pd
    from expm2.analyze import reference_rows, reference_path, paired_table, plots, report
    paths = [reference_path(s.source, s.beta) for s in cfg.GRID] + [reference_path()]
    before = {p: {f.name: (f.stat().st_size, f.stat().st_mtime_ns) for f in p.iterdir() if f.is_file()} for p in paths}
    refs = reference_rows()
    assert len(refs) == 9
    assert sum(r['origin'] == 'forward' for r in refs) == 8
    rows = [dict(r, origin='inverse', geometry_power='inverse', signed_beta=-r['beta']) for r in refs if r['origin'] == 'forward']
    table = pd.DataFrame(rows + refs)
    paired = paired_table(table)
    assert len(table) == 17 and len(paired) == 8
    assert (paired.delta_inverse_minus_forward == 0).all()
    assert 'statistical significance' in report(paired)
    with tempfile.TemporaryDirectory(dir=cfg.CACHE_ROOT) as temporary:
        plots(table, Path(temporary))
        assert len(list(Path(temporary).glob('*.png'))) == 6
    after = {p: {f.name: (f.stat().st_size, f.stat().st_mtime_ns) for f in p.iterdir() if f.is_file()} for p in paths}
    assert before == after
    print('References recognized read-only: 8 forward + 1 DP-SGD.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    protocol_checks()
    toy_checks()
    analysis_check()
    print('PASS: 8 inverse runs, static queues, inverse powers/metric/covariance, global raw clipping, trace/noise energy, identity, A-only, analysis and shell syntax. No formal training in checks.')

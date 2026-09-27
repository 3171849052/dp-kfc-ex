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
from expm1c import ROOT, config as cfg
from expm1c.bk import BKClipper, NUMERICAL_EPS
from expm1c.geometry import FactorBatch, affine_modules, augmented_width, build_factors, output_width
from expm1c.mechanism import Shape, add_noise_and_step

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
        g = torch.randn(output_width(module), output_width(module), dtype=torch.double)
        factors[name] = {'A': a @ a.T, 'G': g @ g.T}
    x, y = torch.randn(5, 3, dtype=torch.double), torch.tensor([0, 1, 0, 1, 1])
    params = list(model.parameters())
    per_sample = {p: [] for p in params}
    for xx, yy in zip(x, y):
        grads = torch.autograd.grad(F.cross_entropy(model(xx[None]), yy[None]), params)
        for p, grad in zip(params, grads):
            per_sample[p].append(grad)
    per_sample = {p: torch.stack(values) for p, values in per_sample.items()}
    for method in cfg.METHODS:
        chosen = factors if method == 'dp_kfm' else {n: {'A': v['A']} for n, v in factors.items()}
        for beta, damping in ((b, d) for b in (0., *cfg.BETA_CANDIDATES) for d in cfg.LAMBDA_CANDIDATES):
            shape = Shape(model, method, chosen, beta, damping)
            assert not hasattr(shape, 'tau')
            squared = torch.zeros(len(x), dtype=torch.double)
            for name, module in shape.modules.items():
                a = chosen[name]['A'] + damping * torch.eye(augmented_width(module), dtype=torch.double)
                va, ua = torch.linalg.eigh(a)
                ap = (ua * va.pow(beta)) @ ua.T
                if method == 'dp_kfm':
                    g = chosen[name]['G'] + .001 * torch.eye(output_width(module), dtype=torch.double)
                    vg, ug = torch.linalg.eigh(g)
                    gp = (ug * vg.pow(beta)) @ ug.T
                else:
                    gp = torch.eye(output_width(module), dtype=torch.double)
                    assert shape.data[name].noise_g is shape.data[name].metric_g is None
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
            for p in shape.identity_parameters:
                assert torch.equal(noise[p], base[p] * 2)
                assert energy[shape.parameter_names[p]] == 4 * p.numel()
            if beta == 0:
                assert all(torch.equal(noise[p], base[p] * 2) for p in params)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.)
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
    assert len(cfg.BETA_GRID) == len({s.name for s in cfg.BETA_GRID}) == 8
    assert cfg.BETA_CANDIDATES == (.1, .2, .25, .3, .4)
    assert cfg.LAMBDA_CANDIDATES == (1e-4, 1e-3, 1e-2, 1e-1)
    assert len(cfg.LAMBDA_TEMPLATE) == 6
    dynamic = cfg.lambda_grid({'public': .25, 'pink': .4})
    assert [s.gpu for s in dynamic] == [0, 0, 1, 1, 2, 3]
    for gpu, beta in enumerate(cfg.BETAS):
        runs = [s for s in cfg.BETA_GRID if s.gpu == gpu]
        assert [(s.source, s.beta) for s in runs] == [('public', beta), ('pink', beta)]
    extra = cfg.extra_lambda_grid({'public': .1, 'pink': .1})
    assert len(extra) == 4 and [s.gpu for s in extra] == [0, 1, 2, 3]
    assert {s.damping for s in extra} == {1e-5, 1e-6}
    assert all(s.beta == .1 and s.phase == 'lambda' for s in extra)
    all_runs = (*cfg.BETA_GRID, *dynamic, *extra)
    assert {s.seed for s in all_runs} == {42}
    assert {s.task for s in all_runs} == {'vit'}
    assert {s.method for s in all_runs} == {'dp_kfm_a'}
    assert {s.source for s in all_runs} == {'public', 'pink'}
    assert cfg.VIT.accountant_steps == 975 and cfg.VIT.physical_batch_size == 128
    assert cfg.DATA_ROOT == cfg.REPO_ROOT / 'data'
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    tree = ast.parse((ROOT / 'data.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'datasets']
    assert calls
    assert all(any(k.arg == 'download' and isinstance(k.value, ast.Constant) and k.value.value is False
                   for k in call.keywords) for call in calls)
    from expm1c import data
    with patch.object(torch, 'randint', side_effect=AssertionError('A-only pseudo-labels')), \
         patch.object(data, '_pink_input_shape', return_value=(3, 8, 8)):
        batch = next(data.pink_calibration('vit', 42, 1, 'cpu', need_g=False))
        assert torch.equal(batch.y, torch.zeros_like(batch.y))
    from expm1c.analyze_beta import reference, summarize
    for source in cfg.SOURCES:
        assert reference(source)['beta'] == .25
    summarize(cfg.REPO_ROOT / 'expm1/results/runs/vit_dp_sgd_none_seed42',
              'none', None, cfg.DAMPING, 'expm1_reference')
    assert 'expm1' not in sys.modules and 'expm1b' not in sys.modules
    for script in ('run_beta.sh', 'run_lambda.sh', 'run_all.sh', 'run_lambda_extra.sh'):
        subprocess.run(['bash', '-n', str(ROOT / script)], check=True)


def analysis_check():
    import json
    import pandas as pd
    from expm1c.analyze_beta import select_betas, summarize, plots
    from expm1c.analyze import lambda_table
    rows = []
    for source in cfg.SOURCES:
        for beta in cfg.BETA_CANDIDATES:
            rows.append(dict(source=source, beta=beta, **{'lambda': .001},
                final_accuracy=.6, accuracy_auc=2., mean_clip_factor=.5,
                clip_fraction=.4, median_condition_S=2., origin='synthetic'))
    table = pd.DataFrame(rows)
    assert select_betas(table) == {'public': .1, 'pink': .1}
    table.loc[(table.source == 'public') & (table.beta == .25), 'accuracy_auc'] = 3.
    table.loc[(table.source == 'pink') & (table.beta == .4), 'mean_clip_factor'] = .6
    selected = select_betas(table)
    assert selected == {'public': .25, 'pink': .4}
    table.loc[(table.source == 'public') & (table.beta == .2), 'final_accuracy'] = .7
    assert select_betas(table)['public'] == .2
    table.loc[(table.source == 'public') & (table.beta == .2), 'final_accuracy'] = .6
    with tempfile.TemporaryDirectory(dir=cfg.CACHE_ROOT) as temporary:
        output = Path(temporary)
        def synthetic(directory, source, beta, damping, origin):
            return dict(source=source, beta=beta, **{'lambda': damping},
                        final_accuracy=.5, accuracy_auc=2., clip_fraction=.4,
                        mean_clip_factor=.5, median_condition_S=2., origin=origin)
        with patch('expm1c.analyze.summarize', side_effect=synthetic):
            merged = lambda_table(selected, table)
        assert len(merged) == 8
        assert merged[merged['lambda'] == .001].origin.tolist() == ['synthetic', 'synthetic']
        plots(table, 'beta', output / 'beta')
        plots(merged, 'lambda', output / 'lambda')
        assert len(list(output.rglob('*.png'))) == 10
        directory = output / 'fixture'
        directory.mkdir()
        (directory / 'complete.json').write_text(json.dumps(dict(run_name='fixture', epochs=5)))
        (directory / 'config.json').write_text(json.dumps(dict(task='vit', seed=42,
            source='public', beta=.25, damping=.001)))
        pd.DataFrame([dict(epoch=e, test_accuracy=e / 10, accuracy_auc=1.2,
            clip_fraction=.4, mean_clip_factor=.5, matched_norm_p90=2., raw_norm_p90=1.)
            for e in range(1, 6)]).to_csv(directory / 'metrics.csv', index=False)
        pd.DataFrame([dict(epoch=e, layer='fc', condition_S=2., log_eigenvalue_spread_S=.3,
            effective_rank_S=3.) for e in range(1, 6)]).to_csv(directory / 'geometry.csv', index=False)
        row = summarize(directory, 'public', .25, .001, 'fixture')
        assert row['final_accuracy'] == .5 and row['matched_raw_p90_ratio'] == 2.
        (directory / 'complete.json').unlink()
        try:
            summarize(directory, 'public', .25, .001, 'fixture')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('incomplete run accepted')


if __name__ == '__main__':
    torch.set_num_threads(2)
    protocol_checks()
    toy_checks()
    analysis_check()
    print('PASS: 8+6 grids, dynamic selection, per-layer covariance/noise, identity, raw global clipping, A-only, analysis, shell syntax; no formal training.')

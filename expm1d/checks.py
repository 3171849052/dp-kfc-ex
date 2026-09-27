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
from expm1d import ROOT, config as cfg
from expm1d.bk import BKClipper, NUMERICAL_EPS
from expm1d.geometry import FactorBatch, affine_modules, augmented_width, build_factors, output_width
from expm1d.mechanism import Shape, add_noise_and_step

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
        chosen = factors if method == 'dp_kfm' else {n: {'A': v['A']} for n, v in factors.items()}
        for beta, damping in ((0., 1e-4), (.1, 1e-4)):
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
    assert len(cfg.GRID) == len({s.name for s in cfg.GRID}) == 8
    assert {s.seed for s in cfg.GRID} == {42}
    assert {s.task for s in cfg.GRID} == {'vit'}
    assert {s.method for s in cfg.GRID} == {'dp_kfm_a'}
    assert {s.source for s in cfg.GRID} == {'public', 'pink'}
    assert {s.beta for s in cfg.GRID} == {.1}
    assert {s.damping for s in cfg.GRID} == {1e-4}
    assert {s.frequency for s in cfg.GRID} == set(cfg.NEW_FREQUENCIES)
    expected = {
        'once': (0,), 'every_2_epochs': (0, 390, 780),
        'every_epoch': (0, 195, 390, 585, 780),
        'twice_per_epoch': tuple(e * 195 + slot for e in range(5) for slot in (0, 98)),
        'four_per_epoch': tuple(e * 195 + slot for e in range(5) for slot in (0, 49, 98, 147)),
    }
    assert {f: cfg.refresh_steps(f) for f in cfg.FREQUENCIES} == expected
    assert [len(cfg.refresh_steps(f)) for f in cfg.NEW_FREQUENCIES] == [1, 3, 10, 20]
    assert cfg.GPU_ASSIGNMENTS == (
        (('public', 'four_per_epoch'), ('pink', 'once')),
        (('pink', 'four_per_epoch'), ('public', 'once')),
        (('public', 'twice_per_epoch'), ('pink', 'every_2_epochs')),
        (('pink', 'twice_per_epoch'), ('public', 'every_2_epochs')),
    )
    for gpu, pairs in enumerate(cfg.GPU_ASSIGNMENTS):
        assert tuple((s.source, s.frequency) for s in cfg.GRID if s.gpu == gpu) == pairs
    assert all(s.directory.is_relative_to(ROOT) for s in cfg.GRID)
    assert all(p.is_relative_to(ROOT) for p in (cfg.RESULTS_ROOT, cfg.LOGS_ROOT, cfg.TMP_ROOT, cfg.CACHE_ROOT))
    assert cfg.DATA_ROOT == cfg.REPO_ROOT / 'data'
    p = cfg.VIT
    assert (p.steps_per_epoch, p.accountant_steps, p.logical_batch_size, p.physical_batch_size) == (195, 975, 256, 128)
    assert (p.epsilon, p.delta, p.max_grad_norm, p.epochs) == (3, 1e-5, 1, 5)
    assert (p.learning_rate, p.weight_decay, p.betas, p.optimizer_eps) == (1e-4, .01, (.9, .999), 1e-8)
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    tree = ast.parse((ROOT / 'data.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'datasets']
    assert len(calls) == 3
    assert all(any(k.arg == 'download' and isinstance(k.value, ast.Constant) and k.value.value is False
                   for k in call.keywords) for call in calls)
    # The copied algorithm/backend/data bindings differ only in namespace and damping.
    for name in ('data.py', 'vit.py', 'bk.py', 'mechanism.py', '__init__.py'):
        baseline = (cfg.REPO_ROOT / 'expm1c' / name).read_text().replace('expm1c', 'expm1d').replace('ExpM1c', 'ExpM1d')
        assert (ROOT / name).read_text() == baseline
    worker = ast.parse((ROOT / 'worker.py').read_text())
    run = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    epoch_loop = next(n for n in run.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'epoch')
    step_loop = next(n for n in epoch_loop.body if isinstance(n, ast.For) and ast.unparse(n.target) == '(local_step, indices)')
    calls = lambda n: [ast.unparse(x.func) for x in ast.walk(n) if isinstance(x, ast.Call)]
    for constructor in ('module.build_optimizer', 'RDPAccountant', 'torch.Generator'):
        assert calls(run).count(constructor) == 1
        assert constructor not in calls(epoch_loop)
    assert calls(epoch_loop).count('module.oracle_calibration') == 1
    assert 'module.oracle_calibration' not in calls(step_loop)
    refresh = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'refresh_geometry')
    assert 'module.oracle_calibration' not in calls(refresh)
    assert not any('optimizer' in c or 'accountant' in c or 'Generator' in c for c in calls(refresh))
    assert 'clipper.remove' in calls(step_loop) and 'refresh_geometry' in calls(step_loop)
    assert 'aggregate.clipped_sum' in ast.unparse(step_loop)
    subprocess.run(['bash', '-n', str(ROOT / 'run_all.sh')], check=True)


def calibration_and_refresh_checks():
    from expm1d import data
    from expm1d.geometry import training_drift
    from expm1d.worker import refresh_geometry
    from types import SimpleNamespace
    # Public and pink inputs repeat within an epoch, and change across epochs.
    dataset = torch.utils.data.TensorDataset(torch.arange(3000.).reshape(-1, 1), torch.zeros(3000, dtype=torch.long))
    with patch.object(data, 'public_dataset', return_value=dataset), \
         patch.object(data, '_pink_input_shape', return_value=(3, 8, 8)):
        for calibration in (data.public_calibration, data.pink_calibration):
            state = torch.random.get_rng_state().clone()
            first = list(calibration('vit', 42, 1, 'cpu', need_g=False))
            second = list(calibration('vit', 42, 1, 'cpu', need_g=False))
            third = list(calibration('vit', 42, 2, 'cpu', need_g=False))
            assert all(torch.equal(a.x, b.x) for a, b in zip(first, second))
            assert not torch.equal(first[0].x, third[0].x)
            assert torch.equal(state, torch.random.get_rng_state())
    model = Toy()
    batches = [FactorBatch(torch.randn(256, 3), torch.zeros(256, dtype=torch.long), 'public') for _ in range(10)]
    module = SimpleNamespace(public_calibration=lambda seed, epoch, device, need_g: iter(batches))
    spec = cfg.RunSpec('public', 'four_per_epoch')
    optimizer = torch.optim.AdamW(model.parameters())
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    accountant.step(noise_multiplier=1., sample_rate=.01)
    noise = torch.Generator().manual_seed(123)
    before = noise.get_state().clone()
    ids = (id(model), id(optimizer), id(accountant))
    with patch('expm1d.worker._synchronize'):
        factors, shape, first = refresh_geometry(model, module, spec, cfg.VIT, torch.device('cpu'), 1, 0, None, None, 0)
        with torch.no_grad():
            model.fc1.weight.add_(.1)
        new_factors, new_shape, second = refresh_geometry(model, module, spec, cfg.VIT, torch.device('cpu'), 1, 49, factors, shape, 1)
    assert first['builder_samples'] == second['builder_samples'] == 2560
    assert second['builder_forward_calls'] == 20 and second['builder_backward_calls'] == 0
    assert first['global_step'] == 0 and second['global_step'] == 49
    assert math.isnan(first['median_Q_cos_prev'])
    assert second['median_A_rel_change_prev'] > 0 and second['median_Q_rel_change_prev'] > 0
    same = training_drift(new_factors, new_shape, new_factors, new_shape)
    assert math.isclose(same['median_Q_cos_prev'], 1) and same['median_Q_rel_change_prev'] == 0
    assert ids == (id(model), id(optimizer), id(accountant))
    assert accountant.history == [(1., .01, 1)] and torch.equal(before, noise.get_state())


def analysis_checks():
    import json
    import pandas as pd
    from expm1d.analyze import summarize, frequency_table, plots, DRIFT
    # Read the actual references without importing their package or writing there.
    for source in cfg.SOURCES:
        ref = summarize(cfg.reference_directory(source), source, 'every_epoch', 'expm1c_reference')
        assert ref['geometry_build_count'] == 5
        assert all(math.isnan(ref[k]) for k in DRIFT)
    with tempfile.TemporaryDirectory(dir=cfg.CACHE_ROOT) as temporary:
        output = Path(temporary)
        def fixture(directory, source, frequency, origin):
            return dict(source=source, frequency=frequency, geometry_build_count=len(cfg.refresh_steps(frequency)),
                        final_accuracy=.5, accuracy_auc=2., clip_fraction=.4,
                        total_geometry_build_seconds=10., mean_refresh_interval_steps=49.,
                        origin=origin, **dict.fromkeys(DRIFT, .1))
        with patch('expm1d.analyze.summarize', side_effect=fixture):
            table = frequency_table()
        assert len(table) == 10 and (table.origin == 'expm1c_reference').sum() == 2
        plots(table, output / 'plots')
        assert len(list((output / 'plots').glob('*.png'))) == 5
        run = cfg.RunSpec('public', 'once')
        directory = output / run.name
        directory.mkdir()
        (directory / 'complete.json').write_text(json.dumps(dict(run_name=run.name, epochs=5,
            geometry_build_count=1, oracle_build_count=5, total_geometry_build_seconds=2.,
            total_runtime_seconds=20., total_training_seconds=10.)))
        (directory / 'config.json').write_text(json.dumps(dict(task='vit', method='dp_kfm_a', seed=42,
            source='public', frequency='once', beta=.1, damping=1e-4)))
        pd.DataFrame([dict(epoch=e, test_accuracy=e/10, accuracy_auc=1.2, test_loss=1.,
            clip_fraction=.4, mean_clip_factor=.5, matched_norm_p90=2., raw_norm_p90=1.,
            geometry_build_seconds=2. if e == 1 else 0., private_train_seconds=2., run_wall_time_seconds=e*4.)
            for e in range(1,6)]).to_csv(directory / 'metrics.csv', index=False)
        pd.DataFrame([dict(epoch=e, layer='fc', condition_S=2.) for e in range(1,6)]).to_csv(directory / 'geometry.csv', index=False)
        pd.DataFrame([dict(global_step=0, **dict.fromkeys(DRIFT, float('nan')))]).to_csv(directory / 'refresh.csv', index=False)
        row = summarize(directory, 'public', 'once', 'expm1d')
        assert row['final_accuracy'] == .5 and row['geometry_runtime_fraction'] == .1
        assert math.isnan(row['median_Q_cos_prev'])
    assert 'expm1c' not in sys.modules


if __name__ == '__main__':
    torch.set_num_threads(2)
    protocol_checks()
    toy_checks()
    calibration_and_refresh_checks()
    analysis_checks()
    print('PASS: 8-run grid, schedules, static GPUs, calibration RNG, refresh lifecycle, oracle scope, A-only, raw global clipping, trace/noise, references, analysis, output isolation, shell syntax; no formal training.')

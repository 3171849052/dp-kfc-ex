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
from expm1b import ROOT, config as cfg
from expm1b.bk import BKClipper, NUMERICAL_EPS
from expm1b.geometry import FactorBatch, affine_modules, augmented_width, build_factors, output_width
from expm1b.mechanism import Shape, add_noise_and_step

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
        for beta in (0., *cfg.BETAS):
            shape = Shape(model, method, chosen, beta)
            assert not hasattr(shape, 'tau')
            squared = torch.zeros(len(x), dtype=torch.double)
            for name, module in shape.modules.items():
                a = chosen[name]['A'] + .001 * torch.eye(augmented_width(module), dtype=torch.double)
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
    assert len(cfg.FORMAL_GRID) == len({s.name for s in cfg.FORMAL_GRID}) == 16
    assert {s.beta for s in cfg.FORMAL_GRID} == {.25, .5}
    for gpu, (method, beta) in enumerate((('dp_kfm', .25), ('dp_kfm', .5), ('dp_kfm_a', .25), ('dp_kfm_a', .5))):
        runs = cfg.GPU_RUNS[gpu]
        assert len(runs) == 4
        assert [(s.task, s.source) for s in runs] == [('vit', 'pink'), ('mnist', 'pink'), ('vit', 'public'), ('mnist', 'public')]
        assert all(s.gpu == gpu and s.method == method and s.beta == beta and s.seed == 42 for s in runs)
    assert cfg.DATA_ROOT == cfg.REPO_ROOT / 'data'
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    tree = ast.parse((ROOT / 'data.py').read_text())
    dataset_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'datasets']
    assert len(dataset_calls) == 6
    assert all(any(k.arg == 'download' and isinstance(k.value, ast.Constant) and k.value.value is False for k in call.keywords) for call in dataset_calls)
    from expm1b import data
    with patch.object(torch, 'randint', side_effect=AssertionError('A-only pseudo-labels')):
        batch = next(data.pink_calibration('mnist', 42, 1, 'cpu', need_g=False))
        assert torch.equal(batch.y, torch.zeros_like(batch.y))
    # No import of ExpM1, whose __init__ would create caches in its directory.
    assert 'expm1' not in sys.modules
    subprocess.run(['bash', '-n', str(ROOT / 'run_all.sh')], check=True)


def analysis_check():
    """Exercise CSV joins and all plots with small synthetic data, never real results."""
    import pandas as pd
    from expm1b.analyze import write_outputs
    metrics, groups, geometry = [], [], []
    conditions = [(s.name, s.task, s.method, s.source, s.beta, norm)
                  for s in cfg.FORMAL_GRID for norm in ('global', 'layer')]
    conditions += [(f'{task}_dp_sgd_none_seed42', task, 'dp_sgd', 'none', None, 'dp_sgd') for task in cfg.TASKS]
    for name, task, method, source, beta, norm in conditions:
        base = dict(run_name=name, task=task, method=method, source=source, beta=beta, normalization=norm, epoch=5)
        metrics.append(base | dict(test_accuracy=.6 if norm == 'layer' else .5,
            clip_fraction=.4, mean_clip_factor=.5, matched_norm_p90=2., raw_norm_p90=1.,
            total_noise_rms=1., clip_cos=.9, signal_cos=.9))
        geometry.append(base | dict(layer='fc1', trace_S=10.))
        groups.append(base | dict(level='group', group='fc1', snr=.1, noise_energy_share=1.))
    with tempfile.TemporaryDirectory(dir=cfg.CACHE_ROOT) as directory:
        output = Path(directory)
        write_outputs(pd.DataFrame(metrics), pd.DataFrame(geometry), pd.DataFrame(groups), output)
        paired = pd.read_csv(output / 'paired_summary.csv')
        assert len(paired) == 16
        assert (paired.delta_layer_minus_global.sub(.1).abs() < 1e-12).all()
        assert len(pd.read_csv(output / 'final_summary.csv')) == 34
        assert len(list(output.glob('*.png'))) == 10


if __name__ == '__main__':
    torch.set_num_threads(2)
    protocol_checks()
    toy_checks()
    analysis_check()
    print('PASS: 16-run grid, per-layer covariance/noise, identity, raw global clipping, A-only, analysis, shell syntax; no formal training.')

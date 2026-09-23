"""CPU contracts and small-model numerical checks; never launch ViT training."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp37.config import ROOT, cfg, RUNS, POWERS, GPU_RUNS, GPU_BY_RUN
from exp37.geometry import LayerScaleOperator, scalar_scales, build_from_batches
from exp37 import worker, analyze
from exp22 import geometry as reference
from exp22.methods import Clipper
from exp35.adapters import bind
import os
import subprocess
import tempfile
from unittest import TestCase
from unittest.mock import patch
import torch
from torch import nn


def numerical():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2, bias=False))
    model.num_classes = 2
    x, y = torch.randn(6, 3) * 4, torch.tensor([0, 1, 0, 1, 1, 0])
    before = [p.detach().clone() for p in model.parameters()]
    # Obtain reference A/G definitions without invoking a matrix operator/eigh.
    class CaptureFactors:
        def __init__(self, factors, damping):
            self.factors = factors
            self.operator_state_bytes = 0
    reference_build = bind(reference.build_full_operator, FullKFACOperator=CaptureFactors)
    captured, _ = reference_build(model, [x[:4], x[4:]], 42, 1, .001)
    expected_a = {n: f['A'].trace().item() / len(f['A']) for n, f in captured.factors.items()}
    expected_g = {n: f['G'].trace().item() / len(f['G']) for n, f in captured.factors.items()}
    dimensions = {n: sum(p.numel() for p in m.parameters()) for n, m in reference.affine_modules(model).items()}
    for family in ('trace_a', 'trace_ag'):
        for power in POWERS:
            with patch.object(torch.linalg, 'eigh', side_effect=AssertionError('eigh forbidden')):
                if family == 'trace_a':
                    with patch.object(reference, 'synthetic_labels', side_effect=AssertionError('labels forbidden')):
                        op, stats = build_from_batches(model, family, [x[:4], x[4:]], 42, 1, .001, power)
                else:
                    op, stats = build_from_batches(model, family, [x[:4], x[4:]], 42, 1, .001, power)
            assert set(vars(op)) == {'data'} and all(isinstance(v, float) for v in op.data.values())
            assert stats['builder_samples'] == 6
            assert stats['builder_vjp_calls'] == (2 if family == 'trace_ag' else 0)
            for name in dimensions:
                torch.testing.assert_close(torch.tensor(stats['trace_a_per_dim_' + name]), torch.tensor(expected_a[name]))
                base = expected_a[name] + .001
                if family == 'trace_ag':
                    torch.testing.assert_close(torch.tensor(stats['trace_g_per_dim_' + name]), torch.tensor(expected_g[name]))
                    base *= expected_g[name] + .001
                assert abs(stats['raw_scale_' + name] / base ** (-power) - 1) < 1e-5
            assert abs(sum(dimensions[n] * s*s for n, s in op.data.items()) / sum(dimensions.values()) - 1) < 1e-12
            grads = []
            for xi, yi in zip(x, y):
                sample = torch.autograd.grad(nn.functional.cross_entropy(model(xi[None]), yi[None]), tuple(model.parameters()))
                grads.append([g * op.data[name.rsplit('.', 1)[0]] for (name, _), g in zip(model.named_parameters(), sample)])
            expected_norms = torch.stack([sum(g.square().sum() for g in sample).sqrt() for sample in grads])
            factors = (2 / (expected_norms + 1e-6)).clamp(max=1)
            clipper = Clipper(model, op, method='bk', max_grad_norm=2)
            _, norms, actual_factors, _, _ = clipper.aggregate_logical(x, y, 2)
            torch.testing.assert_close(norms, expected_norms)
            torch.testing.assert_close(actual_factors, factors)
            expected_rng = torch.Generator().manual_seed(40042)
            expected_noisy = []
            for j, parameter in enumerate(model.parameters()):
                aggregate = sum(factor * sample[j] for factor, sample in zip(factors, grads))
                torch.testing.assert_close(parameter.grad, aggregate)
                expected_noisy.append((aggregate + .7 * 2 * torch.randn(parameter.numel(), generator=expected_rng).reshape_as(parameter)) / len(x))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0)
            clipper.step(optimizer, .7, len(x), torch.Generator().manual_seed(40042))
            for parameter, expected in zip(model.parameters(), expected_noisy):
                torch.testing.assert_close(parameter.grad, expected)
            assert clipper.noise_events == clipper.optimizer_steps == 1
            clipper.remove()
    for p, old in zip(model.parameters(), before):
        torch.testing.assert_close(p, old)
    assert all(not m._forward_hooks for m in model.modules())
    for g in (None, {'a': 0., 'b': 3.}):
        for power in (*POWERS, 0.):
            raw, scales = scalar_scales({'a': 0., 'b': 4.}, g, {'a': 2, 'b': 9}, power, .001)
            for n, a in {'a': 0., 'b': 4.}.items():
                assert raw[n] == ((a + .001) * (g[n] + .001) if g else a + .001) ** (-power)
            assert abs((2 * scales['a']**2 + 9 * scales['b']**2) / 11 - 1) < 1e-12
            if power == 0:
                assert scales == {'a': 1., 'b': 1.}
    op = LayerScaleOperator({'0': 3.})
    gradient = torch.randn(2, 4)
    torch.testing.assert_close(op.transform_gradient('0', gradient), 3 * gradient)
    baseline, stats = build_from_batches(model, 'baseline', iter(()))
    assert baseline is None and stats['builder_samples'] == 0 and stats['scale_min'] == 1


def main():
    torch.set_num_threads(2)
    for path in ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    subprocess.run(['bash', '-n', str(ROOT / 'run_all.sh')], check=True)
    assert POWERS == (.1, .2, .3, .4, .5) and len(RUNS) == 11
    expected = {1: ('dp_adamw', 'trace_ag_p01', 'trace_ag_p04', 'trace_a_p01'),
                2: ('trace_ag_p02', 'trace_ag_p05', 'trace_a_p02', 'trace_a_p04'),
                3: ('trace_ag_p03', 'trace_a_p03', 'trace_a_p05')}
    assert GPU_RUNS == expected and set(GPU_BY_RUN) == set(RUNS)
    assert sum(map(len, GPU_RUNS.values())) == 11
    for family in ('trace_a', 'trace_ag'):
        assert [p for f, p in RUNS.values() if f == family] == list(POWERS)
    shell = (ROOT / 'run_all.sh').read_text()
    for gpu, methods in expected.items():
        assert f'worker {gpu} {" ".join(methods)} &' in shell
    assert 'export CUDA_VISIBLE_DEVICES="$1"' in shell
    assert shell.index('if (( status != 0 ))') < shell.index('python -B exp37/analyze.py')
    assert (cfg.MAX_GRAD_NORM, cfg.DAMPING, cfg.SEED, cfg.EPOCHS) == (2., .001, 42, 5)
    assert (cfg.LOGICAL_BATCH_SIZE, cfg.PHYSICAL_BATCH_SIZE, cfg.EPSILON, cfg.DELTA) == (256, 128, 3, 1e-5)
    assert (cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.BETAS, cfg.ADAM_EPS) == (1e-4, .01, (.9, .999), 1e-8)
    assert (cfg.SYNTHETIC_BATCHES, cfg.SYNTHETIC_BATCH_SIZE, cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE) == (10, 256, 256)
    assert cfg.EPOCHS * (cfg.TRAIN_SAMPLES // cfg.LOGICAL_BATCH_SIZE) == 975
    assert cfg.MODEL_NAME == 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
    for gpu in GPU_RUNS:
        vit = worker.setup(GPU_RUNS[gpu][0])
        assert vit.reference.cfg is cfg and vit.reference.Clipper is Clipper
        assert vit.reference.synthetic_stream is reference.synthetic_stream
        assert vit.model_cfg.ROOT == ROOT / 'workers' / f'gpu{gpu}'
        assert vit.prepare_checkpoint.__globals__['ROOT'] == vit.model_cfg.ROOT
        for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'HF_HOME', 'HF_HUB_CACHE',
                    'HUGGINGFACE_HUB_CACHE', 'HF_XET_CACHE', 'TORCH_HOME', 'CUDA_CACHE_PATH',
                    'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TMPDIR', 'TMP', 'TEMP'):
            assert Path(os.environ[key]).resolve().is_relative_to(ROOT)
    assert cfg.RESULTS == ROOT / 'results/vit' and cfg.DATA_ROOT == ROOT.parent / 'data'
    with patch.object(vit.datasets, 'CIFAR10', wraps=vit.datasets.CIFAR10) as constructor:
        train, test = vit.load_data(download=False)
        assert (len(train), len(test)) == (50000, 10000)
        for call in constructor.call_args_list:
            assert Path(call.args[0]) == cfg.DATA_ROOT and call.kwargs['download'] is False
    transform = vit.reference.data_transform().transforms
    assert len(transform) == 3 and transform[0].size == (224, 224)
    assert transform[0].interpolation.value == 'bicubic'
    assert transform[2].mean == transform[2].std == (.5, .5, .5)
    assert worker.prepare.__globals__['ROOT'] == ROOT
    assert worker.MetricsSink.DataFrame.__globals__['ROOT'] == ROOT
    # Verify the metrics boundary writes locally and preserves cumulative counters.
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache/tmp') as temporary:
        directory = Path(temporary)
        row = dict(epoch=1, test_accuracy=.5, logical_steps=195, optimizer_steps=195,
                   noise_events=195, accountant_steps=195, parameters_finite=True,
                   parameters_updated=True, group_norm_attention_qkv=2.,
                   group_norm_attention_out=3., group_norm_patch_head=4.,
                   layer_norm_blocks_0_mlp_fc1=3., layer_norm_blocks_1_mlp_fc1=4.,
                   layer_norm_blocks_0_mlp_fc2=2.)
        sink = worker.FamilyMetricsSink(directory, 'vit', 'dp_adamw', lambda: {})
        sink.DataFrame([row]).to_csv(ROOT.parent / 'must_not_be_written.csv')
        assert (directory / 'metrics.csv').is_file()
        assert row['family_norm_mlp_fc1'] == 5. and row['family_norm_mlp_fc2'] == 2.
        assert row['best_accuracy'] == .5
    numerical()
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache/tmp') as temporary:
        with patch.object(cfg, 'RESULTS', Path(temporary)):
            with TestCase().assertRaises(FileNotFoundError):
                analyze.load_complete()
    print('PASS: 11-run grid, fixed GPUs 1–3, protocol/data/cache isolation, trace formulas,')
    print('reference A/G traces and label RNG, weighted RMS, BK gradients/norms/clipping, Gaussian noise order.')
    print('No formal training started; no formal results generated.')


if __name__ == '__main__':
    main()

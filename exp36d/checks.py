"""CPU-only contract/numerical checks; no formal training or result writes."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36d.config import ROOT, cfg, RUNS, POWERS, GPU_RUNS, GPU_BY_RUN
from exp36d.geometry import RawAOperator, builder, raw_spectrum
from exp36d import worker, analyze
from exp22 import geometry as reference
from exp22.methods import Clipper
import subprocess
import tempfile
from unittest.mock import patch
import torch
from torch import nn


def main():
    torch.set_num_threads(2)
    for path in ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    subprocess.run(['bash', '-n', str(ROOT / 'run_all.sh')], check=True)
    assert len(RUNS) == 6 and POWERS == (.1, .2, .3, .4, .5)
    assert {family for family, _ in RUNS.values()} == {'baseline', 'a_only'}
    assert sum(map(len, GPU_RUNS.values())) == 6
    assert set(m for runs in GPU_RUNS.values() for m in runs) == set(RUNS)
    expected = {0: ('dp_adamw', 'dp_kfc_a_p01'),
                1: ('dp_kfc_a_p02', 'dp_kfc_a_p05'),
                2: ('dp_kfc_a_p03',),
                3: ('dp_kfc_a_p04',)}
    assert GPU_RUNS == expected
    assert GPU_BY_RUN == {method: gpu for gpu, methods in expected.items() for method in methods}
    shell = (ROOT / 'run_all.sh').read_text()
    for gpu, methods in expected.items():
        assert f'worker {gpu} {" ".join(methods)} &' in shell
    assert cfg.MAX_GRAD_NORM == 2 and cfg.DAMPING == .001
    assert cfg.DATA_ROOT == ROOT.parent / 'data'
    assert (cfg.EPOCHS, cfg.SEED, cfg.LOGICAL_BATCH_SIZE, cfg.PHYSICAL_BATCH_SIZE) == (5, 42, 256, 128)
    assert (cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.BETAS, cfg.ADAM_EPS) == (1e-4, .01, (.9, .999), 1e-8)
    assert (cfg.SYNTHETIC_BATCHES, cfg.SYNTHETIC_BATCH_SIZE, cfg.EPSILON, cfg.DELTA) == (10, 256, 3, 1e-5)
    vit = worker.setup('dp_kfc_a_p05')
    assert vit.reference.cfg is cfg
    with patch.object(vit.datasets, 'CIFAR10', wraps=vit.datasets.CIFAR10) as constructor:
        train, test = vit.load_data(download=False)
        assert (len(train), len(test)) == (50000, 10000)
        for call in constructor.call_args_list:
            assert Path(call.args[0]) == ROOT.parent / 'data'
            assert call.kwargs['download'] is False
    assert vit.reference.synthetic_stream is reference.synthetic_stream
    for key in ('HF_HOME', 'TORCH_HOME', 'TMPDIR', 'MPLCONFIGDIR'):
        assert Path(os.environ[key]).is_relative_to(ROOT)
    torch.manual_seed(42)
    q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.double))
    eigenvalues = torch.tensor([0., .01, 1., 10.], dtype=torch.double)
    covariance = (q * eigenvalues) @ q.T
    factors = {'0': {'A': covariance, 'G': covariance * 2, 'output_dimension': 4}}
    for p in POWERS:
        matrix, gains = raw_spectrum(covariance, p)
        expected_gain = (eigenvalues + .001).pow(-p)
        torch.testing.assert_close(gains, expected_gain)
        torch.testing.assert_close(matrix, (q * expected_gain) @ q.T)
        a = RawAOperator(factors, p, .001)
        assert a.scale == 1 and a.power == p
        torch.testing.assert_close(a.data['0'], matrix)
        gradient = torch.randn(4, 4, dtype=torch.double)
        torch.testing.assert_close(a.transform_gradient('0', gradient), gradient @ matrix)
        assert a.diagnostics['operator_gain_max'] > 1  # Raw gains are not capped/normalized.
        assert 'layer_gain_min_0' in a.diagnostics and 'operator_gain_p99' in a.diagnostics
        for method in ('dp_kfc_a_bk',):
            model = nn.Sequential(nn.Linear(3, 2))
            model.num_classes = 2
            before = [v.detach().clone() for v in model.parameters()]
            x, y = torch.randn(4, 3), torch.tensor([0, 1, 1, 0])
            op, stats = builder(p)(model, method, [x], 42, 1, .001, p)
            assert all(torch.equal(v, old) for v, old in zip(model.parameters(), before))
            assert all(not m._forward_hooks for m in model.modules())
            assert stats['builder_samples'] == 4
            gradients = []
            for xi, yi in zip(x, y):
                w, b = torch.autograd.grad(nn.functional.cross_entropy(model(xi[None]), yi[None]), tuple(model.parameters()))
                gradients.append(op.transform_gradient('0', torch.cat((w, b[:, None]), 1)))
            norms = torch.stack(gradients).flatten(1).norm(dim=1)
            clipper = Clipper(model, op, method='bk', max_grad_norm=2)
            _, actual_norms, actual_factors, _, _ = clipper.aggregate_logical(x, y, 2)
            torch.testing.assert_close(actual_norms, norms)
            torch.testing.assert_close(actual_factors, (2 / (norms + 1e-6)).clamp(max=1))
            # Verify the inherited Gaussian noise uses sigma*C before averaging.
            old_grads = [v.grad.clone() for v in model.parameters()]
            expected_rng = torch.Generator().manual_seed(40042)
            expected_grads = [(g + .7 * 2 * torch.randn(v.numel(), generator=expected_rng).reshape_as(v)) / len(x)
                              for v, g in zip(model.parameters(), old_grads)]
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            clipper.step(optimizer, .7, len(x), torch.Generator().manual_seed(40042))
            for v, expected_grad in zip(model.parameters(), expected_grads):
                torch.testing.assert_close(v.grad, expected_grad)
            assert clipper.optimizer_steps == clipper.noise_events == 1
            clipper.remove()
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache/tmp') as temporary:
        with patch.object(cfg, 'RESULTS', Path(temporary)):
            try:
                analyze.load_complete()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing runs were accepted')
            assert not list(Path(temporary).iterdir())
    print('PASS: imports/compile, shell, 6-run grid/GPU assignment, protocol/data/cache paths,')
    print('raw A formulas, calibration, C=2 BK clipping/noise scaling, analysis gate.')
    print('No formal training started; no formal results generated.')


if __name__ == '__main__':
    main()

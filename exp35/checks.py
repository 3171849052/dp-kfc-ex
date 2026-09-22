"""CPU-only lightweight validation; never starts a formal training run."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp35 import ROOT, REPO
from exp35 import config as cfg
import torch
from torch import nn
from exp35.geometry import spectral_operator, SpectralAOperator, TOL
from exp35 import mnist, vit
from exp35.adapters import bind, MetricsSink
from exp22.geometry import build_a_operator
from exp22.methods import Clipper
from unittest.mock import patch
import tempfile
import shutil
import subprocess
from exp35 import analyze, worker
import pandas as pd


def main():
    subprocess.run(['bash', '-n', str(ROOT/'run_all.sh')], check=True)
    torch.set_num_threads(2)
    for path in ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    generator = torch.Generator().manual_seed(35)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.double))
    eigenvalues = torch.tensor([0., .01, .1, 1., 4., 10.], dtype=torch.double)
    covariance = (q * eigenvalues) @ q.T
    raw_gain = (eigenvalues + .001).pow(-.4)
    expected = {'raw':raw_gain, 'norm':raw_gain/raw_gain.max(), 'mix05':.5+.5*raw_gain/raw_gain.max()}
    for variant, gain in expected.items():
        matrix, actual_gain = spectral_operator(covariance, variant)
        torch.testing.assert_close(matrix, (q*gain)@q.T)
        torch.testing.assert_close(actual_gain, gain)
        torch.testing.assert_close(q.T @ matrix @ q, torch.diag(gain))
        if variant != 'raw':
            assert abs(torch.linalg.eigvalsh(matrix).max().item()-1) < TOL
        if variant == 'mix05':
            assert torch.linalg.eigvalsh(matrix).min() >= .5-TOL
        # Exercise the real Exp22 builder and BK aggregate, compare against a
        # direct per-example autograd calculation (no Explicit engine).
        model = nn.Sequential(nn.Linear(3, 2))
        x = torch.randn(4,3,generator=generator)
        y = torch.tensor([0,1,1,0])
        builder = bind(build_a_operator.__wrapped__,
            AOnlyOperator=lambda factors,power,damping: SpectralAOperator(factors,variant))
        with torch.no_grad():
            operator, stats = builder(model, [x], .4, .001)
        gradients = []
        for xi, yi in zip(x,y):
            loss = nn.functional.cross_entropy(model(xi[None]),yi[None])
            weight, bias = torch.autograd.grad(loss, tuple(model.parameters()))
            gradients.append(torch.cat((weight,bias[:,None]),dim=1) @ operator.data['0'])
        norms = torch.stack(gradients).flatten(1).norm(dim=1)
        factors = (1/(norms+1e-6)).clamp(max=1)
        clipper = Clipper(model, operator, method='bk', max_grad_norm=1)
        _, bk_norms, bk_factors, _, _ = clipper.aggregate_logical(x,y,2)
        torch.testing.assert_close(bk_norms, norms)
        torch.testing.assert_close(bk_factors, factors)
        clipper.remove()
    assert len(cfg.grid('mnist')) == len(cfg.grid('vit')) == 4
    assert len(set(cfg.grid('mnist'))) == len(set(cfg.grid('vit'))) == 4
    assert cfg.GPU == 2 and cfg.DATA_ROOT == REPO/'data'
    # Check both real datasets and the exact arguments at the loader boundary.
    for module, name, size in ((mnist,'MNIST',60000),(vit,'CIFAR10',50000)):
        original = getattr(module.datasets, name)
        with patch.object(module.datasets, name, wraps=original) as constructor:
            train, test = module.load_data()
            assert len(train.dataset if name=='MNIST' else train) == size
            for call in constructor.call_args_list:
                assert Path(call.args[0]) == REPO/'data'
                assert call.kwargs['download'] is False
    checkpoint = REPO/'exp30/.cache/huggingface/hub'/('models--timm--'+cfg.MODEL_NAME)
    revision = (checkpoint/'refs/main').read_text().strip()
    assert (checkpoint/'snapshots'/revision/'model.safetensors').is_file()
    with tempfile.TemporaryDirectory(dir=ROOT/'.cache/tmp') as temporary:
        with patch.object(analyze, 'ROOT', Path(temporary)):
            try:
                analyze.load_complete()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('analysis accepted absent runs')
            assert not list(Path(temporary).rglob('summary.csv'))
    # Reproduce sequential runs, including the symlink cache left by the old code.
    with tempfile.TemporaryDirectory(dir=ROOT/'.cache/tmp') as temporary:
        base = Path(temporary)
        relative = Path('.cache/huggingface/hub') / ('models--timm--'+cfg.MODEL_NAME)
        source = base/'exp30'/relative
        (source/'blobs').mkdir(parents=True)
        (source/'snapshots/revision').mkdir(parents=True)
        (source/'blobs/weights').write_bytes(b'toy checkpoint')
        (source/'snapshots/revision/model.safetensors').symlink_to('../../blobs/weights')
        destination = base/'exp35'/relative
        shutil.copytree(source, destination, symlinks=True)
        with patch.object(vit, 'REPO', base), patch.object(vit, 'ROOT', base/'exp35'):
            vit.prepare_checkpoint()
            vit.prepare_checkpoint()
        assert (destination/'snapshots/revision/model.safetensors').read_bytes() == b'toy checkpoint'
        with patch.object(vit, 'REPO', base), patch.object(vit, 'ROOT', base/'fresh'):
            vit.prepare_checkpoint()
            vit.prepare_checkpoint()
        assert not (base/'fresh'/relative/'snapshots/revision/model.safetensors').is_symlink()
    assert os.environ['HF_HOME'].startswith(str(ROOT))
    assert str(vit.model_cfg.ROOT) == str(ROOT)
    assert vit.reference.cfg is cfg
    # Ensure the metrics boundary preserves history and reconciles counters.
    with tempfile.TemporaryDirectory(dir=ROOT/'.cache/tmp') as temporary:
        sink = MetricsSink(Path(temporary),'mnist','dp_adam',lambda: {'parameters_finite':True,'parameters_updated':True})
        rows=[]
        for epoch in (1,2):
            rows.append(dict(epoch=epoch, accuracy=.1*epoch, epsilon_spent=epoch,
                norm_p50=1., norm_p90=2., norm_p99=3., norm_max=4.,
                logical_steps=2,optimizer_steps=2,noise_events=2,accountant_steps=2))
            sink.DataFrame(rows).to_csv('ignored')
        frame=pd.read_csv(Path(temporary)/'metrics.csv')
        assert frame.logical_steps.tolist()==[2,4]
        assert frame.best_accuracy.tolist()==[.1,.2]
    print('PASS: imports/compile/shell syntax, formulas/eigenvectors, unit norm gain, mix05 bounds,')
    print('Exp22 builder/BK norms, 4+4 grids, GPU 2, root datasets download=False, metrics counters/cache isolation, pretrained cache/repeated copy, incomplete-analysis gate.')


if __name__ == '__main__':
    main()

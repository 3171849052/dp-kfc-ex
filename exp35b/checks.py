"""CPU checks only; no formal result or training run is created."""
import argparse
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp35b.worker import setup
from exp35b.geometry import ALPHAS, GPU_BY_VARIANT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=tuple(ALPHAS), required=True)
    args = parser.parse_args()
    cfg = setup(args.variant)
    import subprocess
    import torch
    from unittest.mock import patch
    from exp35 import mnist, vit, geometry
    from exp35b import analyze, worker
    from exp35.adapters import bind
    from exp22.geometry import build_a_operator
    from exp22.methods import Clipper
    torch.set_num_threads(2)
    for path in cfg.ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    subprocess.run(['bash','-n',str(cfg.ROOT/'run_all.sh')], check=True)
    assert GPU_BY_VARIANT == {'alpha025':1, 'alpha075':2}
    shell = (cfg.ROOT/'run_all.sh').read_text()
    assert 'run_gpu 1 alpha025 &' in shell and 'run_gpu 2 alpha075 &' in shell
    assert 'for dataset in mnist vit' in shell
    alpha = ALPHAS[args.variant]
    values = torch.tensor([0.,.01,.1,1.,4.,10.], dtype=torch.double)
    q, _ = torch.linalg.qr(torch.randn(6,6,dtype=torch.double))
    covariance = (q*values)@q.T
    raw = (values+.001).pow(-.4)
    expected = 1-alpha+alpha*raw/raw.max()
    matrix, gain = geometry.spectral_operator(covariance,args.variant)
    torch.testing.assert_close(matrix,(q*expected)@q.T)
    torch.testing.assert_close(gain,expected)
    eig = torch.linalg.eigvalsh(matrix)
    assert eig.min() >= 1-alpha-2e-6 and eig.max() <= 1+2e-6
    model = torch.nn.Sequential(torch.nn.Linear(3,2))
    x, y = torch.randn(4,3), torch.tensor([0,1,1,0])
    build = bind(build_a_operator.__wrapped__, AOnlyOperator=lambda f,p,d: geometry.SpectralAOperator(f,args.variant))
    with torch.no_grad():
        operator, _ = build(model,[x],.4,.001)
    gradients = []
    for xi, yi in zip(x,y):
        loss = torch.nn.functional.cross_entropy(model(xi[None]),yi[None])
        weight, bias = torch.autograd.grad(loss,tuple(model.parameters()))
        gradients.append(torch.cat((weight,bias[:,None]),dim=1)@operator.data['0'])
    expected_norm = torch.stack(gradients).flatten(1).norm(dim=1)
    clipper = Clipper(model,operator,method='bk',max_grad_norm=1)
    _, norms, factors, _, _ = clipper.aggregate_logical(x,y,2)
    torch.testing.assert_close(norms,expected_norm)
    torch.testing.assert_close(factors,(1/(expected_norm+1e-6)).clamp(max=1))
    clipper.remove()
    for module, name, size in ((mnist,'MNIST',60000),(vit,'CIFAR10',50000)):
        with patch.object(module.datasets,name,wraps=getattr(module.datasets,name)) as constructor:
            train, test = module.load_data()
            assert len(train.dataset if name == 'MNIST' else train) == size
            for call in constructor.call_args_list:
                assert Path(call.args[0]) == cfg.REPO/'data'
                assert call.kwargs['download'] is False
    assert cfg.METHODS == ('dp_kfc_a_alpha025','dp_kfc_a_alpha075')
    assert vit.reference.cfg is cfg and vit.model_cfg.ROOT == cfg.CACHE_ROOT
    assert Path(os.environ['HF_HOME']).is_relative_to(cfg.ROOT)
    from huggingface_hub import constants
    assert Path(constants.HF_HUB_CACHE) == cfg.CACHE_ROOT/'.cache/huggingface/hub'
    cache = cfg.REPO/'exp30/.cache/huggingface/hub'/('models--timm--'+cfg.MODEL_NAME)
    revision = (cache/'refs/main').read_text().strip()
    assert (cache/'snapshots'/revision/'model.safetensors').is_file()
    print(f'PASS {args.variant}: compile/import/shell, GPU mapping, root data/download=False, spectral formula/bounds, Exp22 BK norms, local cache, pretrained source')


if __name__ == '__main__':
    main()

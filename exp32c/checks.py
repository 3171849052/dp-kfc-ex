"""Lightweight CPU checks; never runs a formal experiment."""
import ast
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp32c import config as cfg
import torch
from torch import nn
from exp32c.model import initialize, make_optimizer
from exp32c.geometry import conv_modules, build_from_batches
from exp32c.clipping import Clipper


def main():
    torch.set_num_threads(2)
    for path in cfg.ROOT.glob('*.py'):
        ast.parse(path.read_text())
    from exp32c import run, worker, analyze
    assert len(cfg.grid()) == 7
    assert [len(cfg.GPU_RUNS[i]) for i in range(4)] == [2, 2, 2, 1]
    assert cfg.LOGICAL_BATCH_SIZE == cfg.PHYSICAL_BATCH_SIZE == 256
    assert cfg.TRAIN_SAMPLES // 256 == 195 and 195*cfg.EPOCHS == 975
    assert cfg.SEED == 42 and cfg.WEIGHTS is None
    assert cfg.A_POWER == 0.4 and cfg.A_SCALE == 1.0
    assert cfg.DAMPING_VALUES == (1e-3, 1e-2, 1e-1)
    from unittest.mock import patch
    with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('pretrained download forbidden')):
        model = initialize(cfg.SEED, 'cpu')
        repeat = initialize(cfg.SEED, 'cpu')
    assert all(torch.equal(v, repeat.state_dict()[k]) for k, v in model.state_dict().items())
    del repeat
    # Compare source against Exp32: only initialization/import/docstring differ.
    original = (cfg.ROOT.parent / 'exp32' / 'model.py').read_text()
    expected = original.replace('exp32', 'exp32c').replace(
        'ImageNet initialization', 'Random initialization').replace(
        ', SqueezeNet1_1_Weights', '').replace(
        'weights=SqueezeNet1_1_Weights.IMAGENET1K_V1', 'weights=None')
    assert (cfg.ROOT / 'model.py').read_text() == expected
    for name in ('geometry.py', 'clipping.py', 'worker.py', 'analyze.py'):
        assert (cfg.ROOT / name).read_text() == (cfg.ROOT.parent / 'exp32' / name).read_text().replace('exp32', 'exp32c').replace('Exp32', 'Exp32c')
    train, test = run.load_data(download=False)
    assert len(train) == 50000 and len(test) == 10000
    assert all(not m.inplace for m in model.modules() if isinstance(m, nn.ReLU))
    assert all(p.requires_grad for p in model.parameters())
    assert model.classifier[0].p == 0 and model.classifier[1].out_channels == 10
    assert len(conv_modules(model)) == 26
    optimizer = make_optimizer(model)
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.defaults['lr'] == 1e-4 and optimizer.defaults['weight_decay'] == 0
    assert optimizer.defaults['betas'] == (0.9, 0.999) and optimizer.defaults['eps'] == 1e-8
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 10)
    print(f'Random SqueezeNet: {len(conv_modules(model))} convolutions, '
          f'{sum(p.numel() for p in model.parameters())} trainable parameters')
    # Exact per-example oracle on a tiny Conv2d network including bias and padding.
    for method in cfg.METHODS:
        torch.manual_seed(42)
        small = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                              nn.Conv2d(4, 10, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        x, y = torch.randn(2, 3, 5, 5), torch.tensor([2, 7])
        op, stats = build_from_batches(small, method, [x], 42, 1,
                                      damping=None if method == 'dp_adam' else 0.01)
        if op:
            assert op.scale == 1.0 and len(op.data) == 2
        expected, squared = [], []
        for i in range(2):
            small.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(small(x[i:i+1]), y[i:i+1]).backward()
            gradients = []
            for name, module in conv_modules(small).items():
                g = torch.cat([module.weight.grad.flatten(1), module.bias.grad[:, None]], 1)
                g = op.transform_matrix(name, g) if op else g
                gradients.extend([g[:, :-1].reshape_as(module.weight), g[:, -1]])
            expected.append(gradients)
            squared.append(sum(g.square().sum() for g in gradients))
        norms = torch.stack(squared).sqrt()
        factors = (1/(norms+1e-6)).clamp(max=1)
        clipper = Clipper(small, op)
        _, actual, _, _, diagnostic = clipper.aggregate_logical(x, y, 2)
        torch.testing.assert_close(actual, norms, rtol=2e-4, atol=2e-5)
        for j, parameter in enumerate(small.parameters()):
            target = sum(expected[i][j]*factors[i] for i in range(2))
            torch.testing.assert_close(parameter.grad, target, rtol=2e-4, atol=2e-5)
        clipper.step(make_optimizer(small), 0.5, 2, torch.Generator().manual_seed(40042))
        assert clipper.optimizer_steps == clipper.noise_events == 1
        assert all(torch.isfinite(p).all() for p in small.parameters())
        clipper.remove()
        print(method, 'exact transformed norms/aggregates and noisy Adam step passed')
    print('Syntax/import, grid, GPU assignment, model and tiny smoke checks passed.')

if __name__ == '__main__':
    main()

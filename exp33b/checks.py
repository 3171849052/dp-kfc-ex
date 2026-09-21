"""CPU-only formula, integration, protocol and artifact checks; no formal training."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33b import config as cfg
from exp33b.run import initialize, data_transform, private_loader
from exp33b.wiener import Wiener, build_wiener, matrix, private_step, make_optimizer, diagnostic_parameter_groups
from exp22 import config as base
from exp22.methods import Clipper
from exp22.model import convert_vit
import copy
import inspect
from itertools import product
import json
import os
import tempfile
import torch
from torch import nn
import pandas as pd
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 6)
        self.norm = nn.LayerNorm(6)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'attn': nn.ModuleDict({name: nn.Linear(6, 6) for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj')}),
            'mlp': nn.ModuleDict({'fc1': nn.Linear(6, 8), 'fc2': nn.Linear(8, 6)}),
        })])
        self.head = nn.Linear(6, 10)

    def forward(self, x):
        x = torch.tanh(self.norm(self.patch_embed(x)))
        block = self.blocks[0]
        attention = block['attn']
        x = x + attention['out_proj'](sum(attention[n](x) for n in ('q_proj', 'k_proj', 'v_proj')))
        x = x + block['mlp']['fc2'](torch.tanh(block['mlp']['fc1'](x)))
        return self.head(x)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    expected = [f'wiener_a_{mode}_lr_{lr}' for lr in ('1e-4', '3e-4', '1e-3', '3e-3') for mode in ('none', 'rms')]
    assert list(cfg.RUNS) == expected and len(cfg.grid()) == 8
    assert list(cfg.GPU_RUNS) == [0, 1, 2, 3]
    for gpu, lr in enumerate(cfg.LINEAR_LRS):
        assert len(cfg.GPU_RUNS[gpu]) == 2
        assert [cfg.RUNS[n]['scale_mode'] for n in cfg.GPU_RUNS[gpu]] == ['none', 'rms_match']
        assert all(cfg.RUNS[n]['linear_lr'] == lr for n in cfg.GPU_RUNS[gpu])
    for key in ('MODEL_NAME', 'IMG_SIZE', 'EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'BETAS', 'ADAM_EPS',
                'EPSILON', 'DELTA', 'MAX_GRAD_NORM', 'TRAIN_SAMPLES', 'LOGICAL_BATCH_SIZE',
                'PHYSICAL_BATCH_SIZE', 'ACCUMULATION_STEPS', 'SYNTHETIC_BATCHES', 'SYNTHETIC_BATCH_SIZE', 'SYNTHETIC_ALPHA'):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.SEED == 42 and cfg.IDENTITY_LR == 1e-4 and cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE == 128
    assert cfg.TRAIN_SAMPLES // cfg.LOGICAL_BATCH_SIZE == 195
    assert cfg.DATA_ROOT == cfg.ROOT.parent / 'exp30' / 'data'
    assert (cfg.DATA_ROOT / 'cifar-10-batches-py' / 'data_batch_1').is_file()
    from exp22.model import initialize as original_initialize
    assert initialize is original_initialize and 'pretrained=True' in inspect.getsource(initialize)
    assert 'vit_tiny_patch16_224.augreg_in21k_ft_in1k' == cfg.MODEL_NAME
    transform = data_transform().transforms
    assert len(transform) == 3 and transform[0].size == (224, 224)
    assert transform[0].interpolation.value == 'bicubic'
    assert transform[2].mean == (0.5,) * 3 and transform[2].std == (0.5,) * 3
    sigma = get_noise_multiplier(target_epsilon=3., target_delta=1e-5, sample_rate=256/50000, steps=975, accountant='rdp')
    accountant = RDPAccountant()
    for _ in range(975):
        accountant.step(noise_multiplier=sigma, sample_rate=256/50000)
    epsilon = accountant.get_epsilon(1e-5)
    assert epsilon <= 3. and sum(v[2] for v in accountant.history) == 975

    model = Toy()
    xs = [torch.randn(4, 4), torch.randn(4, 4)]
    params = list(model.parameters())
    matrices = {name: [] for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    label_rng = torch.Generator().manual_seed(42 + 20000 + 1)
    clipper = Clipper(model)
    for x in xs:
        y = torch.randint(10, (len(x),), generator=label_rng)
        per_sample = [torch.autograd.grad(nn.functional.cross_entropy(model(xi[None]), yi[None]), params) for xi, yi in zip(x, y)]
        norms = torch.stack([sum(g.square().sum() for g in grads).sqrt() for grads in per_sample])
        factors = (1 / (norms + 1e-6)).clamp(max=1)
        exact = [sum(f * grads[j] for f, grads in zip(factors, per_sample)) for j in range(len(params))]
        _, actual_norms, actual_factors, _, _ = clipper.aggregate_logical(x, y, 2)
        torch.testing.assert_close(actual_norms, norms)
        torch.testing.assert_close(actual_factors, factors)
        for p, g in zip(params, exact):
            torch.testing.assert_close(p.grad, g, atol=2e-6, rtol=2e-5)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                matrices[name].append(matrix(module).clone().double() / len(x))
    clipper.remove()
    covariance = {n: {'A': sum(z.T @ z / z.shape[0] for z in zs) / len(zs)} for n, zs in matrices.items()}
    ops = {}
    for mode in cfg.SCALE_MODES:
        op, stats = build_wiener(model, 42, 1, sigma, mode, batches=xs, physical_batch_size=2)
        ops[mode] = op
        assert stats['builder_samples'] == 8 and stats['builder_logical_batches'] == 2 and stats['builder_noise_events'] == 0
        for name, zs in matrices.items():
            vals, vecs = torch.linalg.eigh(covariance[name]['A'])
            vals = vals.clamp_min(0)
            gain = vals / (vals + (sigma / 256)**2)
            raw_energy = sum(z.square().sum() for z in zs) / len(zs)
            filtered_energy = sum((((z @ vecs) * gain) @ vecs.T).square().sum() for z in zs) / len(zs)
            scale = (raw_energy / filtered_energy).sqrt() if mode == 'rms_match' else torch.tensor(1.)
            ua, actual_gain, actual_scale = op.data[name]
            torch.testing.assert_close(actual_gain, gain.float(), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual_scale, scale.float())
            for z in zs:
                expected_z = scale * (((z @ vecs) * gain) @ vecs.T)
                torch.testing.assert_close(op.transform(name, z.float()), expected_z.float(), atol=1e-6, rtol=1e-5)
            if mode == 'none':
                assert actual_scale.item() == 1.
            else:
                matched = sum(op.transform(name, z.float()).double().square().sum() for z in zs) / len(zs)
                torch.testing.assert_close(matched, raw_energy, rtol=1e-5, atol=1e-7)
    for name in matrices:
        torch.testing.assert_close(ops['none'].data[name][0], ops['rms_match'].data[name][0], rtol=0, atol=0)
        torch.testing.assert_close(ops['none'].data[name][1], ops['rms_match'].data[name][1], rtol=0, atol=0)
    # Strong attenuation oracle: effective gain is allowed above one, no scale cap.
    diagonal = {'head': {'A': torch.diag(torch.tensor([0., 1e-12, 1e-8], dtype=torch.float64))}}
    diagonal_op = Wiener(diagonal, .5, 'rms_match')
    assert diagonal_op.data['head'][2] > 1e6
    assert diagonal_op.diagnostics['effective_gain_max'] > 1.
    assert Wiener(diagonal, .5, 'none').data['head'][1][0] == 0
    # Epoch labels change the fitted operator; repeated builds are deterministic.
    repeated, _ = build_wiener(model, 42, 1, sigma, 'rms_match', batches=xs, physical_batch_size=2)
    next_epoch, _ = build_wiener(model, 42, 2, sigma, 'rms_match', batches=xs, physical_batch_size=2)
    for name in matrices:
        for a, b in zip(ops['rms_match'].data[name], repeated.data[name]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(not torch.equal(repeated.data[n][2], next_epoch.data[n][2]) for n in matrices)

    # Independent explicit noise/filter/AdamW path, two steps to exercise moments.
    for lr, mode in product(cfg.LINEAR_LRS, cfg.SCALE_MODES):
        trained = copy.deepcopy(model)
        reference = copy.deepcopy(model)
        opt = make_optimizer(trained, lr)
        ref_opt = make_optimizer(reference, lr)
        assert len(opt.param_groups) == 2
        assert [g['lr'] for g in opt.param_groups] == [lr, 1e-4]
        assert all(g['weight_decay'] == .01 and g['betas'] == (.9, .999) and g['eps'] == 1e-8 for g in opt.param_groups)
        assert {id(p) for p in opt.param_groups[1]['params']} == {id(p) for p in trained.norm.parameters()}
        clipper = Clipper(trained)
        rng = torch.Generator().manual_seed(40042)
        reference_rng = torch.Generator().manual_seed(40042)
        op = ops[mode]
        saved_operator = {n: tuple(t.clone() for t in values) for n, values in op.data.items()}
        for x in xs:
            y = torch.tensor([0, 1, 2, 3])
            clipper.aggregate_logical(x, y, 2)
            for p, q in zip(trained.parameters(), reference.parameters()):
                noise = torch.randn(p.numel(), generator=reference_rng).reshape_as(p)
                q.grad = p.grad.detach().clone().add_(noise, alpha=sigma).div_(len(x))
            for name, m in reference.named_modules():
                if isinstance(m, nn.Linear):
                    z = op.transform(name, matrix(m))
                    m.weight.grad.copy_(z[:, :-1])
                    m.bias.grad.copy_(z[:, -1])
            before = {n: p.detach().clone() for n, p in trained.named_parameters()}
            stats = private_step(clipper, opt, sigma, len(x), rng, op)
            ref_opt.step()
            for p, q in zip(trained.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            groups = diagnostic_parameter_groups(trained)
            for group in cfg.UPDATE_GROUPS:
                selected = [(n, p) for n, p in trained.named_parameters() if groups[id(p)] == group]
                u = sum(((p.detach() - before[n]).double().square().sum() for n, p in selected), torch.tensor(0., dtype=torch.float64)).sqrt()
                norm = sum((before[n].double().square().sum() for n, _ in selected), torch.tensor(0., dtype=torch.float64)).sqrt()
                assert u.item() > 0, group
                assert abs(stats[f'update_norm_{group}'] - u.item()) <= 1e-7
                assert abs(stats[f'relative_update_norm_{group}'] - (u / (norm + 1e-30)).item()) <= 1e-7
            for n in op.data:
                for a, b in zip(op.data[n], saved_operator[n]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert clipper.optimizer_steps == clipper.noise_events == 2
        clipper.remove()
    print('PASS: eight-run grid, 2/2/2/2 GPUs, protocol/RDP, synthetic clipping/covariance, scale oracle, epoch rebuild, two LR groups, post-noise filtering, real AdamW updates', flush=True)

    # Structure/conversion check avoids downloading pretrained weights.
    import timm
    source = timm.create_model(cfg.MODEL_NAME, pretrained=False).eval()
    vit = convert_vit(source).eval()
    assert sum(isinstance(m, nn.Linear) for m in vit.modules()) == 74
    assert all(p.requires_grad for p in vit.parameters())
    with torch.no_grad():
        probe = torch.randn(1, 3, 224, 224)
        torch.testing.assert_close(vit(probe), source(probe), rtol=1e-4, atol=2e-5)
    groups = diagnostic_parameter_groups(vit)
    assert set(groups.values()) == set(cfg.UPDATE_GROUPS)
    assert groups[id(vit.cls_token)] == groups[id(vit.pos_embed)] == 'identity'
    assert groups[id(vit.blocks[0].attn.q_proj.weight)] == 'attention_qkv'
    assert groups[id(vit.blocks[0].attn.out_proj.weight)] == 'attention_out'
    assert groups[id(vit.blocks[0].mlp.fc1.weight)] == 'mlp'
    assert groups[id(vit.head.weight)] == groups[id(vit.patch_embed.weight)] == 'patch_head'
    optimizer = make_optimizer(vit, 3e-3)
    linear_ids = {id(p) for m in vit.modules() if isinstance(m, nn.Linear) for p in m.parameters()}
    assert {id(p) for p in optimizer.param_groups[0]['params']} == linear_ids
    assert len(linear_ids) == 148
    from exp33b.wiener import synthetic_stream
    from exp22.geometry import synthetic_stream as original_stream
    assert synthetic_stream is original_stream
    a = next(synthetic_stream(42, 1, 'cpu', batches=1, batch_size=2))
    b = next(synthetic_stream(42, 1, 'cpu', batches=1, batch_size=2))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a.shape == (2, 3, 224, 224)
    dataset = torch.utils.data.TensorDataset(torch.arange(768))
    assert torch.equal(next(iter(private_loader(dataset, 42)))[0], next(iter(private_loader(dataset, 42)))[0])
    for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'TMPDIR'):
        assert Path(os.environ[key]).is_relative_to(cfg.ROOT / '.cache'), key
    print('PASS: shared pretrained initialization, 74 Linear explicit conversion, synthetic/shuffle RNG reproducibility, cache confinement', flush=True)

    from exp33b.analyze import analyze
    with tempfile.TemporaryDirectory(prefix='analysis-check-', dir=cfg.ROOT / '.cache') as tmp:
        results = Path(tmp)
        for name, spec in cfg.RUNS.items():
            directory = results / 'runs' / name
            directory.mkdir(parents=True)
            (directory / 'config.json').write_text(json.dumps({'method': name, **spec}))
            rows = []
            for epoch in range(1, 6):
                row = dict(method=name, **spec, epoch=epoch, logical_steps=195, accountant_steps=epoch*195,
                           optimizer_steps=epoch*195, noise_events=epoch*195, test_accuracy=.1*epoch,
                           accuracy_auc=.2*epoch, test_loss=1/epoch, clip_fraction=.5,
                           wiener_norm_ratio=.8, nsr_filtered=.2)
                for metric in ('update_norm', 'relative_update_norm'):
                    row.update({f'{metric}_{g}': .01 for g in cfg.UPDATE_GROUPS})
                for metric in ('wiener_scale', 'effective_gain'):
                    row.update({f'{metric}_{q}': 1. for q in ('median', 'p10', 'p90', 'p99')})
                rows.append(row)
            pd.DataFrame(rows).to_csv(directory / 'metrics.csv', index=False)
        summary, paired = analyze(results)
        assert len(summary) == 8 and len(paired) == 4
        assert (summary.epoch == 5).all() and (paired.exp33_dp_adamw_accuracy == .9276).all()
        assert (paired.delta_rms_vs_none == 0).all()
        assert len(list(results.glob('*.png'))) == 10
        # Incomplete grids must fail, not silently produce partial results.
        path = results / 'runs' / expected[0] / 'metrics.csv'
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        try:
            analyze(results)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete run accepted')
    print('PASS: epoch-5 summary (8), paired (4), fixed reference, ten plots, incomplete-run rejection', flush=True)
    print(f'ALL LIGHTWEIGHT CHECKS PASSED; sigma={sigma}; epsilon={epsilon}; no formal training started', flush=True)


if __name__ == '__main__':
    main()

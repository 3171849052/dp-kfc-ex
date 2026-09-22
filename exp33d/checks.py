"""CPU-only formula, integration, protocol and artifact checks; no formal training."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33d import config as cfg
from exp33d.run import initialize, data_transform, private_loader
from exp33d.wiener import Wiener, build_wiener, matrix, private_step, make_optimizer, diagnostic_parameter_groups
from exp22 import config as base
from exp22.methods import Clipper
from exp22.model import convert_vit
import copy
import ast
import inspect
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
    expected = ['dp_adamw', 'wiener_a_beta_1'] + [f'wiener_a_beta_1e-{i}' for i in range(2, 8)]
    assert list(cfg.RUNS) == expected and len(cfg.grid()) == 8
    assert cfg.BETAS == (1., 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    assert all(gpu == 3 for gpu, _ in cfg.grid())
    shell = (cfg.ROOT / 'run_all.sh').read_text()
    assert 'export CUDA_VISIBLE_DEVICES=3' in shell and ' &\n' not in shell
    assert 'for method in ' + ' '.join(expected) + '; do' in shell
    for key in ('MODEL_NAME', 'IMG_SIZE', 'EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'ADAM_EPS',
                'EPSILON', 'DELTA', 'MAX_GRAD_NORM', 'TRAIN_SAMPLES', 'LOGICAL_BATCH_SIZE',
                'PHYSICAL_BATCH_SIZE', 'ACCUMULATION_STEPS', 'SYNTHETIC_BATCHES', 'SYNTHETIC_BATCH_SIZE', 'SYNTHETIC_ALPHA'):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.SEED == 42 and cfg.LEARNING_RATE == 1e-4 and cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE == 128
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
    for beta in cfg.BETAS:
        op, stats = build_wiener(model, 42, 1, sigma, beta, batches=xs, physical_batch_size=2)
        ops[beta] = op
        assert stats['builder_samples'] == 8 and stats['builder_logical_batches'] == 2 and stats['builder_noise_events'] == 0
        assert stats['wiener_gain_min'] >= 0
        assert stats['wiener_gain_max'] <= 1 + 1e-7
        for name, zs in matrices.items():
            vals, vecs = torch.linalg.eigh(covariance[name]['A'])
            gain = vals.clamp_min(0) / (vals.clamp_min(0) + beta * (sigma / 256)**2)
            ua, actual_gain = op.data[name]
            torch.testing.assert_close(actual_gain, gain.float(), atol=1e-6, rtol=1e-5)
            for z in zs:
                z = z.float()
                expected_z = (((z.double() @ vecs) * gain) @ vecs.T).float()
                torch.testing.assert_close(op.transform(name, z), expected_z, atol=1e-6, rtol=1e-5)
                if beta == 1:
                    # Exact Exp33 Wiener-A float32 execution order.
                    assert torch.equal(op.transform(name, z), ((z @ ua) * actual_gain) @ ua.T)
    repeated, _ = build_wiener(model, 42, 1, sigma, 1e-3, batches=xs, physical_batch_size=2)
    next_epoch, _ = build_wiener(model, 42, 2, sigma, 1e-3, batches=xs, physical_batch_size=2)
    for name in matrices:
        for a, b in zip(ops[1e-3].data[name], repeated.data[name]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(not torch.equal(repeated.data[n][1], next_epoch.data[n][1]) for n in matrices)

    # Execute only Exp33's definitions, avoiding its package cache side effects.
    from exp22.methods import _group
    from exp22.geometry import synthetic_stream
    source_path = cfg.ROOT.parent / 'exp33' / 'wiener.py'
    source_tree = ast.parse(source_path.read_text())
    definitions = ast.Module(body=[node for node in source_tree.body
                                  if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
    original = dict(torch=torch, nn=nn, Clipper=Clipper, _group=_group,
                    cfg=cfg, synthetic_stream=synthetic_stream)
    exec(compile(definitions, str(source_path), 'exec'), original)
    original_op, original_stats = original['build_wiener'](
        model, 42, 1, sigma, False, batches=xs, physical_batch_size=2)
    for name, zs in matrices.items():
        assert torch.equal(original_op.data[name][0], ops[1.].data[name][0])
        assert torch.equal(original_op.data[name][2], ops[1.].data[name][1])
        for z in zs:
            assert torch.equal(original_op.transform(name, z.float()), ops[1.].transform(name, z.float()))
    for stat in ('mean', 'median', 'p10', 'p90', 'p99', 'min', 'max'):
        assert original_stats[f'wiener_gain_{stat}'] == ops[1.].diagnostics[f'wiener_gain_{stat}']
    print('PASS: direct Exp33 builder, raw gains and beta=1 transforms are bitwise equal', flush=True)

    # Independent explicit noise/filter/AdamW path, two steps to exercise moments.
    for beta in (None, *cfg.BETAS):
        trained = copy.deepcopy(model)
        reference = copy.deepcopy(model)
        opt = make_optimizer(trained)
        ref_opt = make_optimizer(reference)
        assert len(opt.param_groups) == 1
        assert [g['lr'] for g in opt.param_groups] == [1e-4]
        assert all(g['weight_decay'] == .01 and g['betas'] == (.9, .999) and g['eps'] == 1e-8 for g in opt.param_groups)
        clipper = Clipper(trained)
        rng = torch.Generator().manual_seed(40042)
        reference_rng = torch.Generator().manual_seed(40042)
        op = None if beta is None else ops[beta]
        saved_operator = {n: tuple(t.clone() for t in values) for n, values in (op.data.items() if op is not None else [])}
        for x in xs:
            y = torch.tensor([0, 1, 2, 3])
            clipper.aggregate_logical(x, y, 2)
            clean_shadow = [p.grad.detach().clone() / len(x) for p in trained.parameters()]
            for p, q in zip(trained.parameters(), reference.parameters()):
                noise = torch.randn(p.numel(), generator=reference_rng).reshape_as(p)
                q.grad = p.grad.detach().clone().add_(noise, alpha=sigma).div_(len(x))
            for name, m in reference.named_modules():
                if isinstance(m, nn.Linear):
                    z = matrix(m)
                    if beta is not None:
                        ua, gain = op.data[name]
                        # Independent formula oracle, no production transform call.
                        vals = torch.linalg.eigvalsh(covariance[name]['A']).clamp_min(0)
                        h = (vals / (vals + beta * (sigma/256)**2)).float()
                        z = ((z @ ua) * h) @ ua.T
                    m.weight.grad.copy_(z[:, :-1])
                    m.bias.grad.copy_(z[:, -1])
            before = {n: p.detach().clone() for n, p in trained.named_parameters()}
            stats = private_step(clipper, opt, sigma, len(x), rng, op)
            # Independently verify post-filter shadow diagnostics over all parameters.
            flat = torch.cat([p.grad.flatten().double() for p in reference.parameters()])
            clean_flat = torch.cat([c.flatten().double() for c in clean_shadow])
            cosine = torch.dot(flat, clean_flat) / (flat.norm() * clean_flat.norm())
            nsr = (flat-clean_flat).norm() / clean_flat.norm()
            assert abs(stats['cos_raw_clean' if beta is None else 'cos_filtered_clean'] - cosine.item()) < 1e-6
            assert abs(stats['nsr_raw' if beta is None else 'nsr_filtered'] - nsr.item()) < 1e-5
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
            for n in saved_operator:
                for a, b in zip(op.data[n], saved_operator[n]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert clipper.optimizer_steps == clipper.noise_events == 2
        clipper.remove()
    print('PASS: eight-run grid, GPU 3, protocol/RDP, synthetic clipping/covariance, beta formula/bounds, epoch rebuild, fixed LR, post-noise filtering, real AdamW updates', flush=True)

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
    optimizer = make_optimizer(vit)
    linear_ids = {id(p) for m in vit.modules() if isinstance(m, nn.Linear) for p in m.parameters()}
    assert {id(p) for p in optimizer.param_groups[0]['params']} == {id(p) for p in vit.parameters()}
    assert len(linear_ids) == 148
    from exp33d.wiener import synthetic_stream
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

    # Fixed-spectrum monotonicity and exact zero/near-zero definitions.
    factors = {'head': {'A': torch.diag(torch.tensor([0., 1e-10, 1., 10.], dtype=torch.float64))}}
    previous = None
    for beta in cfg.BETAS:
        op = Wiener(factors, 1., beta)
        gain = op.data['head'][1]
        if previous is not None:
            assert (gain >= previous).all()
        previous = gain
        assert op.diagnostics['eig_zero_fraction'] == .25
        assert op.diagnostics['eig_near_zero_fraction'] == .5
        assert op.diagnostics['effective_rank_fraction'] == .5
        assert op.diagnostics['signal_to_noise_max'] == 10.

    from exp33d.analyze import analyze, PAIRED_FIELDS
    from exp33d.wiener import empty_wiener_diagnostics
    with tempfile.TemporaryDirectory(prefix='analysis-check-', dir=cfg.ROOT / '.cache') as tmp:
        results = Path(tmp)
        for name, spec in cfg.RUNS.items():
            directory = results / 'runs' / name
            directory.mkdir(parents=True)
            (directory / 'config.json').write_text(json.dumps({'method': name, **spec}))
            rows = []
            for epoch in range(1, 6):
                row = dict.fromkeys(PAIRED_FIELDS, .5)
                row.update(empty_wiener_diagnostics() if spec['beta'] is None else op.diagnostics)
                row.update(method=name, **spec, epoch=epoch, logical_steps=195, accountant_steps=epoch*195,
                           optimizer_steps=epoch*195, noise_events=epoch*195, test_accuracy=.1*epoch,
                           accuracy_auc=.2*epoch, test_loss=1/epoch, clip_fraction=.5,
                           nsr_raw=.3, cos_raw_clean=.4, mean_clip_factor=.6)
                for metric in ('update_norm', 'relative_update_norm'):
                    row.update({f'{metric}_{g}': .01 for g in cfg.UPDATE_GROUPS})
                rows.append(row)
            pd.DataFrame(rows).to_csv(directory / 'metrics.csv', index=False)
        summary, paired = analyze(results)
        assert len(summary) == 8 and len(paired) == 7 and (summary.epoch == 5).all()
        assert (paired.delta_accuracy_vs_adamw == 0).all()
        assert len(list(results.glob('*.png'))) == 12
        path = results / 'runs' / expected[0] / 'metrics.csv'
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        try:
            analyze(results)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete run accepted')
    print('PASS: eight sequential GPU-3 runs, monotone beta gains, spectrum diagnostics, summary (8), paired (7), twelve plots', flush=True)
    print(f'ALL LIGHTWEIGHT CHECKS PASSED; sigma={sigma}; epsilon={epsilon}; no formal training started', flush=True)


if __name__ == '__main__':
    main()

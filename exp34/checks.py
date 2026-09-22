"""CPU-only lightweight checks; no dataset downloads or formal training."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp34 import config as cfg
from exp34.model import initialize, convert_vit
from exp34.geometry import spectrum_rows, remove_a_scale, build_geometry, GEOMETRY_COLUMNS
from exp22.geometry import AOnlyOperator, FullKFACOperator, build_from_batches
from exp22.methods import Clipper
import torch
from torch import nn
from unittest.mock import patch
import timm
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant
import pandas as pd
import tempfile
import json


def main():
    torch.set_num_threads(4)
    assert cfg.DATA_ROOT == cfg.ROOT.parent / "exp30" / "data"
    assert (cfg.DATA_ROOT / "cifar-10-batches-py" / "data_batch_1").is_file()
    assert (cfg.DATA_ROOT / "cifar-10-batches-py" / "test_batch").is_file()
    original = timm.create_model
    calls = []
    def create(*args, **kwargs):
        assert kwargs['pretrained'] is False
        assert torch.initial_seed() == cfg.SEED
        calls.append(kwargs)
        return original(*args, **kwargs)
    with patch('timm.create_model', side_effect=create):
        model = initialize(cfg.SEED)
        same = initialize(cfg.SEED)
    assert len(calls) == 2
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), same.parameters()))
    assert all(p.requires_grad for p in model.parameters())
    assert model.embed_dim == 192 and len(model.blocks) == 12 and model.patch_size == (16, 16)
    assert all(b.attn.num_heads == 3 for b in model.blocks)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    norms = {n for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)}
    assert len(linears) == 74 and len(norms) == 25
    count = sum(p.numel() for p in model.parameters())
    source = original(cfg.MODEL_NAME, pretrained=False)
    reference = convert_vit(source)
    reference.head = nn.Linear(192, 10)
    assert count == sum(p.numel() for p in reference.parameters()) == 5526346
    model.eval(); source.eval()
    converted = convert_vit(source).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        torch.testing.assert_close(source(x), converted(x), atol=2e-5, rtol=2e-4)
    # Coverage only needs layer names; no expensive full-model geometry build.
    class Coverage:
        data = dict.fromkeys(linears)
    assert cfg.MAX_GRAD_NORM == 2.
    clipper = Clipper(model, Coverage(), method='bk', max_grad_norm=cfg.MAX_GRAD_NORM)
    assert set(clipper.preconditioned_layers) == linears
    assert set(clipper.identity_geometry_layers) == norms | {'cls_token', 'pos_embed'}
    clipper.remove()
    baseline = Clipper(model, None, method='bk', max_grad_norm=cfg.MAX_GRAD_NORM)
    assert not baseline.preconditioned_layers
    baseline.remove()
    assert cfg.EPOCHS == 20 and cfg.TOTAL_STEPS == 3900
    assert (cfg.LOGICAL_BATCH_SIZE, cfg.PHYSICAL_BATCH_SIZE, cfg.ACCUMULATION_STEPS) == (256,128,2)
    assert (cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.BETAS, cfg.ADAM_EPS) == (1e-4,.01,(.9,.999),1e-8)
    assert (cfg.SYNTHETIC_BATCHES, cfg.SYNTHETIC_BATCH_SIZE, cfg.SYNTHETIC_ALPHA, cfg.A_POWER) == (10,256,1.,.4)
    assert len(cfg.grid()) == 7 and tuple(cfg.GPU_RUNS) == (0, 1)
    assert [len(v) for v in cfg.GPU_RUNS.values()] == [4, 3]
    sigma = get_noise_multiplier(target_epsilon=3, target_delta=1e-5, sample_rate=256/50000, steps=3900, accountant='rdp')
    old = get_noise_multiplier(target_epsilon=3, target_delta=1e-5, sample_rate=256/50000, steps=975, accountant='rdp')
    assert sigma != old
    accountant = RDPAccountant()
    for _ in range(3900):
        accountant.step(noise_multiplier=sigma, sample_rate=256/50000)
    epsilon = accountant.get_epsilon(1e-5)
    assert epsilon <= 3
    # Both actual builders on a tiny classifier; test explicit damping and diagnostics purity.
    tiny = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 10))
    for method in ('dp_kfc', 'dp_kfc_a'):
        batches = [torch.randn(4,3)]
        with patch('exp34.geometry.synthetic_stream', return_value=iter(batches)):
            operator, stats, rows = build_geometry(tiny, method, .01, 42, 1, 'cpu')
        assert operator.damping == .01 and len(rows) == 2
        gradient = torch.randn(4,4)
        expected = operator.transform_gradient('0', gradient).clone()
        rng = torch.get_rng_state().clone()
        again = spectrum_rows(operator, method, .01, 1)
        assert torch.equal(rng, torch.get_rng_state())
        torch.testing.assert_close(operator.transform_gradient('0', gradient), expected, atol=0, rtol=0)
        if method == 'dp_kfc_a':
            assert operator.scale == 1 and operator.power == .4
            assert all('G_trace_per_dim' not in r for r in again)
        else:
            reference_op, _ = build_from_batches(tiny, 'dp_kfc', batches, 42, 1, damping=.01)
            torch.testing.assert_close(reference_op.transform_gradient('0', gradient), expected, atol=0, rtol=0)
        # One tiny logical batch checks BK accumulation/noise event semantics.
        clipping = Clipper(tiny, operator, method='bk', max_grad_norm=cfg.MAX_GRAD_NORM)
        optimizer = torch.optim.AdamW(tiny.parameters(), lr=cfg.LEARNING_RATE,
                                     betas=cfg.BETAS, eps=cfg.ADAM_EPS, weight_decay=cfg.WEIGHT_DECAY)
        _, _, _, _, stats = clipping.aggregate_logical(torch.randn(4,3), torch.tensor([0,1,2,3]), 2)
        clipping.step(optimizer, sigma, 4, torch.Generator().manual_seed(40042))
        assert clipping.optimizer_steps == clipping.noise_events == 1
        assert stats['fallback_temporary_grad_bytes'] == 0
        assert set(stats['layer_strategies'].values()) == {'bk_ghost'}
        assert stats['cache_empty_after_step']
        clipping.remove()
    # Exercise analysis on clearly isolated synthetic fixtures, including all eleven plots.
    from exp34.analyze import analyze
    from exp34.geometry import GROUPS
    with tempfile.TemporaryDirectory(dir=cfg.ROOT / '.cache', prefix='checks-') as tmp:
        output = Path(tmp)
        for _, method, damping in cfg.grid():
            path = output / 'runs' / cfg.run_name(method, damping)
            path.mkdir(parents=True)
            rows = []
            for epoch in range(1,21):
                row = dict(method=method,damping=damping,epoch=epoch,test_accuracy=.1+epoch*.001,
                           test_loss=2.,accuracy_auc=.1*max(0,epoch-1),clip_fraction=.9,
                           mean_clip_factor=.01,transformed_norm_p99=10.)
                row.update({k:epoch*195 for k in ('logical_steps','accountant_steps','optimizer_steps','noise_events')})
                row.update({f'{g}_median_{m}':1. for g in GROUPS for m in
                            ('A_trace_per_dim','G_trace_per_dim','relative_damping_A','relative_damping_G')})
                rows.append(row)
            pd.DataFrame(rows).to_csv(path/'metrics.csv',index=False)
        analyze(output)
        assert len(pd.read_csv(output/'summary.csv')) == 7
        assert len(pd.read_csv(output/'paired.csv')) == 3
        assert len(list(output.glob('*.png'))) == 11
    report = dict(status='passed', trainable_parameter_count=count, linear_layers=74,
                  layernorm_layers=25, total_steps=3900, noise_multiplier=sigma,
                  epsilon=epsilon, exp30_five_epoch_sigma=old, formal_training_started=False)
    (cfg.ROOT/'logs'/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

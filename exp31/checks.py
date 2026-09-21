"""Lightweight real-model checks; never starts the formal grid."""
import ast
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp31 import config as cfg
from exp31.run import initialize_backbone, initialize
from exp31.lora import inject_lora, selected_layers, parameter_metrics
from exp31.geometry import build_geometry, synthetic_stream
from exp31.clipping import Clipper
import torch
from opacus.accountants.utils import get_noise_multiplier


def main():
    torch.set_num_threads(4)
    for path in cfg.ROOT.glob('*.py'):
        ast.parse(path.read_text())
    assert len(cfg.grid()) == 7
    assert [len(cfg.GPU_RUNS[i]) for i in range(4)] == [2, 2, 2, 1]
    assert cfg.LOGICAL_BATCH_SIZE == cfg.PHYSICAL_BATCH_SIZE == 256
    assert cfg.ACCUMULATION_STEPS == 1 and cfg.LEARNING_RATE == 1e-3
    assert cfg.TRAIN_SAMPLES // cfg.LOGICAL_BATCH_SIZE * cfg.EPOCHS == 975
    device = torch.device('cuda:0')
    model = initialize_backbone(cfg.SEED, device).eval()
    x = next(synthetic_stream(cfg.SEED, 1, device, batches=1, batch_size=2))
    y = torch.tensor([1, 3], device=device)
    model.requires_grad_(False)
    with torch.no_grad():
        original = model(x)
    model = initialize(cfg.SEED, device).eval()
    names = selected_layers(model)
    assert len(names) == 48
    assert all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if '.lora_B.' in n)
    with torch.no_grad():
        assert torch.equal(original, model(x))
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    sigma = get_noise_multiplier(target_epsilon=cfg.EPSILON, target_delta=cfg.DELTA,
                                 sample_rate=256/50000, steps=975, accountant='rdp')
    report = {'runs': cfg.grid(), 'selected_layers': names, 'noise_multiplier': sigma, 'methods': {}}
    for method in cfg.METHODS:
        model.load_state_dict(initial)
        model.train()
        damping = None if method == 'dp_adamw' else 1e-3
        operator, builder = build_geometry(model, method, [x], cfg.SEED, 1, damping=damping)
        if operator is not None:
            assert set(operator.data) == set(names)
            assert operator.damping == damping
        if method == 'dp_kfc_a':
            assert operator.scale == 1.0
        clipper = Clipper(model, operator)
        assert 'head' in clipper.identity_geometry_layers
        assert len(clipper.parameters) == 50
        # Independent per-example autograd reference checks geometry and BK aggregation.
        refs = []
        for i in range(2):
            loss = torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1])
            gradients = torch.autograd.grad(loss, clipper.parameters)
            grads = dict(zip(clipper.parameters, gradients))
            for name, module in clipper.linear_modules.items():
                if operator is not None and name in operator.data:
                    grads[module.weight] = operator.transform_gradient(name, grads[module.weight])
            refs.append(grads)
        reference_norms = torch.stack([sum(g.square().sum() for g in ref.values()).sqrt() for ref in refs])
        loss, norms, factors, _, stats = clipper.aggregate_logical(x, y, 2)
        torch.testing.assert_close(norms, reference_norms, rtol=2e-3, atol=2e-4)
        expected_factors = (1/(reference_norms + 1e-6)).clamp(max=1)
        for p in clipper.parameters:
            expected = sum(ref[p] * f for ref, f in zip(refs, expected_factors))
            torch.testing.assert_close(p.grad, expected, rtol=3e-3, atol=3e-5)
        optimizer = torch.optim.AdamW(clipper.parameters, lr=cfg.LEARNING_RATE,
                                     weight_decay=cfg.WEIGHT_DECAY, betas=cfg.BETAS, eps=cfg.ADAM_EPS)
        clipper.step(optimizer, sigma, 2, torch.Generator(device=device).manual_seed(cfg.SEED+40000))
        metrics = parameter_metrics(model, initial)
        assert metrics['frozen_backbone_unchanged']
        assert metrics['trainable_lora_updated'] and metrics['head_updated']
        assert clipper.noise_events == clipper.optimizer_steps == 1
        assert stats['cache_empty_after_step']
        report['methods'][method] = {**metrics, **builder, 'bk_reference_matches': True}
        clipper.remove()
    (cfg.RESULTS / 'checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

"""Lightweight checks: exact small-model gradients, formulas, RNG, pretrained structure."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33 import config as cfg
from exp33.run import initialize, data_transform
from exp33.wiener import Wiener, build_wiener, matrix, private_step
from exp33.geometry import build_reference
from exp22.methods import Clipper
from exp22 import config as base
import torch
from torch import nn
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert list(cfg.GPU_METHODS) == [0, 1, 2, 3] and len(set(cfg.GPU_METHODS.values())) == 4
    for key in ('MODEL_NAME','EPOCHS','LEARNING_RATE','WEIGHT_DECAY','BETAS','ADAM_EPS','EPSILON','DELTA','MAX_GRAD_NORM','A_POWER','LOGICAL_BATCH_SIZE','PHYSICAL_BATCH_SIZE'):
        assert getattr(cfg, key) == getattr(base, key)
    assert cfg.SYNTHETIC_BATCHES == 10 and cfg.SYNTHETIC_BATCH_SIZE == 256 and cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE == 128
    sigma = get_noise_multiplier(target_epsilon=3., target_delta=1e-5, sample_rate=256/50000, steps=975, accountant='rdp')
    accountant = RDPAccountant()
    for _ in range(975):
        accountant.step(noise_multiplier=sigma, sample_rate=256/50000)
    assert accountant.get_epsilon(1e-5) <= 3.
    # Independent diagonal oracle, including exact zero signal.
    diagonal = {'layer': {'A': torch.diag(torch.tensor([0., 2., 4.], dtype=torch.float64)),
                          'G': torch.diag(torch.tensor([1., 3.], dtype=torch.float64))}}
    probe = torch.ones(2, 3)
    for full in (False, True):
        op = Wiener(diagonal, .5, full)
        signal = torch.tensor([0., 2., 4.])
        if full:
            signal = torch.tensor([1., 3.])[:, None] * signal[None, :] / 2.
        torch.testing.assert_close(op.transform('layer', probe), probe * signal / (signal + .5))
    model = nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.Tanh(), nn.Linear(6, 10))
    x = torch.randn(4, 4)
    rng = torch.Generator().manual_seed(42 + 20000 + 1)
    y = torch.randint(10, (4,), generator=rng)
    parameters = list(model.parameters())
    per_sample = []
    for xi, yi in zip(x, y):
        loss = nn.functional.cross_entropy(model(xi[None]), yi[None])
        per_sample.append(torch.autograd.grad(loss, parameters))
    norms = torch.stack([sum(g.square().sum() for g in grads).sqrt() for grads in per_sample])
    factors = (1 / (norms + 1e-6)).clamp(max=1)
    exact = [sum(f * grads[j] for f, grads in zip(factors, per_sample)) for j in range(len(parameters))]
    clipper = Clipper(model)
    _, actual_norms, _, _, _ = clipper.aggregate_logical(x, y, 2)
    torch.testing.assert_close(actual_norms, norms)
    for p, g in zip(parameters, exact):
        torch.testing.assert_close(p.grad, g, atol=2e-6, rtol=2e-5)
    matrices = {n: matrix(m).clone().double()/4 for n,m in model.named_modules() if isinstance(m, nn.Linear)}
    for full in (False, True):
        op, stats = build_wiener(model, 42, 1, sigma, full, batches=[x], physical_batch_size=2)
        expected = {n: {'A': z.T @ z / z.shape[0], **({'G': z @ z.T / z.shape[1]} if full else {})} for n,z in matrices.items()}
        reference = Wiener(expected, (sigma/256)**2, full)
        assert stats['builder_noise_events'] == 0
        for name, z in matrices.items():
            probe = torch.randn_like(z.float())
            filtered = op.transform(name, probe)
            torch.testing.assert_close(filtered, reference.transform(name, probe), atol=1e-5, rtol=1e-4)
            assert filtered.norm() <= probe.norm() + 1e-5
        clipper.aggregate_logical(x, y, 2)
        noisy = []
        gen = torch.Generator().manual_seed(40042)
        for p in parameters:
            noise = torch.randn(p.numel(), generator=gen).reshape_as(p)
            noisy.append((p.grad + sigma*noise)/4)
        class Capture:
            def step(self):
                self.grads = [p.grad.clone() for p in parameters]
        capture = Capture()
        private_step(clipper, capture, sigma, 4, torch.Generator().manual_seed(40042), op)
        positions = {id(p): i for i,p in enumerate(parameters)}
        for n,m in model.named_modules():
            if isinstance(m, nn.Linear):
                z = torch.cat((noisy[positions[id(m.weight)]], noisy[positions[id(m.bias)]][:,None]), 1)
                torch.testing.assert_close(matrix(m), op.transform(n,z))
            elif isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    torch.testing.assert_close(p.grad, noisy[positions[id(p)]])
    print('PASS: grid, protocol, accounting, exact global clipping, synthetic covariance, post-noise filtering, identity parameters, gains and contraction', flush=True)
    vit = initialize(42, 'cuda:0')
    assert sum(isinstance(m,nn.Linear) for m in vit.modules()) == 74
    assert all(p.requires_grad for p in vit.parameters())
    from exp33.geometry import synthetic_stream
    probes = synthetic_stream(42, 1, 'cuda:0', batches=1, batch_size=2)
    op, stats = build_wiener(vit, 42, 1, sigma, True, batches=probes, physical_batch_size=1)
    assert len(op.data) == 74
    for name, (ua, ug, gain) in op.data.items():
        z = torch.randn(ug.shape[0], ua.shape[0], device='cuda:0')
        assert op.transform(name, z).norm() <= z.norm() * (1+2e-5) + 1e-8
    from exp22.geometry import build_a_operator
    from exp33.geometry import remove_a_scale
    ref, stats = build_a_operator(vit, synthetic_stream(42,1,'cuda:0',batches=1,batch_size=2), power=.4,damping=.1)
    remove_a_scale(ref, stats)
    assert ref.scale == 1. and stats['scale_match'] == 1.
    print('PASS: pretrained Exp22/30 initialization, 74 trainable Linear layers, actual ViT synthetic backward, all-layer contraction, no-scale KFC reference', flush=True)
    print(f'noise_multiplier={sigma}; epsilon={accountant.get_epsilon(1e-5)}; 195 steps/epoch, 975 total', flush=True)

if __name__ == '__main__':
    main()

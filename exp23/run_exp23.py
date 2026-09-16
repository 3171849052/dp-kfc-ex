"""One (method, seed) per process; training primitives come from Exp19/20."""
import argparse
import hashlib
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from dp_kfac.models import SimpleCNN
from exp12.runtime import runtime
from exp13.run_exp13 import evaluate
from exp19.methods import GhostNorm, ghost_aggregate, noise_and_step
from exp20.profiling import timestamp, phase_start, phase_end, EventProfiler
from exp23 import HERE
from exp23 import config as cfg
from exp23.public_data import private_data, fixed_batches, calibration
from exp23.geometry import build, alignment


def initialize(seed, device):
    torch.random.default_generator.manual_seed(seed)
    return SimpleCNN().to(device)


def private_loader(data, seed):
    return DataLoader(data, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def noise_generator(seed, device):
    return torch.Generator(device=device).manual_seed(seed+40000)


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(method, seed, smoke):
    device = torch.device('cuda:0')
    train, test = private_data()
    # Keep the same full-data accounting schedule in smoke; execute only two steps.
    sigma = get_noise_multiplier(target_epsilon=cfg.EPSILON, target_delta=cfg.DELTA,
        sample_rate=cfg.BATCH_SIZE/len(train), steps=cfg.EPOCHS*(len(train)//cfg.BATCH_SIZE), accountant='rdp')
    sample_rate = cfg.BATCH_SIZE/len(train)
    oracle = fixed_batches(train, cfg.ORACLE_SEED, device, cfg.ORACLE_BATCHES)
    if smoke:
        train, test = Subset(train, range(512)), Subset(test, range(512))
    epochs = 1 if smoke else cfg.EPOCHS
    model = initialize(seed, device)
    initialization_hash = digest(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0, weight_decay=0)
    loader = private_loader(train, seed)
    test_loader = DataLoader(test, batch_size=256, generator=torch.Generator().manual_seed(seed))
    noise_rng = noise_generator(seed, device)
    accountant = RDPAccountant()
    output = HERE/'results'/('smoke' if smoke else 'formal')
    directory = output/'runs'/f'{method}_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    rows, geometry_rows = [], []
    for epoch in range(1, epochs+1):
        model.train()
        model.zero_grad(set_to_none=True)
        phase = phase_start(device)
        start = timestamp(device)
        batches = [] if method == 'dp_sgd' else calibration(cfg.source(method), seed, epoch, device)
        operator, factors, stats = build(model, method, batches)
        stats['builder_seconds'] = timestamp(device)-start
        stats.update(phase_end(device, 'builder', phase))
        del batches
        start = timestamp(device)
        if method != 'dp_sgd':
            for row in alignment(model, factors, oracle, method.startswith('full_')):
                geometry_rows.append(dict(method=method, source=cfg.source(method), seed=seed, epoch=epoch, **row))
        stats['oracle_seconds'] = timestamp(device)-start
        del factors
        # Oracle is read-only: neither parameter gradients nor random draws enter training.
        assert all(p.grad is None for p in model.parameters())
        phase = phase_start(device)
        start = timestamp(device)
        profiler = EventProfiler()
        hooks = GhostNorm(model, operator)
        norms_all, clips_all = [], []
        batch_hashes = []
        for x, y in loader:
            if smoke:
                batch_hashes.append(digest((x, y)))
            x, y = x.to(device), y.to(device)
            loss, norms, clips, _, _ = ghost_aggregate(model, hooks, x, y, profiler)
            noise_and_step(model, optimizer, sigma, len(x), noise_rng)
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
            norms_all.append(norms.detach())
            clips_all.append(clips.detach())
        hooks.remove()
        model.zero_grad(set_to_none=True)
        stats['private_train_seconds'] = timestamp(device)-start
        stats.update(profiler.resolve())
        stats.update(phase_end(device, 'train', phase))
        stats['peak_allocated_cuda_bytes'] = max(stats['builder_peak_cuda_allocated_bytes'], stats['train_peak_cuda_allocated_bytes'])
        norm, clip = torch.cat(norms_all), torch.cat(clips_all)
        assert torch.isfinite(norm).all()
        test_loss, accuracy = evaluate(model, test_loader, device)
        stats.update(clip_fraction=(clip < 1).float().mean().item(), transformed_norm_mean=norm.mean().item(),
                     transformed_norm_p90=norm.quantile(.9).item(), transformed_norm_p99=norm.quantile(.99).item())
        rows.append(dict(method=method, source=cfg.source(method), seed=seed, epoch=epoch,
            test_accuracy=accuracy, test_loss=test_loss, noise_multiplier=sigma,
            epsilon=accountant.get_epsilon(cfg.DELTA), accountant_steps=sum(x[2] for x in accountant.history),
            batches=len(loader), diagnostic_scope=cfg.RESEARCH_ONLY, **stats))
        pd.DataFrame(rows).to_csv(directory/'metrics.csv', index=False)
        if geometry_rows:
            pd.DataFrame(geometry_rows).to_csv(directory/'geometry.csv', index=False)
        print(f'{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f}', flush=True)
        del operator, hooks, norm, clip, norms_all, clips_all
    frame = pd.DataFrame(rows)
    summary = dict(method=method, source=cfg.source(method), seed=seed, smoke=smoke,
        final_accuracy=float(frame.test_accuracy.iloc[-1]), best_accuracy=float(frame.test_accuracy.max()),
        accuracy_auc=float(np.trapezoid(frame.test_accuracy, frame.epoch)),
        builder_seconds=float(frame.builder_seconds.sum()), private_train_seconds=float(frame.private_train_seconds.sum()),
        peak_allocated_cuda_bytes=int(frame.peak_allocated_cuda_bytes.max()),
        **{key: float(frame[key].mean()) for key in ('clip_fraction', 'transformed_norm_mean', 'transformed_norm_p90', 'transformed_norm_p99')})
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    configuration = dict(method=method, seed=seed, smoke=smoke, epochs=epochs, calibration_samples=2560,
        a_power=cfg.A_POWER, damping=cfg.DAMPING, clip=cfg.CLIP, logical_batch=256, epsilon_target=cfg.EPSILON,
        delta=cfg.DELTA, learning_rate=cfg.LEARNING_RATE, noise_multiplier=sigma,
        accounting='Exp19/20 shuffled fixed-batch RDP convention; smoke uses formal sigma and sample rate',
        oracle='Fixed 10x256 MNIST samples, seed 23000, before private training each epoch; read-only',
        diagnostic_scope=cfg.RESEARCH_ONLY, initialization_hash=initialization_hash,
        private_batch_hashes=batch_hashes, final_noise_rng_hash=digest((noise_rng.get_state(),)),
        rng=dict(initialization='seed (CPU only)', private_shuffle='seed', dp_noise='seed+40000',
                 calibration='seed+10000+epoch', pink_labels='seed+20000+epoch'),
        timing='CUDA synchronized phase boundaries; no disposable warmup; oracle/evaluation excluded from algorithm timings')
    (directory/'config.json').write_text(json.dumps(configuration, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=cfg.METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=cfg.SEEDS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with runtime('cuda:0'):
        run(args.method, args.seed, args.smoke)


if __name__ == '__main__':
    main()

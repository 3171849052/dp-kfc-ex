"""Synthetic-KFAC RMS matched MNIST training; --smoke is one zero-noise batch for each beta."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import argparse
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from exp12.runtime import runtime
from exp14b import config as cfg
from exp14b.builders import synthetic_cache, build_from_cache
from exp12.state_metrics import synthetic_state_metrics
from exp13.ghost import GhostNorm, ghost_aggregate, noise_and_step
from exp13.run_exp13 import initialize, timestamp, evaluate
from exp14b.analyze import save
from exp14b.geometry import diagnose

OUTPUT = Path(__file__).parent / 'results'


def disposable_warmup(train, device):
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    with torch.random.fork_rng():
        model = initialize(90000, device)
        cache = synthetic_cache(90000, 0, device, 1, cfg.SYNTHETIC_BATCH_SIZE)
        for beta in cfg.BETAS:
            operator, _, _ = build_from_cache(model, beta, cache, 90000, 0)
            del operator
        # Also warm the diagnostic forward, without recording its values.
        synthetic_state_metrics(model, cache)
        loader = DataLoader(train, batch_size=cfg.BATCH_SIZE,
                            generator=torch.Generator().manual_seed(90000))
        x, y = next(iter(loader))
        operator, _, _ = build_from_cache(model, cfg.BETAS[0], cache, 90000, 0)
        hooks = GhostNorm(model, operator)
        # ghost_aggregate already transforms the aggregate exactly once.
        ghost_aggregate(model, hooks, x.to(device), y.to(device), cfg.MAX_GRAD_NORM)
        hooks.remove()
        torch.cuda.synchronize(device)
        del model, operator, hooks, cache, loader, x, y
    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_states))


def save_config(output, sigma, epochs, seeds, train_size, smoke):
    output.mkdir(parents=True, exist_ok=True)
    config = dict(dataset='MNIST', model='dp_kfac.models.SimpleCNN', betas=cfg.BETAS, estimator='KFAC-U',
        seeds=seeds, epochs=epochs, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
        optimizer='SGD', learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY, epsilon=cfg.EPSILON if not smoke else None,
        delta=cfg.DELTA, max_grad_norm=cfg.MAX_GRAD_NORM, damping=cfg.DAMPING,
        noise_multiplier=sigma, sample_rate=cfg.BATCH_SIZE/train_size,
        accountant_steps=epochs*(train_size//cfg.BATCH_SIZE) if not smoke else 0,
        accounting='RDP; exp11 shuffled fixed-batch convention',
        train_samples=train_size, synthetic_batches=1 if smoke else cfg.SYNTHETIC_BATCHES,
        synthetic_batch_size=cfg.SYNTHETIC_BATCH_SIZE, rebuild='once per epoch',
        synthetic_seed='seed + 10000 + epoch', label_seed='seed + 20000 + epoch',
        noise_seed='seed + 40000', shuffle_seed='seed', smoke=smoke,
        cudnn_benchmark=False, cudnn_deterministic=True, matmul_tf32=False, cudnn_tf32=False,
        scale_matching='synthetic KFAC global RMS', scale_reference_beta=.5,
        scale_match='sqrt(m_0.5 / m_beta) on current run/epoch factors',
        disposable_warmup=True, cost_metric='algorithm_epoch_seconds',
        preconditioner_build_seconds='cache + curvature + factor power + scale matching',
        algorithm_epoch_seconds='preconditioner_build_seconds + private_train_seconds',
        wall_epoch_seconds='algorithm_epoch_seconds + geometry_seconds + state_metric_seconds + evaluation_seconds')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def run(beta, seed, train, test, sigma, epochs, device, smoke=False):
    model = initialize(seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
                               momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    loader = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=cfg.BATCH_SIZE,
                             generator=torch.Generator().manual_seed(seed))
    noise_rng = torch.Generator(device=device).manual_seed(seed + 40000)
    accountant = RDPAccountant()
    rows, geometry = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        start = timestamp(device)
        cache = synthetic_cache(seed, epoch, device,
                                1 if smoke else cfg.SYNTHETIC_BATCHES, cfg.SYNTHETIC_BATCH_SIZE)
        operator, stats, curvature = build_from_cache(model, beta, cache, seed, epoch)
        build_seconds = timestamp(device) - start
        build_peak = torch.cuda.max_memory_allocated(device)
        geometry_start = timestamp(device)
        geometry.extend(diagnose(curvature, beta, seed, epoch))
        del curvature
        geometry_seconds = timestamp(device) - geometry_start
        state_start = timestamp(device)
        stats.update(synthetic_state_metrics(model, cache))
        stats['state_metric_forward_calls'] = len(cache)
        state_seconds = timestamp(device) - state_start
        del cache
        torch.cuda.reset_peak_memory_stats(device)
        hooks = GhostNorm(model, operator)
        loss_sum = clipped = factor_sum = 0.
        count = 0
        epoch_norms = []
        initial = [p.detach().clone() for p in model.parameters()] if smoke else None
        train_start = timestamp(device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, norms, factors = ghost_aggregate(model, hooks, x, y, cfg.MAX_GRAD_NORM)
            if smoke:
                assert torch.isfinite(loss) and torch.isfinite(norms).all()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            noise_and_step(model, optimizer, sigma, len(x), noise_rng, cfg.MAX_GRAD_NORM)
            if sigma:
                accountant.step(noise_multiplier=sigma, sample_rate=cfg.BATCH_SIZE / len(train))
            loss_sum += loss.item()
            clipped += (norms > cfg.MAX_GRAD_NORM).sum().item()
            factor_sum += factors.sum().item()
            count += len(x)
            epoch_norms.append(norms.detach())
        train_seconds = timestamp(device) - train_start
        train_peak = torch.cuda.max_memory_allocated(device)
        hooks.remove()
        if smoke:
            assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), initial))
            assert all(torch.isfinite(p).all() for p in model.parameters())
        evaluation_start = timestamp(device)
        test_loss, accuracy = evaluate(model, test_loader, device)
        evaluation_seconds = timestamp(device) - evaluation_start
        quantiles = torch.cat(epoch_norms).quantile(torch.tensor([.5, .9, .99], device=device)).tolist()
        row = dict(beta=beta, seed=seed, epoch=epoch, test_accuracy=accuracy,
            test_loss=test_loss, train_loss=loss_sum/count,
            epsilon=accountant.get_epsilon(delta=cfg.DELTA) if sigma else None,
            noise_multiplier=sigma, clip_fraction=clipped/count, mean_clip_factor=factor_sum/count,
            transformed_norm_p50=quantiles[0], transformed_norm_p90=quantiles[1], transformed_norm_p99=quantiles[2],
            preconditioner_build_seconds=build_seconds, private_train_seconds=train_seconds,
            geometry_seconds=geometry_seconds, state_metric_seconds=state_seconds, evaluation_seconds=evaluation_seconds,
            algorithm_epoch_seconds=build_seconds + train_seconds,
            wall_epoch_seconds=build_seconds + geometry_seconds + state_seconds + train_seconds + evaluation_seconds,
            seconds_per_batch=train_seconds/len(loader),
            samples_per_second=count/train_seconds, build_peak_cuda_allocated_bytes=build_peak,
            train_peak_cuda_allocated_bytes=train_peak, total_peak_cuda_allocated_bytes=max(build_peak, train_peak),
            batches=len(loader), samples=count, accountant_steps=epoch*len(loader) if sigma else 0,
            **stats)
        rows.append(row)
        print(f'{beta} seed={seed} epoch={epoch} accuracy={accuracy:.4f} '
              f'build={build_seconds:.2f}s train={train_seconds:.2f}s '
              f'VJP={stats["builder_vjp_calls"]} vectors={stats["builder_reverse_vectors"]}', flush=True)
        del operator, hooks, initial, epoch_norms
    return rows, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train = datasets.MNIST(ROOT / 'exp1/data', train=True, download=False, transform=transform)
    test = datasets.MNIST(ROOT / 'exp1/data', train=False, download=False, transform=transform)
    epochs, seeds = cfg.EPOCHS, cfg.SEEDS
    if args.smoke:
        train, test = Subset(train, range(cfg.BATCH_SIZE)), Subset(test, range(cfg.BATCH_SIZE))
        epochs, seeds = 1, (42,)
    # Exactly one calibration shared by all betas and both seeds.
    sigma = 0. if args.smoke else get_noise_multiplier(target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA, sample_rate=cfg.BATCH_SIZE/len(train),
        steps=epochs*(len(train)//cfg.BATCH_SIZE), accountant='rdp')
    output = OUTPUT / 'smoke' if args.smoke else OUTPUT
    rows, geometry = [], []
    with runtime(device):  # Deterministic settings and CUDA/cuBLAS warmup before timing.
        disposable_warmup(train, device)
        save_config(output, sigma, epochs, seeds, len(train), args.smoke)
        for seed in seeds:
            for beta in cfg.BETAS:
                run_rows, run_geometry = run(beta, seed, train, test, sigma, epochs, device, args.smoke)
                rows.extend(run_rows)
                geometry.extend(run_geometry)
                save(rows, geometry, output)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

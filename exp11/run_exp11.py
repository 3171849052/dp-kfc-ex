"""MNIST: six methods x two seeds; --smoke runs one batch per method."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from exp10 import run_exp10 as base
from exp10b import config as cfg
from exp11.operators import build
from exp11.ghost import GhostNorm, ghost_aggregate, exact_aggregate, noise_and_step

torch = base.torch
OUTPUT = Path(__file__).parent / 'results'
KINDS = ('DP-SGD', 'Factorized Equil', 'DP-KFC')
METHODS = tuple((kind, mode) for kind in KINDS for mode in ('Exact', 'Ghost'))


def run(kind, mode, seed, train, test, sigma, epochs, device):
    torch.manual_seed(seed)
    model = base.SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    loader = base.DataLoader(train, batch_size=256, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    test_loader = base.DataLoader(test, batch_size=256,
        generator=torch.Generator().manual_seed(seed))
    accountant = base.RDPAccountant()
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        # Reset allocator between epochs/methods so reserved-memory comparisons
        # do not inherit another method's cache. Never empty inside training.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = base.timestamp(device)
        operator = build(model, kind, seed, epoch, device)
        build_seconds = base.timestamp(device) - start
        build_allocated = torch.cuda.max_memory_allocated(device)
        build_reserved = torch.cuda.max_memory_reserved(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        wrapped = base.GradSampleModule(model, loss_reduction='sum') if mode == 'Exact' else model
        hooks = GhostNorm(model, operator) if mode == 'Ghost' else None
        loss_sum = clipped = factor_sum = 0.
        count = 0
        timing = dict(ghost_first_pass_norm_seconds=0.,
                      ghost_second_pass_backward_seconds=0., aggregate_transform_seconds=0.)
        start = base.timestamp(device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if mode == 'Ghost':
                loss, norms, factors, times = ghost_aggregate(model, hooks, x, y, cfg.MAX_GRAD_NORM)
            else:
                loss, norms, factors, times = exact_aggregate(wrapped, operator, x, y, cfg.MAX_GRAD_NORM)
            noise_and_step(wrapped, optimizer, sigma, 256, cfg.MAX_GRAD_NORM)
            accountant.step(noise_multiplier=sigma, sample_rate=256 / len(train))
            loss_sum += loss.item()
            clipped += (norms > cfg.MAX_GRAD_NORM).sum().item()
            factor_sum += factors.sum().item()
            count += len(x)
            for key in timing:
                timing[key] += times[key]
        train_seconds = base.timestamp(device) - start
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        if hooks is not None:
            hooks.remove()
        if mode == 'Exact':
            wrapped.to_standard_module()
        test_loss, accuracy = base.evaluate(model, test_loader, device)
        rows.append(dict(method=f'{kind} {mode}', algorithm=kind, mode=mode, seed=seed,
            epoch=epoch, test_accuracy=accuracy, test_loss=test_loss, train_loss=loss_sum/count,
            epsilon=accountant.get_epsilon(delta=cfg.DELTA) if sigma else float('inf'),
            noise_multiplier=sigma, clip_fraction=clipped/count, mean_clip_factor=factor_sum/count,
            preconditioner_build_seconds=build_seconds, private_train_seconds=train_seconds,
            seconds_per_batch=train_seconds/len(loader), samples_per_second=count/train_seconds,
            peak_cuda_allocated_bytes=allocated, peak_cuda_reserved_bytes=reserved,
            build_peak_cuda_allocated_bytes=build_allocated, build_peak_cuda_reserved_bytes=build_reserved,
            total_peak_cuda_allocated_bytes=max(allocated, build_allocated),
            total_peak_cuda_reserved_bytes=max(reserved, build_reserved),
            batches=len(loader), samples=count, accountant_steps=epoch*len(loader), **timing))
        print(f'{kind} {mode} seed={seed} epoch={epoch} accuracy={accuracy:.4f} '
              f'build={build_seconds:.2f}s train={train_seconds:.2f}s', flush=True)
        # Do not retain the previous epoch's operator during the next build.
        del operator, hooks, wrapped
    return rows


def save(rows, output):
    frame = base.pd.DataFrame(rows)
    frame.to_csv(output / 'metrics.csv', index=False)
    final = frame.groupby(['method', 'seed'], sort=False).tail(1)
    final.to_csv(output / 'summary.csv', index=False)
    pairs = frame.pivot(index=['algorithm', 'seed', 'epoch'], columns='mode', values='test_accuracy')
    pairs['ghost_minus_exact'] = pairs['Ghost'] - pairs['Exact'] if 'Ghost' in pairs else float('nan')
    pairs.to_csv(output / 'paired_accuracy.csv')
    comparison = frame[(frame['mode'] == 'Ghost') & (frame.algorithm != 'DP-SGD')]
    comparison.to_csv(output / 'ghost_comparison.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='256 MNIST train/test, one epoch, seed 42, sigma=0.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    output = OUTPUT / 'smoke' if args.smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    transform = base.transforms.Compose([base.transforms.ToTensor(),
        base.transforms.Normalize((.1307,), (.3081,))])
    train = base.datasets.MNIST(base.ROOT / 'exp1/data', train=True, download=False, transform=transform)
    test = base.datasets.MNIST(base.ROOT / 'exp1/data', train=False, download=False, transform=transform)
    epochs, seeds = cfg.EPOCHS, cfg.SEEDS
    if args.smoke:
        train, test = base.Subset(train, range(256)), base.Subset(test, range(256))
        epochs, seeds = 1, (42,)
    sigma = 0. if args.smoke else base.get_noise_multiplier(target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA, sample_rate=256/len(train),
        steps=epochs*(len(train)//256), accountant='rdp')
    config = dict(dataset='MNIST', model='SimpleCNN', methods=[f'{k} {m}' for k, m in METHODS],
        epochs=epochs, seeds=seeds, batch_size=256, physical_batch_size=256, drop_last=True,
        optimizer='SGD', learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY, epsilon=cfg.EPSILON, delta=cfg.DELTA,
        max_grad_norm=cfg.MAX_GRAD_NORM, noise_multiplier=sigma, tau=cfg.TAU,
        probes=cfg.PROBES, synthetic_batch_size=cfg.SYNTHETIC_BATCH_SIZE,
        preconditioner_batches=cfg.PRECONDITIONER_BATCHES, hard_clamp=False,
        kfc_damping=.001, accounting='RDP; same shuffled fixed-batch convention as exp10b',
        train_samples=len(train), batches_per_epoch=len(train)//256)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = []
    for seed in seeds:
        for kind, mode in METHODS:
            rows.extend(run(kind, mode, seed, train, test, sigma, epochs, device))
            save(rows, output)


if __name__ == '__main__':
    main()

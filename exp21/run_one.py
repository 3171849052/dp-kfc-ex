"""One fresh CUDA process per method/seed; only --smoke runs a short protocol."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
for key, value in dict(XDG_CACHE_HOME='.cache', MPLCONFIGDIR='.cache/matplotlib', CUDA_CACHE_PATH='.cache/cuda', TMPDIR='.cache/tmp').items():
    path = HERE/value
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from exp12.runtime import runtime
from exp13.run_exp13 import evaluate
from dp_kfac.models import SimpleCNN
from exp21 import config as cfg
from exp21.methods import Clipper, synthetic_cache, build_from_cache, noise_and_step
from exp21.profiling import timestamp, timed, phase_start, phase_end, EventProfiler, PHASES


def initialize(seed, device):
    torch.random.default_generator.manual_seed(seed)
    return SimpleCNN().to(device)


def load_data():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    return tuple(datasets.MNIST(ROOT/'exp1/data', train=train, download=False, transform=t) for train in (True, False))


def private_loader(data, seed):
    return DataLoader(data, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def warmup(method, train, device):
    with torch.random.fork_rng(devices=[device.index]):
        net = initialize(90021, device)
        cache = synthetic_cache(90021, 1, device, 1, 256)
        op, _ = build_from_cache(net, cfg.POWER, cache, 90021, 1)
        x, y = next(iter(private_loader(Subset(train, range(256)), 90021)))
        clipper = Clipper(net, op, method, max_grad_norm=cfg.MAX_GRAD_NORM)
        clipper.aggregate(x.to(device), y.to(device))
        clipper.remove()
        torch.cuda.synchronize(device)


def run(method, seed, smoke, output):
    device = torch.device('cuda:0')
    train, test = load_data()
    if smoke:
        train, test = Subset(train, range(768)), Subset(test, range(256))
    epochs = 1 if smoke else cfg.EPOCHS
    steps = epochs*(len(train)//cfg.BATCH_SIZE)
    sigma = 0. if smoke else get_noise_multiplier(target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA, sample_rate=256/len(train), steps=steps, accountant='rdp')
    warmup(method, train, device)
    model = initialize(seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE)
    loader = private_loader(train, seed)
    test_loader = DataLoader(test, batch_size=256, generator=torch.Generator().manual_seed(seed))
    noise_rng = torch.Generator(device=device).manual_seed(seed+40000)
    accountant = RDPAccountant()
    run_dir = output/'runs'/f'{method}_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in range(1, epochs+1):
        model.train()
        model.zero_grad(set_to_none=True)
        build_start = phase_start(device)
        start = timestamp(device)
        cache = synthetic_cache(seed, epoch, device, 1 if smoke else cfg.SYNTHETIC_BATCHES, 256)
        op, builder = build_from_cache(model, cfg.POWER, cache, seed, epoch)
        del cache
        build_seconds = timestamp(device)-start
        build_memory = phase_end(device, 'build', build_start)
        clipper = Clipper(model, op, method, max_grad_norm=cfg.MAX_GRAD_NORM)
        prof = EventProfiler()
        stats = {k: 0. for k in PHASES}
        peak_keys = ('bk_cache_bytes', 'temporary_per_sample_grad_bytes')
        stats.update({k: 0 for k in peak_keys})
        losses, norms_all, factors_all, allocations = [], [], [], []
        train_start = phase_start(device)
        start = timestamp(device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, norms, factors, _, phases = clipper.aggregate(x, y, prof)
            with timed(prof, 'noise_step_seconds', device):
                clipper.step(optimizer, sigma, len(x), noise_rng)
            if sigma:
                accountant.step(noise_multiplier=sigma, sample_rate=256/len(train))
            losses.append(loss.detach())
            norms_all.append(norms.detach())
            factors_all.append(factors.detach())
            for k in peak_keys:
                stats[k] = max(stats[k], phases[k])
            assert clipper.hooks is None or not (clipper.hooks.records or clipper.hooks.pending)
            allocations.append(torch.cuda.memory_allocated(device))
            del x, y, loss, norms, factors
        elapsed = timestamp(device)-start
        stats.update(prof.resolve())
        memory = phase_end(device, 'train', train_start)
        clipper.remove()
        model.zero_grad(set_to_none=True)
        stats.update({k: phases[k] for k in ('ghost_layer_count', 'fast_layer_count',
            'fallback_layer_count', 'fallback_layers', 'requires_second_backward', 'backward_calls',
            'first_pass_parameter_grad_count', 'gd_applied', 'input_gradient_computed')})
        stats['gd_anchor_module'] = phases['gd_anchor_module']
        for k in ('layer_strategies', 'bk_ghost_layers', 'bk_fast_layers', 'fallback_layer_names',
                  'fallback_parameter_names', 'preconditioned_layers', 'identity_geometry_layers', 'gd_anchor_modules'):
            stats[k] = json.dumps(phases[k])
        stats.update(cache_empty_after_step=True, batch_end_allocated_bytes=json.dumps(allocations))
        if smoke:
            torch.save(dict(model={n: p.detach().cpu() for n, p in model.state_dict().items()},
                norms=torch.cat(norms_all).cpu(), factors=torch.cat(factors_all).cpu()), run_dir/'smoke_state.pt')
        test_loss, accuracy = evaluate(model, test_loader, device)
        rows.append(dict(method=method, seed=seed, epoch=epoch, p=cfg.POWER,
            train_loss=torch.stack(losses).sum().item()/(len(loader)*256), test_accuracy=accuracy,
            test_loss=test_loss, noise_multiplier=sigma,
            epsilon=accountant.get_epsilon(delta=cfg.DELTA) if sigma else 0.,
            accountant_steps=sum(v[2] for v in accountant.history),
            batches=len(loader), samples=len(loader)*256, private_train_seconds=elapsed,
            seconds_per_batch=elapsed/len(loader), private_samples_per_second=len(loader)*256/elapsed,
            preconditioner_build_seconds=build_seconds,
            peak_cuda_allocated_bytes=memory['train_peak_cuda_allocated_bytes'],
            peak_cuda_reserved_bytes=memory['train_peak_cuda_reserved_bytes'],
            **stats, **memory, **build_memory, **builder))
        pd.DataFrame(rows).to_csv(run_dir/'metrics.csv', index=False)
        print(f'{method} seed={seed} epoch={epoch}: {elapsed/len(loader):.6f} s/batch, {accuracy:.4f} accuracy', flush=True)
        del clipper, op, losses, norms_all, factors_all
    configuration = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    configuration.update(smoke=smoke, method=method, seed=seed, epochs=epochs, total_steps=steps,
        noise_multiplier=sigma, synthetic_batches=1 if smoke else cfg.SYNTHETIC_BATCHES,
        accounting='Exp20 RDP shuffled fixed-batch convention; not a new privacy proof',
        rng=dict(initialization='seed', shuffle='seed', noise='seed+40000', synthetic='seed+10000+epoch'))
    (run_dir/'config.json').write_text(json.dumps(configuration, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=cfg.METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=cfg.SEEDS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with runtime('cuda:0'):
        run(args.method, args.seed, args.smoke, HERE/'results'/'smoke' if args.smoke else HERE/'results')


if __name__ == '__main__':
    main()

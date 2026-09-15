"""One method/seed per fresh CUDA subprocess."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
for key, value in dict(XDG_CACHE_HOME='.cache', MPLCONFIGDIR='.cache/matplotlib', CUDA_CACHE_PATH='.cache/cuda').items():
    os.environ[key] = str(HERE/value)
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from exp12.runtime import runtime
from exp13.run_exp13 import initialize, evaluate
from exp19 import config as cfg
from exp19.methods import (synthetic_cache, build_from_cache, GhostNorm,
    GradSampleModule, exact_aggregate, ghost_aggregate, noise_and_step, noise_and_normalize)
from exp19.profiling import timestamp, timed, phase_start, phase_end, EventProfiler


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    return tuple(datasets.MNIST(ROOT/'exp1/data', train=train, download=False, transform=transform) for train in (True, False))


def private_loader(data, seed):
    return DataLoader(data, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def disposable_warmup(method, train, device):
    """Exercise the actual 256-sample action path without optimizer/accountant."""
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        model = initialize(90019, device)
        cache = synthetic_cache(90019, 1, device, 1, 256)
        operator, _ = build_from_cache(model, method, cache, 90019, 1)
        loader = private_loader(Subset(train, range(256)), 90019)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        if method == cfg.METHODS[0]:
            wrapper = GradSampleModule(model, loss_reduction='sum')
            exact_aggregate(wrapper, operator, x, y)
            wrapper.to_standard_module()
            del wrapper
        else:
            hooks = GhostNorm(model, operator)
            ghost_aggregate(model, hooks, x, y)
            hooks.remove()
            del hooks
        generator = torch.Generator(device=device).manual_seed(90019)
        noise_and_normalize(model, 0., 256, generator)
        model.zero_grad(set_to_none=True)
        del model, operator, cache, loader, x, y, generator
        torch.cuda.synchronize(device)
    assert torch.equal(cpu, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))



def run(method, seed, smoke, output):
    device = torch.device('cuda:0')
    train, test = load_data()
    if smoke:
        train, test = Subset(train, range(256)), Subset(test, range(256))
    epochs = 1 if smoke else cfg.EPOCHS
    steps = epochs*(len(train)//cfg.BATCH_SIZE)
    assert smoke or steps == 1170
    sigma = 0. if smoke else get_noise_multiplier(target_epsilon=cfg.EPSILON, target_delta=cfg.DELTA,
        sample_rate=cfg.BATCH_SIZE/len(train), steps=steps, accountant='rdp')
    disposable_warmup(method, train, device)
    model = initialize(seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0, weight_decay=0)
    loader = private_loader(train, seed)
    test_loader = DataLoader(test, batch_size=256, generator=torch.Generator().manual_seed(seed))
    noise_rng = torch.Generator(device=device).manual_seed(seed+40000)
    accountant = RDPAccountant()
    rows, layer_rows = [], []
    run_dir = output/'runs'/f'{method}_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs+1):
        wall_start = timestamp(device)
        model.train()
        model.zero_grad(set_to_none=True)
        build_start = phase_start(device)
        stats = {name: 0. for name in ('exact_forward_backward_seconds',
            'per_sample_precondition_clip_seconds', 'ghost_first_pass_seconds',
            'ghost_second_pass_seconds', 'aggregate_transform_seconds', 'noise_step_seconds')}
        build_profiler = EventProfiler()
        start = timestamp(device)
        with timed(build_profiler, 'synthetic_generation_seconds', device):
            cache = synthetic_cache(seed, epoch, device, 1 if smoke else 10, 256)
        operator, builder_stats = build_from_cache(model, method, cache, seed, epoch, build_profiler)
        stats.update(builder_stats)
        del cache
        stats['preconditioner_build_seconds'] = timestamp(device)-start
        stats.update(build_profiler.resolve())
        del build_profiler
        stats.update(phase_end(device, 'build', build_start))
        train_profiler = EventProfiler()
        train_start = phase_start(device)
        start = timestamp(device)
        exact = method == cfg.METHODS[0]
        wrapper = GradSampleModule(model, loss_reduction='sum') if exact else None
        hooks = None if exact else GhostNorm(model, operator)
        losses_gpu = []
        norms_gpu, factors_gpu = [], []
        shares = {n: [] for n in operator.data}
        stats['grad_sample_bytes'] = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, norms, factors, layer_sq, phases = (exact_aggregate(wrapper, operator, x, y, train_profiler) if exact
                else ghost_aggregate(model, hooks, x, y, train_profiler))
            with timed(train_profiler, 'noise_step_seconds', device):
                noise_and_step(model, optimizer, sigma, len(x), noise_rng)
            if sigma:
                accountant.step(noise_multiplier=sigma, sample_rate=256/len(train))
            for k, v in phases.items():
                stats[k] = max(stats[k], v) if k == 'grad_sample_bytes' else stats.get(k, 0.)+v
            losses_gpu.append(loss.detach())
            norms_gpu.append(norms.detach())
            factors_gpu.append(factors.detach())
            for n, sq in layer_sq.items():
                # Zero total gradient has zero contribution for all layers.
                shares[n].append(torch.where(norms > 0, sq/norms.square(), 0.).detach())
            del x, y, loss, norms, factors, layer_sq, sq
        if exact:
            wrapper.to_standard_module()
            del wrapper
        else:
            hooks.remove()
            del hooks
        model.zero_grad(set_to_none=True)
        stats['private_train_seconds'] = timestamp(device)-start
        stats.update(train_profiler.resolve())
        del train_profiler
        stats.update(phase_end(device, 'train', train_start))
        del operator
        diagnostic_start = timestamp(device)
        norm = torch.cat(norms_gpu).cpu()
        clip = torch.cat(factors_gpu).cpu()
        loss_sum = torch.stack(losses_gpu).sum().cpu().item()
        assert torch.isfinite(norm).all()
        del norms_gpu, factors_gpu, losses_gpu
        q = norm.quantile(torch.tensor([.5, .9, .99])).tolist()
        for n, values in shares.items():
            v = torch.cat(values).cpu()
            quantiles = v.quantile(torch.tensor([.5, .9, .99])).tolist()
            layer_rows.append(dict(method=method, seed=seed, epoch=epoch, layer=n,
                mean=v.mean().item(), median=quantiles[0], p90=quantiles[1], p99=quantiles[2]))
        del shares, values
        diagnostics = dict(clip_fraction=(clip < 1).float().mean().item(),
            mean_clip_factor=clip.mean().item(), mean_clipping_severity=(1-clip).mean().item(),
            mean_transformed_norm=norm.mean().item(), transformed_norm_p50=q[0],
            transformed_norm_p90=q[1], transformed_norm_p99=q[2], transformed_norm_max=norm.max().item())
        if smoke:
            torch.save(dict(model={n: p.detach().cpu() for n, p in model.state_dict().items()},
                norms=norm, factors=clip), run_dir/'smoke_state.pt')
        stats['diagnostic_postprocess_seconds'] = timestamp(device)-diagnostic_start
        evaluation_start = timestamp(device)
        test_loss, accuracy = evaluate(model, test_loader, device)
        stats['evaluation_seconds'] = timestamp(device)-evaluation_start
        for kind in ('allocated', 'reserved'):
            stats[f'total_peak_cuda_{kind}_bytes'] = max(stats[f'{p}_peak_cuda_{kind}_bytes'] for p in ('build', 'train'))
        rows.append(dict(method=method, seed=seed, epoch=epoch, train_loss=loss_sum/len(norm),
            test_loss=test_loss, test_accuracy=accuracy,
            epsilon=accountant.get_epsilon(delta=cfg.DELTA) if sigma else 0.,
            accountant_steps=sum(entry[2] for entry in accountant.history), noise_multiplier=sigma,
            **diagnostics, samples=len(norm), batches=len(loader),
            algorithm_epoch_seconds=stats['preconditioner_build_seconds']+stats['private_train_seconds'],
            wall_epoch_seconds=timestamp(device)-wall_start,
            private_samples_per_second=len(norm)/stats['private_train_seconds'],
            seconds_per_batch=stats['private_train_seconds']/len(loader), **stats))
        pd.DataFrame(rows).to_csv(run_dir/'metrics.csv', index=False)
        pd.DataFrame(layer_rows).to_csv(run_dir/'layer_norm_diagnostics.csv', index=False)
        print(f'{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} build={stats["preconditioner_build_seconds"]:.3f}s train={stats["private_train_seconds"]:.3f}s', flush=True)
    configuration = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    configuration.update(smoke=smoke, epochs=epochs, noise_multiplier=sigma, method=method, seed=seed,
        synthetic_batches=1 if smoke else 10, total_steps=steps, drop_last=True,
        method_order=cfg.ORDER, action_path_warmup=True,
        accounting='RDP, repository shuffled fixed-batch convention',
        rng=dict(initialization='seed', shuffle='seed', noise='seed+40000', synthetic='seed+10000+epoch', labels='seed+20000+epoch'))
    (run_dir/'config.json').write_text(json.dumps(configuration, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=cfg.METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=cfg.SEEDS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = HERE/'results'/'smoke' if args.smoke else HERE/'results'
    with runtime('cuda:0'):
        run(args.method, args.seed, args.smoke, output)


if __name__ == '__main__':
    main()

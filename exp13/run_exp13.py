"""Real MNIST DP training; --smoke is one zero-noise batch for each method."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import argparse
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from dp_kfac.models import SimpleCNN
from exp12.runtime import runtime
from exp13 import config as cfg
from exp13.builders import build_preconditioner
from exp13.ghost import GhostNorm, ghost_aggregate, noise_and_step
from exp13.analyze import save

OUTPUT = Path(__file__).parent / 'results'


def initialize(seed, device):
    torch.manual_seed(seed)
    return SimpleCNN().to(device)


def timestamp(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(x)
    return loss / count, correct / count


def run(method, seed, train, test, sigma, epochs, device, smoke=False):
    model = initialize(seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
                               momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    loader = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=cfg.BATCH_SIZE,
                             generator=torch.Generator().manual_seed(seed))
    noise_rng = torch.Generator(device=device).manual_seed(seed + 40000)
    accountant = RDPAccountant()
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        start = timestamp(device)
        operator, stats = build_preconditioner(model, method, seed, epoch, device,
                                               batches=1 if smoke else cfg.SYNTHETIC_BATCHES)
        build_seconds = timestamp(device) - start
        build_peak = torch.cuda.max_memory_allocated(device)
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
        test_loss, accuracy = evaluate(model, test_loader, device)
        quantiles = torch.cat(epoch_norms).quantile(torch.tensor([.5, .9, .99], device=device)).tolist()
        row = dict(method=method, seed=seed, epoch=epoch, test_accuracy=accuracy,
            test_loss=test_loss, train_loss=loss_sum/count,
            epsilon=accountant.get_epsilon(delta=cfg.DELTA) if sigma else None,
            noise_multiplier=sigma, clip_fraction=clipped/count, mean_clip_factor=factor_sum/count,
            transformed_norm_p50=quantiles[0], transformed_norm_p90=quantiles[1], transformed_norm_p99=quantiles[2],
            preconditioner_build_seconds=build_seconds, private_train_seconds=train_seconds,
            total_epoch_seconds=timestamp(device)-start, seconds_per_batch=train_seconds/len(loader),
            samples_per_second=count/train_seconds, build_peak_cuda_allocated_bytes=build_peak,
            train_peak_cuda_allocated_bytes=train_peak, total_peak_cuda_allocated_bytes=max(build_peak, train_peak),
            batches=len(loader), samples=count, accountant_steps=epoch*len(loader) if sigma else 0,
            **stats)
        rows.append(row)
        print(f'{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} '
              f'build={build_seconds:.2f}s train={train_seconds:.2f}s '
              f'VJP={stats["builder_vjp_calls"]} vectors={stats["builder_reverse_vectors"]}', flush=True)
        del operator, hooks, initial, epoch_norms
    return rows


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
    # Exactly one calibration shared by all methods and both seeds.
    sigma = 0. if args.smoke else get_noise_multiplier(target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA, sample_rate=cfg.BATCH_SIZE/len(train),
        steps=epochs*(len(train)//cfg.BATCH_SIZE), accountant='rdp')
    output = OUTPUT / 'smoke' if args.smoke else OUTPUT
    rows = []
    with runtime(device):  # Deterministic settings and CUDA/cuBLAS warmup before timing.
        for seed in seeds:
            for method in cfg.METHODS:
                rows.extend(run(method, seed, train, test, sigma, epochs, device, args.smoke))
                save(rows, output)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

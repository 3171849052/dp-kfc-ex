"""Synthetic Fisher Equilibration on full MNIST; run directly for all eight runs.

Diagnostics access private examples without DP and are analysis-only outputs.
privacy_valid describes the training mechanism, not release of these diagnostics
or the unnoised training metrics. KFC scale statistics are operator column norms.
"""
from pathlib import Path
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from dp_kfac.models import SimpleCNN
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, compute_inverse_sqrt
from dp_kfac.precondition import precondition_per_sample_gradients
from dp_kfac.privacy import (clip_and_noise_gradients,
                            _compute_per_sample_norms_squared, _compute_clip_factors)

METHODS = ("DP-SGD", "Synthetic Diag Fisher", "Synthetic Equil", "Synthetic DP-KFC")
SEEDS = (42, 7)
EPOCHS = 5
BATCH_SIZE = 256
PRECOND_STEPS = 10
PROBES = 8
DIAGNOSTIC_SAMPLES = 2048
SCALE_FIELDS = ("scale_min", "scale_p10", "scale_median", "scale_p90",
                "scale_max", "scale_cap_fraction")


def pink_batches(batch_size, device):
    for _ in range(PRECOND_STEPS):
        yield (generate_pink_noise(batch_size, (1, 28, 28), device),
               torch.randint(0, 10, (batch_size,), device=device))


def rademacher(params):
    return {p: torch.randint(0, 2, (p.numel(), PROBES), device=p.device)
            .to(p.dtype).mul_(2).sub_(1) for p in params}


def fisher_statistics(model, batches, device, probes=None):
    """Store only parameter blocks, B x K projections, and d x K accumulators."""
    params = [p for p in model.parameters() if p.requires_grad]
    diagonal = {p: torch.zeros(p.numel(), device=device) for p in params}
    products = {p: torch.zeros_like(probes[p]) for p in params} if probes is not None else {}
    count = 0
    model.train()
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        # GSM uses sum reduction: these are unscaled per-example gradients.
        F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        with torch.no_grad():
            if probes is not None:
                projection = torch.zeros(len(x), PROBES, device=device)
                for p in params:
                    projection.add_(p.grad_sample.flatten(1) @ probes[p])
            for p in params:
                block = p.grad_sample.flatten(1)
                diagonal[p].add_(block.square().sum(0))
                if probes is not None:
                    products[p].add_(block.T @ projection)
        count += len(x)
    model.zero_grad(set_to_none=True)
    diagonal = {p: (v / count).view_as(p) for p, v in diagonal.items()}
    equil = {p: (v / count).square().mean(1).sqrt().view_as(p)
             for p, v in products.items()}
    return diagonal, equil


@torch.no_grad()
def stabilized_scales(statistic):
    values = torch.cat([v.flatten() for v in statistic.values()])
    gamma = 1e-2 * values.median()
    scales = {p: (v + gamma).rsqrt() for p, v in statistic.items()}
    # One geometric mean over all coordinates, including every bias.
    log_mean = sum(v.log().sum() for v in scales.values()) / values.numel()
    normalizer = log_mean.exp()
    return {p: (v / normalizer).clamp(0.1, 10) for p, v in scales.items()}


def kfac_factors(model, batch_size, device):
    recorder = KFACRecorder(model)
    recorder.enable()
    for index, (x, y) in enumerate(pink_batches(batch_size, device)):
        model.zero_grad(set_to_none=True)
        # Preserve the existing synthetic KFAC mean-loss calibration convention.
        F.cross_entropy(model(x), y).backward()
        cov = compute_covariances(model, recorder.activations, recorder.backprops)
        if index == 0:
            total = cov
        else:
            for name in total.A:
                total.A[name].add_(cov.A[name])
                total.G[name].add_(cov.G[name])
        recorder.clear()
    recorder.remove()
    model.zero_grad(set_to_none=True)
    for name in total.A:
        total.A[name].div_(PRECOND_STEPS)
        total.G[name].div_(PRECOND_STEPS)
    return compute_inverse_sqrt(total, damping=1e-3)


@torch.no_grad()
def kfac_scales(model, a, g):
    # ||P e_i||_2 for P = A tensor-product G, including augmented biases.
    # KFC mixes coordinates; these describe its gains, not diagonal multipliers.
    scales = {}
    for name, module in model._module.named_modules():
        if name in a:
            gains = g[name].norm(dim=0)[:, None] * a[name].norm(dim=0)[None, :]
            scales[module.weight] = gains[:, :-1].reshape_as(module.weight)
            scales[module.bias] = gains[:, -1].reshape_as(module.bias)
    return scales


@torch.no_grad()
def scale_statistics(scales):
    values = torch.cat([v.flatten() for v in scales.values()])
    quantiles = values.quantile(values.new_tensor([0., .1, .5, .9, 1.])).tolist()
    # For KFC (which is not clamped), report the fraction outside these bounds.
    cap_fraction = ((values <= .1) | (values >= 10)).float().mean().item()
    return dict(zip(SCALE_FIELDS, quantiles + [cap_fraction]))


def timestamp(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct = 0., 0
    for x, y in loader:
        y = y.to(device)
        output = model(x.to(device))
        loss += F.cross_entropy(output, y, reduction="sum").item()
        correct += (output.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def diagnostic(model, loader, seed, batch_size, device):
    # No optimizer/accountant calls. These statistics never feed training.
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(seed + 20000)
        probes = rademacher([p for p in model.parameters() if p.requires_grad])
        private_d, private_e = fisher_statistics(model, loader, device, probes)
        synthetic_d, synthetic_e = fisher_statistics(
            model, pink_batches(batch_size, device), device, probes)
    d, e, sd, se = [torch.cat([v.flatten() for v in stat.values()]).double()
                    for stat in (private_d, private_e, synthetic_d, synthetic_e)]
    corr_d = torch.corrcoef(torch.stack(((d + 1e-12).log(), (sd + 1e-12).log())))[0, 1]
    corr_e = torch.corrcoef(torch.stack(((e + 1e-12).log(), (se + 1e-12).log())))[0, 1]
    coupling = (e / (d + 1e-12)).quantile(d.new_tensor([.5, .9, .99])).tolist()
    return dict(corr_log_diag=corr_d.item(), corr_log_equil=corr_e.item(),
                median_coupling_ratio=coupling[0], p90_coupling_ratio=coupling[1],
                p99_coupling_ratio=coupling[2])


def run(method, seed, train, test, epochs, batch_size, sigma, device):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=.1, momentum=.9)
    # Explicit loader generators isolate iterator seed draws from DP noise.
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=batch_size,
                             generator=torch.Generator().manual_seed(seed))
    diagnostic_loader = DataLoader(Subset(train, range(DIAGNOSTIC_SAMPLES)),
        batch_size=batch_size, generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        stats = dict.fromkeys(SCALE_FIELDS, float("nan"))
        start = timestamp(device)
        if method != METHODS[0]:
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.manual_seed(seed + 10000 + epoch)
                if method == METHODS[3]:
                    a, g = kfac_factors(model, batch_size, device)
                    stats = scale_statistics(kfac_scales(model, a, g))
                else:
                    # Isolate probe draws so all methods use identical pink samples.
                    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                        torch.manual_seed(seed + 30000 + epoch)
                        probes = rademacher(params) if method == METHODS[2] else None
                    d, e = fisher_statistics(model, pink_batches(batch_size, device), device, probes)
                    scales = stabilized_scales(e if method == METHODS[2] else d)
                    stats = scale_statistics(scales)
        precond_seconds = timestamp(device) - start
        start = timestamp(device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            if method == METHODS[3]:
                precondition_per_sample_gradients(model, a, g)
            elif method in METHODS[1:3]:
                with torch.no_grad():
                    for p in params:
                        p.grad_sample.mul_(scales[p])
            norms_sq = _compute_per_sample_norms_squared(params, len(x), device)
            factors = _compute_clip_factors(norms_sq, 1.)
            clipped += (norms_sq.sqrt() > 1.).sum().item()
            factor_sum += factors.sum().item()
            # Precondition -> global per-example clip -> isotropic noise -> SGD.
            clip_and_noise_gradients(model, sigma, 1., len(x))
            optimizer.step()
            # Use the calibrated upper-bound rate, also for the last partial batch.
            accountant.step(noise_multiplier=sigma, sample_rate=batch_size / len(train))
            loss_sum += loss.item()
            count += len(x)
        train_seconds = timestamp(device) - start
        test_loss, accuracy = evaluate(model, test_loader, device)
        row = dict(method=method, seed=seed, epoch=epoch, privacy_valid=True,
            train_loss=loss_sum/count, test_loss=test_loss, test_accuracy=accuracy,
            epsilon_spent=accountant.get_epsilon(delta=1e-5), clip_fraction=clipped/count,
            mean_clip_factor=factor_sum/count, precond_seconds=precond_seconds,
            epoch_train_seconds=train_seconds, **stats)
        rows.append(row)
        print(f"{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} "
              f"epsilon={row['epsilon_spent']:.4f}", flush=True)
    analysis = dict(method=method, seed=seed,
                    **diagnostic(model, diagnostic_loader, seed, batch_size, device))
    return rows, analysis


def save_results(rows, diagnostics, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    frame[["method", "seed", "epoch", "precond_seconds", *SCALE_FIELDS]].to_csv(
        output / "preconditioner_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    final = frame.groupby(["method", "seed"], sort=False).tail(1)
    final.groupby("method", sort=False).agg(
        mean_accuracy=("test_accuracy", "mean"), std_accuracy=("test_accuracy", "std"),
        mean_clipping=("clip_fraction", "mean"), mean_clip_factor=("mean_clip_factor", "mean"),
        mean_precond_seconds=("precond_seconds", "mean"),
        mean_epoch_train_seconds=("epoch_train_seconds", "mean"),
    ).to_csv(output / "summary.csv")
    for metric, filename in (("test_accuracy", "accuracy.png"), ("clip_fraction", "clipping.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, method in enumerate(METHODS):
            part = frame[frame.method == method]
            for _, seed_rows in part.groupby("seed"):
                ax.plot(seed_rows.epoch, seed_rows[metric], color=f"C{i}", alpha=.3)
            mean = part.groupby("epoch")[metric].mean()
            ax.plot(mean.index, mean.values, marker="o", color=f"C{i}", label=method)
        ax.set(xlabel="Epoch", ylabel=metric, xticks=sorted(frame.epoch.unique()))
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, method in zip(axes, METHODS[1:]):
        part = frame[frame.method == method].groupby("epoch")[list(SCALE_FIELDS)].mean()
        ax.fill_between(part.index, part.scale_min, part.scale_max, alpha=.12, label="min–max")
        ax.fill_between(part.index, part.scale_p10, part.scale_p90, alpha=.3, label="p10–p90")
        ax.plot(part.index, part.scale_median, marker="o", label="median")
        ax.set(title=method, xlabel="Epoch", ylabel="Scale (seed mean)", yscale="log")
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "scale_distribution.png", dpi=160)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train = datasets.MNIST(ROOT / "exp1/data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "exp1/data", train=False, download=True, transform=transform)
    sigma = get_noise_multiplier(target_epsilon=1., target_delta=1e-5,
        sample_rate=BATCH_SIZE/len(train), steps=EPOCHS*math.ceil(len(train)/BATCH_SIZE),
        accountant="rdp")
    output = ROOT / "exp6/results"
    print(f"device={device}, noise_multiplier={sigma}, output={output}", flush=True)
    rows, diagnostics = [], []
    for seed in SEEDS:
        for method in METHODS:
            run_rows, analysis = run(method, seed, train, test, EPOCHS, BATCH_SIZE, sigma, device)
            rows.extend(run_rows)
            diagnostics.append(analysis)
            save_results(rows, diagnostics, output)


if __name__ == "__main__":
    main()

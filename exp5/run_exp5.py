"""MNIST: synthetic KFAC versus diagnostic oracle KFAC/full Fisher whitening."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, compute_inverse_sqrt
from dp_kfac.precondition import precondition_per_sample_gradients
from dp_kfac.privacy import (clip_and_noise_gradients,
                             _compute_per_sample_norms_squared, _compute_clip_factors)
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.types import CovariancePair

METHODS = ("DP-SGD", "Synthetic DP-KFC", "Oracle KFAC", "Full Fisher Whitening (Oracle)")
PRECOND_STEPS = 10
BATCH_SIZE = 256
DAMPING = 1e-3


class WhiteningCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 2 * 2, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(F.adaptive_avg_pool2d(x, (2, 2)).flatten(1))


def pink_batches(batch_size, device):
    for _ in range(PRECOND_STEPS):
        yield (generate_pink_noise(batch_size, (1, 28, 28), device),
               torch.randint(0, 10, (batch_size,), device=device))


def kfac_factors(model, batches, device):
    recorder = KFACRecorder(model)
    recorder.enable()
    totals = CovariancePair(A={}, G={})
    count = 0
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        cov = compute_covariances(model, recorder.activations, recorder.backprops, eps=0.0)
        for total, current in ((totals.A, cov.A), (totals.G, cov.G)):
            for name, value in current.items():
                if name not in total:
                    total[name] = torch.zeros_like(value)
                total[name].add_(value, alpha=len(x))
        count += len(x)
        recorder.clear()
    recorder.remove()
    model.zero_grad(set_to_none=True)
    assert set(totals.A) == set(totals.G) == {"conv1", "conv2", "fc"}
    for total in (totals.A, totals.G):
        for value in total.values():
            value.div_(count)
    return compute_inverse_sqrt(totals, damping=DAMPING)


def flat_grad_samples(params):
    return torch.cat([p.grad_sample.flatten(1) for p in params], dim=1)


def full_fisher(model, loader, device):
    params = [p for p in model.parameters() if p.requires_grad]
    samples = []
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        samples.append(flat_grad_samples(params).double())
    gradients = torch.cat(samples)
    assert gradients.shape == (len(loader.dataset), 994)
    fisher = gradients.T @ gradients / len(gradients)
    identity = torch.eye(fisher.shape[0], dtype=torch.float64, device=device)
    regularized = fisher + DAMPING * identity
    eigenvalues, q = torch.linalg.eigh(regularized)
    assert (eigenvalues > 0).all()
    p = (q * eigenvalues.rsqrt()) @ q.T
    assert torch.allclose(p @ regularized @ p.T, identity, atol=1e-7, rtol=1e-7)
    model.zero_grad(set_to_none=True)
    return p.to(dtype=params[0].dtype)


@torch.no_grad()
def whiten(model, p):
    params = [param for param in model.parameters() if param.requires_grad]
    gradients = flat_grad_samples(params) @ p.T
    for param, part in zip(params, gradients.split([v.numel() for v in params], dim=1)):
        param.grad_sample = part.reshape(len(gradients), *param.shape)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct = 0., 0
    for x, y in loader:
        y = y.to(device)
        out = model(x.to(device))
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def run(method, seed, train, test, calibration, epochs, batch_size, sigma, device):
    torch.manual_seed(seed)
    model = GradSampleModule(WhiteningCNN().to(device), loss_reduction="sum")
    assert sum(p.numel() for p in model.parameters()) == 994
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    # Even non-shuffled DataLoader iterators draw a base seed: isolate these draws.
    calibration_loader = DataLoader(calibration, batch_size=batch_size, shuffle=False,
                                   generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=256,
                            generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        if method == METHODS[1]:
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.manual_seed(seed + 10000 + epoch)
                a, g = kfac_factors(model, pink_batches(batch_size, device), device)
        elif method == METHODS[2]:
            a, g = kfac_factors(model, calibration_loader, device)
        elif method == METHODS[3]:
            p = full_fisher(model, calibration_loader, device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            if method in METHODS[1:3]:
                precondition_per_sample_gradients(model, a, g)
            elif method == METHODS[3]:
                whiten(model, p)
            norms_sq = _compute_per_sample_norms_squared(list(model.parameters()), batch_size, device)
            factors = _compute_clip_factors(norms_sq, 1.0)
            clipped += (norms_sq.sqrt() > 1.0).sum().item()
            factor_sum += factors.sum().item()
            # Precondition -> global per-example clip -> isotropic noise -> SGD.
            clip_and_noise_gradients(model, sigma, 1.0, batch_size)
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=batch_size / len(train))
            loss_sum += loss.item()
            count += batch_size
        test_loss, accuracy = evaluate(model, test_loader, device)
        # For oracles this accounts for training noise only, not calibration access.
        row = dict(method=method, seed=seed, epoch=epoch, privacy_valid=method in METHODS[:2],
                   train_loss=loss_sum/count, test_loss=test_loss, test_accuracy=accuracy,
                   epsilon_spent=accountant.get_epsilon(delta=1e-5),
                   clip_fraction=clipped/count, mean_clip_factor=factor_sum/count)
        rows.append(row)
        print(f"{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} "
              f"epsilon_spent={row['epsilon_spent']:.4f} privacy_valid={row['privacy_valid']}", flush=True)
    return rows


def save_results(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    final = frame[frame.epoch == frame.epoch.max()]
    summary = final.groupby("method", sort=False).agg(
        privacy_valid=("privacy_valid", "first"),
        mean_accuracy=("test_accuracy", "mean"), std_accuracy=("test_accuracy", "std"))
    summary.to_csv(output / "summary.csv")  # Sample std is undefined for the one-seed smoke.
    acc = summary.mean_accuracy
    gaps = dict(source_gap_pp=100 * (acc[METHODS[2]] - acc[METHODS[1]]),
                geometry_gap_pp=100 * (acc[METHODS[3]] - acc[METHODS[2]]),
                total_full_whitening_gain_pp=100 * (acc[METHODS[3]] - acc[METHODS[1]]))
    pd.DataFrame([gaps]).to_csv(output / "comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(METHODS):
        part = frame[frame.method == method]
        for _, seed_rows in part.groupby("seed"):
            ax.plot(seed_rows.epoch, seed_rows.test_accuracy, color=f"C{i}", alpha=.3)
        mean = part.groupby("epoch").test_accuracy.mean()
        ax.plot(mean.index, mean.values, marker="o", color=f"C{i}", label=method)
    ax.set(xlabel="Epoch", ylabel="Test accuracy", xticks=sorted(frame.epoch.unique()))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)
    for name, value in gaps.items():
        print(f"{'>>> ' if name == 'total_full_whitening_gain_pp' else ''}{name} = {value:+.4f} pp", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(ROOT / "exp1/data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "exp1/data", train=False, download=True, transform=transform)
    epochs, batch_size, seeds = 5, BATCH_SIZE, (42, 7)
    output = ROOT / "exp5/results"
    if args.smoke:
        train, test = Subset(train, range(40)), Subset(test, range(16))
        epochs, batch_size, seeds = 1, 4, (42,)
        output = output / "smoke"
    calibration = Subset(train, range(PRECOND_STEPS * batch_size))
    sigma = get_noise_multiplier(target_epsilon=1.0, target_delta=1e-5,
        sample_rate=batch_size/len(train), steps=epochs*(len(train)//batch_size), accountant="rdp")
    print(f"device={device}, noise_multiplier={sigma}, output={output}", flush=True)
    rows = []
    for seed in seeds:
        for method in METHODS:
            rows.extend(run(method, seed, train, test, calibration, epochs, batch_size, sigma, device))
    save_results(rows, output)


if __name__ == "__main__":
    main()

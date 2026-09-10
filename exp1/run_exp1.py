"""Fixed MNIST Exp1; --smoke runs the same pipeline on tiny subsets."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp1"))
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
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, compute_inverse_sqrt
from dp_kfac.precondition import precondition_per_sample_gradients
from dp_kfac.privacy import (clip_and_noise_gradients, _compute_per_sample_norms_squared,
                             _compute_clip_factors)
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.types import KFACConfig
from clw_kron import CLWKron, LAYERS, identity_factors

METHODS = ("DP-SGD", "Synthetic DP-KFC (Pink Noise)", "CLW-Kron")


def pink_factors(model, batch_size, device):
    recorder = KFACRecorder(model)
    recorder.enable()
    model.zero_grad(set_to_none=True)
    x = generate_pink_noise(batch_size, (1, 28, 28), device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    F.cross_entropy(model(x), y).backward()  # Same mean reduction as existing Pink KFAC.
    cov = compute_covariances(model, recorder.activations, recorder.backprops)
    factors = compute_inverse_sqrt(cov, damping=KFACConfig().damping)
    recorder.remove()
    model.zero_grad(set_to_none=True)
    return factors


def oracle(model, loader, a, g, device):
    # SimpleCNN has no stochastic/train-mode layers. Opacus needs train mode.
    model.train()
    cg = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in g.items()}
    ca = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in a.items()}
    recorder = KFACRecorder(model)
    recorder.enable()
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        # Sum reduction gives unscaled per-example loss backprops.
        F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        with torch.no_grad():
            cov = compute_covariances(
                model,
                {k: v.double() for k, v in recorder.activations.items()},
                {k: v.double() for k, v in recorder.backprops.items()},
                eps=0.0,  # Oracle factors must not acquire an artificial rank floor.
            )
            for name in LAYERS:
                cg[name].add_(cov.G[name], alpha=len(x))
                ca[name].add_(cov.A[name], alpha=len(x))
        recorder.clear()
    recorder.remove()
    model.zero_grad(set_to_none=True)
    rows = []
    for name in LAYERS:
        # Each layer has the same spatial size across batches; sample weighting
        # therefore also weights Conv2d patch averages correctly.
        oracle_a = ca[name] / len(loader.dataset)
        oracle_g = cg[name] / len(loader.dataset)
        ua, ug = a[name].double(), g[name].double()
        transformed_a = ua.T @ oracle_a @ ua
        transformed_g = ug @ oracle_g @ ug.T
        gbar = transformed_g / (transformed_g.trace() / transformed_g.shape[0])
        abar = transformed_a / (transformed_a.trace() / transformed_a.shape[0])
        m, n = gbar.shape[0], abar.shape[0]
        logs = []
        for c in (gbar, abar):
            ev = torch.linalg.eigvalsh(c)
            # Relative floating-point rank tolerance, not regularization.
            tol = c.shape[0] * torch.finfo(c.dtype).eps * ev[-1]
            logs.append(float("inf") if ev[0] <= tol else (ev[-1].log() - ev[0].log()).item())
        rows.append(dict(layer=name,
            W_kron=((abar.square().sum() * gbar.square().sum() / (m*n) - 1).clamp(min=0).sqrt().item()),
            W_G=((gbar - torch.eye(m, device=device)).norm() / m**0.5).item(),
            W_A=((abar - torch.eye(n, device=device)).norm() / n**0.5).item(),
            log_kappa_kron=sum(logs)))
    return rows


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


def run(method, seed, train, test_loader, diagnostic_loader, epochs, batch_size, sigma, device):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    controller = CLWKron(model) if method == "CLW-Kron" else None
    a, g = identity_factors(model)
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        if method == METHODS[1]:
            # Synthetic draws must not shift the DP-noise stream across methods.
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.manual_seed(seed + 10000 + epoch)
                a, g = pink_factors(model, batch_size, device)
        if controller is not None:
            a, g = controller.U_A, controller.U_G
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            precondition_per_sample_gradients(model, a, g)
            norms_sq = _compute_per_sample_norms_squared(list(model.parameters()), batch_size, device)
            factors = _compute_clip_factors(norms_sq, 1.0)
            clipped += (norms_sq.sqrt() > 1.0).sum().item()
            factor_sum += factors.sum().item()
            clip_and_noise_gradients(model, sigma, 1.0, batch_size)
            if controller is not None:
                controller.observe(model)
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=batch_size / len(train))
            loss_sum += loss.item()
            count += batch_size
        test_loss, accuracy = evaluate(model, test_loader, device)
        diagnostics = oracle(model, diagnostic_loader, a, g, device)
        common = dict(method=method, seed=seed, epoch=epoch, train_loss=loss_sum/count,
                      test_loss=test_loss, test_accuracy=accuracy,
                      epsilon_spent=accountant.get_epsilon(delta=1e-5),
                      clip_fraction=clipped/count, mean_clip_factor=factor_sum/count)
        rows.extend(dict(**common, **d) for d in diagnostics)
        print(f"{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} epsilon={common['epsilon_spent']:.4f}", flush=True)
        if controller is not None:
            controller.update()
    return rows


def plot_lines(ax, frame, metric):
    for color, method in zip(("C0", "C1", "C2"), METHODS):
        part = frame[frame.method == method]
        for _, seed_rows in part.groupby("seed"):
            ax.plot(seed_rows.epoch, seed_rows[metric], color=color, alpha=.35, linewidth=.8)
        mean = part.groupby("epoch")[metric].mean()
        ax.plot(mean.index, mean.values, color=color, linewidth=2.5, label=method)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=.2)


def save_results(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output / "metrics.csv", index=False)
    final = df[df.epoch == df.epoch.max()]
    metrics = [c for c in df.columns if c not in ("method", "seed", "epoch", "layer")]
    final.groupby(["method", "layer"])[metrics].mean().reset_index().to_csv(output / "summary.csv", index=False)
    for metric, filename in (("W_kron", "whitening_by_layer.png"),
                             ("log_kappa_kron", "condition_by_layer.png")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, name in zip(axes.flat, LAYERS):
            part = df[df.layer == name]
            plot_lines(ax, part, metric)
            ax.set_title(name)
            if metric == "log_kappa_kron" and (part[metric] == float("inf")).any():
                ax.text(.02, .98, "Singular values: +inf (see CSV)", transform=ax.transAxes, va="top", fontsize=8)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, df[df.layer == "conv1"], "test_accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="One tiny end-to-end run, output in results/smoke")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(ROOT / "exp1/data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "exp1/data", train=False, download=True, transform=transform)
    diagnostic = Subset(train, range(2048))
    epochs, batch_size, seeds = 5, 256, (42, 7)
    output = ROOT / "exp1/results"
    if args.smoke:
        train, test, diagnostic = Subset(train, range(8)), Subset(test, range(8)), Subset(train, range(8))
        epochs, batch_size, seeds = 2, 4, (42,)
        output = output / "smoke"
    sigma = get_noise_multiplier(target_epsilon=1.0, target_delta=1e-5,
        sample_rate=batch_size/len(train), steps=epochs*(len(train)//batch_size), accountant="rdp")
    print(f"device={device}, noise_multiplier={sigma}, output={output}", flush=True)
    test_loader = DataLoader(test, batch_size=256)
    diagnostic_loader = DataLoader(diagnostic, batch_size=16)
    rows = []
    for seed in seeds:
        for method in METHODS:
            rows.extend(run(method, seed, train, test_loader, diagnostic_loader,
                            epochs, batch_size, sigma, device))
            save_results(rows, output)


if __name__ == "__main__":
    main()

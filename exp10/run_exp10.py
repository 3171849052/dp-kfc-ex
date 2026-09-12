"""Exp10: paired Full-Fisher versus module-wise Fisher Equil on MNIST.

Only projection scope differs. Gamma and geometric normalization are global.
Accounting and build + private-train epoch timing follow Exp9. Unnoised
training diagnostics are analysis-only, as in Exp7/9.
"""
from pathlib import Path
import argparse
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp9.run_exp9 import (
    torch, F, pd, plt, DataLoader, datasets, transforms, GradSampleModule,
    RDPAccountant, get_noise_multiplier, SimpleCNN, rademacher,
    fisher_statistics, equil_scales, pink_batches, clip_and_noise_gradients,
    _compute_per_sample_norms_squared, _compute_clip_factors, evaluate, timestamp,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPSILON, DELTA,
    MAX_GRAD_NORM, SYNTHETIC_BATCH_SIZE, PROBES, GAIN_FIELDS,
)
from exp7.run_exp7 import LAYERS, SEEDS
from torch.utils.data import Subset

OUTPUT = ROOT / "exp10/results"
METHODS = ("DP-SGD", "Layerwise Equil", "Full-Fisher Equil")
TAU = 1.0
PRECONDITIONER_BATCHES = 10


def layerwise_statistics(model, batches, device, probes):
    groups = [list(getattr(model._module, name).parameters()) for name in LAYERS]
    products = {p: torch.zeros_like(z) for p, z in probes.items()}
    count = 0
    model.train()
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        with torch.no_grad():
            for group in groups:
                projection = torch.zeros(len(x), PROBES, device=device)
                for p in group:
                    projection.add_(p.grad_sample.flatten(1) @ probes[p])
                for p in group:
                    products[p].add_(p.grad_sample.flatten(1).T @ projection)
        count += len(x)
    model.zero_grad(set_to_none=True)
    return {p: (v / count).square().mean(1).sqrt().view_as(p)
            for p, v in products.items()}


def build_preconditioner(model, method, seed, epoch, device):
    if method == "DP-SGD":
        return {}, dict.fromkeys(GAIN_FIELDS, 1.0)
    # Same nested fork_rng and seeds as Exp7/9; private noise RNG is restored.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed + 10000 + epoch)
        batches = pink_batches(PRECONDITIONER_BATCHES, device)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed + 30000 + epoch)
            probes = rademacher((p for p in model.parameters() if p.requires_grad), PROBES)
        if method == "Full-Fisher Equil":
            _, statistic = fisher_statistics(model, batches, device, probes)
        else:
            statistic = layerwise_statistics(model, batches, device, probes)
        scales, gains = equil_scales(statistic, TAU)
    return dict(scales=scales), gains


def run(method, seed, train, test, sigma, device, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=BATCH_SIZE,
                             generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    rows, state = [], {}
    for epoch in range(1, epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        state.clear()
        start = timestamp(device)
        state, gains = build_preconditioner(model, method, seed, epoch, device)
        build_seconds = timestamp(device) - start
        start = timestamp(device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            with torch.no_grad():
                if method != "DP-SGD":
                    for p, scale in state["scales"].items():
                        p.grad_sample.mul_(scale)
            norms_sq = _compute_per_sample_norms_squared(params, len(x), device)
            factors = _compute_clip_factors(norms_sq, MAX_GRAD_NORM)
            clipped += (norms_sq.sqrt() > MAX_GRAD_NORM).sum().item()
            factor_sum += factors.sum().item()
            # Precondition -> global per-example clipping -> isotropic noise -> SGD.
            clip_and_noise_gradients(model, sigma, MAX_GRAD_NORM, len(x))
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=BATCH_SIZE / len(train))
            loss_sum += loss.item()
            count += len(x)
        train_seconds = timestamp(device) - start
        test_loss, accuracy = evaluate(model, test_loader, device)
        rows.append(dict(method=method, tau=TAU, preconditioner_batches=PRECONDITIONER_BATCHES,
            synthetic_samples=SYNTHETIC_BATCH_SIZE*PRECONDITIONER_BATCHES, seed=seed, epoch=epoch,
            test_accuracy=accuracy, test_loss=test_loss, train_loss=loss_sum/count,
            epsilon_spent=accountant.get_epsilon(delta=DELTA),
            clip_fraction=clipped/count, mean_clip_factor=factor_sum/count,
            precond_build_seconds=build_seconds, private_train_seconds=train_seconds,
            total_epoch_seconds=build_seconds+train_seconds,
            **gains))
        print(f"{method} seed={seed} epoch={epoch} "
              f"accuracy={accuracy:.4f} build={build_seconds:.2f}s "
              f"train={train_seconds:.2f}s", flush=True)
    model.remove_hooks()
    return rows


def save_results(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    final = frame.groupby(["method", "seed"], sort=False).tail(1).copy()
    stats = final.groupby("method").test_accuracy.agg(["mean", "std"])
    final["mean_final_accuracy"] = final.method.map(stats["mean"])
    final["std_final_accuracy"] = final.method.map(stats["std"])
    final.to_csv(output / "summary.csv", index=False)
    return frame, final


def plot_results(frame, final, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(METHODS):
        part = frame[frame.method == method]
        for _, seed_rows in part.groupby("seed"):
            ax.plot(seed_rows.epoch, seed_rows.test_accuracy * 100, color=f"C{i}", alpha=.3)
        mean = part.groupby("epoch").test_accuracy.mean()
        ax.plot(mean.index, mean * 100, marker="o", color=f"C{i}", label=method)
    ax.set(xlabel="Epoch", ylabel="Test accuracy (%)", xticks=sorted(frame.epoch.unique()))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)
    paired = final.pivot(index="seed", columns="method", values="test_accuracy")
    delta = (paired["Full-Fisher Equil"] - paired["Layerwise Equil"]) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(delta.index.astype(str), delta.values)
    ax.bar_label(bars, fmt="%+.2f", padding=3)
    ax.axhline(0, color="black", linewidth=.8)
    ax.margins(y=.2)
    ax.set(xlabel="Seed", ylabel="Full − Layerwise (percentage points)",
           title="Paired final accuracy difference")
    fig.tight_layout()
    fig.savefig(output / "paired_final_accuracy_delta.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="One epoch, seed 42, 512 train/test samples; results/smoke.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    output = OUTPUT / "smoke" if args.smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    # Read the existing MNIST cache without modifying previous experiments.
    train = datasets.MNIST(ROOT / "exp9/data", train=True, download=False, transform=transform)
    test = datasets.MNIST(ROOT / "exp9/data", train=False, download=False, transform=transform)
    epochs, seeds = EPOCHS, SEEDS
    if args.smoke:
        train, test = Subset(train, range(512)), Subset(test, range(512))
        epochs, seeds = 1, (42,)
    sigma = get_noise_multiplier(target_epsilon=EPSILON, target_delta=DELTA,
        sample_rate=BATCH_SIZE/len(train), steps=epochs*math.ceil(len(train)/BATCH_SIZE), accountant="rdp")
    print(f"device={device}, noise_multiplier={sigma}, epochs={epochs}, output={output}", flush=True)
    rows = []
    for seed in seeds:
        for method in METHODS:
            rows.extend(run(method, seed, train, test, sigma, device, epochs))
            frame, final = save_results(rows, output)
    plot_results(frame, final, output)
    print(final[["method", "seed", "test_accuracy", "mean_final_accuracy", "std_final_accuracy"]]
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

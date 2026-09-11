"""Residual DP-KFC prototype; --smoke runs the same pipeline on tiny data."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp1"))
from run_exp1 import (oracle, evaluate, torch, F, pd, plt, DataLoader, Subset,
                      datasets, transforms, GradSampleModule, RDPAccountant,
                      get_noise_multiplier, SimpleCNN, precondition_per_sample_gradients,
                      clip_and_noise_gradients, _compute_per_sample_norms_squared,
                      _compute_clip_factors, LAYERS)
from run_exp1 import pink_factors
from residual_kron import ResidualKron, interval_ends

METHODS = ("Residual-Noisy", "Residual-Clean")


def run(method, seed, train, test_loader, diagnostic_loader, epochs, batch_size, sigma, device):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    sensor = "summed_grad" if "Clean" in method else "grad"
    ends = interval_ends(len(loader))
    rows, feedback = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(seed + 10000 + epoch)
            base_a, base_g = pink_factors(model, batch_size, device)
        controller = ResidualKron(model, base_a, base_g, sensor)
        base_reset_error = max(
            (controller.U_A[k]-base_a[k]).abs().max().item()
            + (controller.U_G[k]-base_g[k]).abs().max().item() for k in LAYERS)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        interval, previous_end = 1, 0
        for step, (x, y) in enumerate(loader, 1):
            a, g = dict(controller.U_A), dict(controller.U_G)
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            precondition_per_sample_gradients(model, a, g)
            norms_sq = _compute_per_sample_norms_squared(list(model.parameters()), batch_size, device)
            factors = _compute_clip_factors(norms_sq, 1.0)
            clipped += (norms_sq.sqrt() > 1.0).sum().item()
            factor_sum += factors.sum().item()
            clip_and_noise_gradients(model, sigma, 1.0, batch_size, store_summed_grad=True)
            controller.observe(model)
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=batch_size / len(train))
            loss_sum += loss.item()
            count += batch_size
            if step in ends:
                common = dict(method=method, seed=seed, epoch=epoch, interval=interval,
                              global_update_index=(epoch-1)*4+interval,
                              interval_steps=step-previous_end, feedback_type=sensor,
                              update_after_step=step, applies_from_step=step+1)
                feedback.extend(dict(**common, **d) for d in controller.update())
                previous_end = step
                interval += 1
        assert controller.update_steps == ends
        assert controller.steps == len(loader)
        test_loss, accuracy = evaluate(model, test_loader, device)
        # SimpleCNN and this sequential loader are deterministic; neither oracle
        # pass updates parameters. Reuse Exp1's exact diagnostic on the same model.
        base = oracle(model, diagnostic_loader, base_a, base_g, device)
        # The additional loader iterator must not shift next epoch's DP-noise RNG.
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            corrected = oracle(model, diagnostic_loader, a, g, device)
        diagnostics = []
        for b, c in zip(base, corrected):
            d = dict(layer=b['layer'])
            d.update({f'base_{k}': v for k, v in b.items() if k != 'layer'})
            d.update({f'corrected_{k}': v for k, v in c.items() if k != 'layer'})
            d.update({f'gain_{side}': (b[f'W_{side}']-c[f'W_{side}'])/b[f'W_{side}']
                      for side in ('A', 'G', 'kron')})
            diagnostics.append(d)
        common = dict(method=method, seed=seed, epoch=epoch, privacy_valid=sensor == "grad",
                      base_reset_error=base_reset_error, train_loss=loss_sum/count, test_loss=test_loss, test_accuracy=accuracy,
                      epsilon_spent=accountant.get_epsilon(delta=1e-5),
                      clip_fraction=clipped/count, mean_clip_factor=factor_sum/count)
        rows.extend(dict(**common, **d) for d in diagnostics)
        print(f"{method} seed={seed} epoch={epoch} updates={len(controller.update_steps)} accuracy={accuracy:.4f}", flush=True)
    return rows, feedback


def plot_lines(ax, frame, metric, style="-"):
    for color, method in zip(("C0", "C1"), METHODS):
        part = frame[frame.method == method]
        for _, seed_rows in part.groupby("seed"):
            ax.plot(seed_rows.epoch, seed_rows[metric], color=color, alpha=.3,
                    linewidth=.8, linestyle=style)
        mean = part.groupby("epoch")[metric].mean()
        ax.plot(mean.index, mean.values, color=color, linewidth=2.5,
                linestyle=style, label=f"{method}: {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=.2)


def save_results(rows, feedback, output):
    output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(feedback).to_csv(output / "controller_metrics.csv", index=False)
    final = df[df.epoch == df.epoch.max()]
    metrics = [c for c in df.columns if c not in ("method", "seed", "epoch", "layer", "privacy_valid")]
    final.groupby(["method", "layer", "privacy_valid"])[metrics].mean().reset_index().to_csv(output / "summary.csv", index=False)
    for filename in ("whitening_gain_by_layer.png", "base_vs_corrected.png"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, layer in zip(axes.flat, LAYERS):
            part = df[df.layer == layer]
            if filename == "whitening_gain_by_layer.png":
                plot_lines(ax, part, "gain_kron")
                ax.axhline(0, color="gray", linewidth=.8)
            else:
                plot_lines(ax, part, "base_W_kron", "--")
                plot_lines(ax, part, "corrected_W_kron")
                ax.set_ylabel("W_kron")
            ax.set_title(layer)
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_lines(ax, df[df.layer == "conv1"], "test_accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(ROOT / "exp1/data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "exp1/data", train=False, download=True, transform=transform)
    diagnostic = Subset(train, range(2048))
    epochs, batch_size, seeds = 5, 256, (42, 7)
    output = ROOT / "exp3/results"
    if args.smoke:
        diagnostic = Subset(train, range(8))
        train, test = Subset(train, range(36)), Subset(test, range(8))
        epochs, batch_size, seeds = 2, 4, (42,)
        output /= "smoke"
    sigma = get_noise_multiplier(target_epsilon=1.0, target_delta=1e-5,
        sample_rate=batch_size/len(train), steps=epochs*(len(train)//batch_size), accountant="rdp")
    print(f"device={device}, noise_multiplier={sigma}, output={output}", flush=True)
    test_loader = DataLoader(test, batch_size=256)
    diagnostic_loader = DataLoader(diagnostic, batch_size=16)
    rows, feedback = [], []
    for seed in seeds:
        for method in METHODS:
            r, f = run(method, seed, train, test_loader, diagnostic_loader, epochs, batch_size, sigma, device)
            rows.extend(r)
            feedback.extend(f)
            save_results(rows, feedback, output)


if __name__ == "__main__":
    main()

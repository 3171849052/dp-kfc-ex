"""Fixed four-method Exp2. Only --smoke changes the experiment size."""
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
from clw_exp2 import Exp2Controller, interval_ends

METHODS = ("CLW-Noisy-Slow", "CLW-Noisy-Fast", "CLW-Clean-Slow", "CLW-Clean-Fast")


def run(method, seed, train, test_loader, diagnostic_loader, epochs, batch_size, sigma, device):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    sensor = "clean" if "Clean" in method else "noisy"
    frequency = 4 if "Fast" in method else 1
    controller = Exp2Controller(model, sensor)
    ends = interval_ends(len(loader), frequency)
    rows, feedback = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        interval, previous_end = 1, 0
        for step, (x, y) in enumerate(loader, 1):
            # update() replaces dictionary values, so retain the actual last-batch U.
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
            if step == ends[interval-1]:
                common = dict(method=method, seed=seed, epoch=epoch, interval=interval,
                              global_update_index=controller.update_count+1,
                              interval_steps=step-previous_end, sensor_type=sensor,
                              update_frequency=frequency)
                feedback.extend(dict(**common, **d) for d in controller.feedback_metrics())
                controller.update()
                previous_end = step
                interval += 1
        assert controller.update_count == epoch * frequency
        test_loss, accuracy = evaluate(model, test_loader, device)
        diagnostics = oracle(model, diagnostic_loader, a, g, device)
        common = dict(method=method, seed=seed, epoch=epoch, privacy_valid=sensor == "noisy",
                      train_loss=loss_sum/count, test_loss=test_loss, test_accuracy=accuracy,
                      epsilon_spent=accountant.get_epsilon(delta=1e-5),
                      clip_fraction=clipped/count, mean_clip_factor=factor_sum/count)
        rows.extend(dict(**common, **d) for d in diagnostics)
        print(f"{method} seed={seed} epoch={epoch} updates={controller.update_count} accuracy={accuracy:.4f}", flush=True)
    return rows, feedback


def plot_lines(ax, frame, metric, feedback=False):
    x = "epoch_position" if feedback else "epoch"
    for color, method in zip(("C0", "C1", "C2", "C3"), METHODS):
        part = frame[frame.method == method]
        style = "--" if "Fast" in method else "-"
        for _, seed_rows in part.groupby("seed"):
            ax.plot(seed_rows[x], seed_rows[metric], color=color, alpha=.3, linewidth=.8, linestyle=style)
        mean = part.groupby(x)[metric].mean()
        ax.plot(mean.index, mean.values, color=color, linewidth=2, linestyle=style, label=method)
    ax.set_xlabel("Epoch (interval endpoint)" if feedback else "Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=.2)


def save_results(rows, feedback, output):
    output.mkdir(parents=True, exist_ok=True)
    df, cf = pd.DataFrame(rows), pd.DataFrame(feedback)
    df.to_csv(output / "metrics.csv", index=False)
    cf.to_csv(output / "controller_metrics.csv", index=False)
    final = df[df.epoch == df.epoch.max()]
    metrics = [c for c in df.columns if c not in ("method", "seed", "epoch", "layer", "privacy_valid")]
    final.groupby(["method", "layer", "privacy_valid"])[metrics].mean().reset_index().to_csv(output / "summary.csv", index=False)
    cf["epoch_position"] = cf.epoch - 1 + cf.groupby(["method", "seed", "epoch", "layer"]).interval_steps.cumsum() / cf.groupby(["method", "seed", "epoch", "layer"]).interval_steps.transform("sum")
    panels = [
        ("whitening_by_layer.png", df, [(l, "W_kron") for l in LAYERS], False),
        ("factor_whitening_fc.png", df, [(l, f"W_{s}") for l in ("fc1", "fc2") for s in ("A", "G")], False),
        ("feedback_nsr.png", cf, [(l, f"NSR_{s}") for l in LAYERS for s in ("A", "G")], True),
        ("feedback_cosine.png", cf, [(l, f"cos_{s}") for l in LAYERS for s in ("A", "G")], True),
    ]
    for filename, frame, specs, is_feedback in panels:
        fig, axes = plt.subplots(len(specs)//2, 2, figsize=(13, 3.5*(len(specs)//2)))
        for ax, (layer, metric) in zip(axes.flat, specs):
            plot_lines(ax, frame[frame.layer == layer], metric, is_feedback)
            ax.set_title(layer)
        axes.flat[0].legend(fontsize=8)
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
    train = datasets.MNIST(ROOT / "exp2/data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "exp2/data", train=False, download=True, transform=transform)
    diagnostic = Subset(train, range(2048))
    epochs, batch_size, seeds = 5, 256, (42, 7)
    output = ROOT / "exp2/results"
    if args.smoke:
        diagnostic = Subset(train, range(8))
        train, test = Subset(train, range(24)), Subset(test, range(8))
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

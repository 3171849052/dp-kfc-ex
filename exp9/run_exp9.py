"""Stage-one Full-Fisher Equil sweep; run all 20 configurations directly.

Training/accounting follows Exp7/8. Unnoised training diagnostics are for
analysis only. Total epoch time means build + private training, as in Exp8.
DP-KFC uses raw inverse-square-root factors, without geometric normalization.
"""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp7.run_exp7 import (
    torch, F, pd, plt, DataLoader, datasets, transforms, GradSampleModule,
    RDPAccountant, get_noise_multiplier, SimpleCNN, generate_pink_noise,
    rademacher, fisher_statistics, average_covariances, kfac_preconditioner,
    precondition_per_sample_gradients, clip_and_noise_gradients,
    _compute_per_sample_norms_squared, _compute_clip_factors, evaluate, timestamp,
    DAMPING,  # Existing KFC builder fixes damping at 1e-3.
)
from exp8.run_exp8 import storage_mb

OUTPUT = ROOT / "exp9/results"
SEED = 42
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.5
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
EPSILON = 1.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
SYNTHETIC_BATCH_SIZE = 256
PROBES = 8
SCALE_CLAMP = (0.1, 10.0)
TAUS = (1e-3, 1e-2, 1e-1, 1.0)
PRECONDITIONER_BATCHES = (1, 4, 10, 16)
EQUIL = "Full-Fisher Equil"
KFC = "DP-KFC"
GAIN_FIELDS = ("gain_p10", "gain_median", "gain_p90", "gain_max")
CONFIG_FIELDS = ["method", "tau", "preconditioner_batches", "seed"]
CONFIGS = [(EQUIL, tau, n) for tau in TAUS for n in PRECONDITIONER_BATCHES]
CONFIGS += [(KFC, float("nan"), n) for n in PRECONDITIONER_BATCHES]


def pink_batches(n, device):
    for _ in range(n):
        yield (generate_pink_noise(SYNTHETIC_BATCH_SIZE, (1, 28, 28), device),
               torch.randint(0, 10, (SYNTHETIC_BATCH_SIZE,), device=device))


@torch.no_grad()
def equil_scales(statistic, tau):
    values = torch.cat([v.flatten() for v in statistic.values()])
    gamma = tau * values.median()
    scales = {p: (v + gamma).rsqrt() for p, v in statistic.items()}
    normalizer = (sum(v.log().sum() for v in scales.values()) / values.numel()).exp()
    scales = {p: (v / normalizer).clamp(*SCALE_CLAMP) for p, v in scales.items()}
    gains = torch.cat([v.flatten() for v in scales.values()])
    stats = gains.quantile(gains.new_tensor([.1, .5, .9, 1.])).tolist()
    return scales, dict(zip(GAIN_FIELDS, stats))


def build_preconditioner(model, method, tau, n, epoch, device):
    # Isolate synthetic/probe RNG from private noise; budgets share batch prefixes.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(SEED + 10000 + epoch)
        batches = pink_batches(n, device)
        if method == EQUIL:
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(SEED + 30000 + epoch)
                probes = rademacher(model.parameters(), PROBES)
            _, statistic = fisher_statistics(model, batches, device, probes)
            scales, stats = equil_scales(statistic, tau)
            return dict(scales=scales), stats
        ca, cg = average_covariances(model, batches, device)
        a, g, _ = kfac_preconditioner(model, ca, cg, "KFAC-Full-Raw")
        return dict(a=a, g=g), dict.fromkeys(GAIN_FIELDS, float("nan"))


def run(method, tau, n, train, test, sigma, device):
    torch.manual_seed(SEED)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test, batch_size=BATCH_SIZE,
                             generator=torch.Generator().manual_seed(SEED))
    accountant = RDPAccountant()
    rows, state = [], {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        state.clear()
        start = timestamp(device)
        state, gains = build_preconditioner(model, method, tau, n, epoch, device)
        build_seconds = timestamp(device) - start
        start = timestamp(device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            with torch.no_grad():
                if method == EQUIL:
                    for p, scale in state["scales"].items():
                        p.grad_sample.mul_(scale)
                else:
                    precondition_per_sample_gradients(model, state["a"], state["g"])
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
        rows.append(dict(method=method, tau=tau, preconditioner_batches=n,
            synthetic_samples=SYNTHETIC_BATCH_SIZE*n, seed=SEED, epoch=epoch,
            test_accuracy=accuracy, test_loss=test_loss, train_loss=loss_sum/count,
            epsilon_spent=accountant.get_epsilon(delta=DELTA),
            clip_fraction=clipped/count, mean_clip_factor=factor_sum/count,
            precond_build_seconds=build_seconds, private_train_seconds=train_seconds,
            total_epoch_seconds=build_seconds+train_seconds,
            preconditioner_storage_mb=storage_mb(state), **gains))
        print(f"{method} tau={tau:g} batches={n} epoch={epoch} "
              f"accuracy={accuracy:.4f} build={build_seconds:.2f}s "
              f"train={train_seconds:.2f}s", flush=True)
    model.remove_hooks()
    return rows


def save_csv(rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "metrics.csv", index=False, na_rep="NaN")
    final = frame.groupby(CONFIG_FIELDS, sort=False, dropna=False).tail(1)
    final = final.sort_values("test_accuracy", ascending=False)
    final.to_csv(OUTPUT / "summary.csv", index=False, na_rep="NaN")
    return final


def plot_results(final):
    equil = final[final.method == EQUIL]
    grid = equil.pivot(index="tau", columns="preconditioner_batches",
                       values="test_accuracy").reindex(index=TAUS, columns=PRECONDITIONER_BATCHES)
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis")
    for i in range(len(TAUS)):
        for j in range(len(PRECONDITIONER_BATCHES)):
            ax.text(j, i, f"{grid.iloc[i, j]:.4f}", ha="center", va="center",
                    color="white", bbox=dict(facecolor="black", alpha=.4, edgecolor="none"))
    ax.set(xticks=range(len(PRECONDITIONER_BATCHES)), xticklabels=PRECONDITIONER_BATCHES,
           yticks=range(len(TAUS)), yticklabels=TAUS, xlabel="Preconditioner batches",
           ylabel="Tau", title="Full-Fisher Equil: final test accuracy")
    fig.colorbar(heatmap, ax=ax, label="Final test accuracy")
    fig.tight_layout()
    fig.savefig(OUTPUT / "equil_accuracy_heatmap.png", dpi=160)
    plt.close(fig)
    panels = (
        ([("test_accuracy", "Final test accuracy")], "accuracy_vs_precond_batches.png"),
        ([("clip_fraction", "Final clip fraction"),
          ("mean_clip_factor", "Final mean clip factor")], "clipping_vs_precond_batches.png"),
        ([("precond_build_seconds", "Final epoch build seconds"),
          ("private_train_seconds", "Final epoch private train seconds"),
          ("total_epoch_seconds", "Final epoch build + train seconds")],
         "runtime_vs_precond_batches.png"),
    )
    for metrics, filename in panels:
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5), squeeze=False)
        curves = [(f"Equil tau={tau:g}", equil[equil.tau == tau]) for tau in TAUS]
        curves.append((KFC, final[final.method == KFC]))
        for ax, (metric, label) in zip(axes.flat, metrics):
            for name, part in curves:
                part = part.sort_values("preconditioner_batches")
                style = dict(color="black", linestyle="--") if name == KFC else {}
                ax.plot(part.preconditioner_batches, part[metric], marker="o", label=name, **style)
            ax.set(xlabel="Preconditioner batches", ylabel=label, xticks=PRECONDITIONER_BATCHES)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT / filename, dpi=160)
        plt.close(fig)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train = datasets.MNIST(ROOT / "exp9/data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "exp9/data", train=False, download=True, transform=transform)
    sigma = get_noise_multiplier(target_epsilon=EPSILON, target_delta=DELTA,
        sample_rate=BATCH_SIZE/len(train), steps=EPOCHS*math.ceil(len(train)/BATCH_SIZE), accountant="rdp")
    print(f"device={device}, noise_multiplier={sigma}, KFC damping={DAMPING}, "
          f"runs={len(CONFIGS)}, output={OUTPUT}", flush=True)
    rows = []
    for method, tau, n in CONFIGS:
        rows.extend(run(method, tau, n, train, test, sigma, device))
        final = save_csv(rows)
    plot_results(final)


if __name__ == "__main__":
    main()

"""Structure versus gain dispersion, using shared epoch statistics.

Run: conda run -n curve python exp9b/run_exp9b.py
The Full-Equil tau=1 trajectory supplies the exact same epoch statistic to
all configurations (including the other taus). It runs first for this reason.
Synthetic data, probes, and permutations are isolated from private-noise RNG.
Unnoised private-gradient diagnostics and train_loss are analysis only, NOT DP
releases. epsilon_spent is the inherited Exp9 training accountant, not a privacy
claim for these CSVs or the joint experiment with shared adaptive statistics.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp9.run_exp9 import (
    torch, F, pd, plt, math, DataLoader, datasets, transforms, GradSampleModule,
    RDPAccountant, get_noise_multiplier, SimpleCNN, rademacher, fisher_statistics,
    clip_and_noise_gradients, _compute_per_sample_norms_squared,
    _compute_clip_factors, evaluate, timestamp, storage_mb, pink_batches,
    equil_scales, SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM,
    WEIGHT_DECAY, EPSILON, DELTA, MAX_GRAD_NORM, SYNTHETIC_BATCH_SIZE,
    PROBES, SCALE_CLAMP,
)

OUTPUT = ROOT / "exp9b/results"
PRECONDITIONER_BATCHES = 1
TAUS = (0.3, 1.0, 3.0, 10.0)
CONFIGS = [("Full-Equil", 1.0), ("DP-SGD", float("nan"))]
CONFIGS += [("Full-Equil", tau) for tau in TAUS if tau != 1.0]
CONFIGS += [(method, 1.0) for method in
            ("Shuffled-Equil", "TensorGM-Equil", "Reverse-Equil")]


@torch.no_grad()
def applied_scales(statistic, method, tau, epoch, device):
    reference, _ = equil_scales(statistic, 1.0)
    if method == "DP-SGD":
        scales = {p: torch.ones_like(v) for p, v in reference.items()}
    elif method == "Full-Equil":
        scales, _ = equil_scales(statistic, tau)
    elif method == "Shuffled-Equil":
        scales = {}
        for i, (p, v) in enumerate(reference.items()):
            generator = torch.Generator(device=device).manual_seed(SEED + 50000 + epoch * 100 + i)
            order = torch.randperm(v.numel(), generator=generator, device=device)
            scales[p] = v.flatten()[order].view_as(v)
    elif method == "TensorGM-Equil":
        scales = {p: torch.ones_like(v) * v.log().mean().exp() for p, v in reference.items()}
        normalizer = (sum(v.log().sum() for v in scales.values()) /
                      sum(v.numel() for v in scales.values())).exp()
        scales = {p: v / normalizer for p, v in scales.items()}
    elif method == "Reverse-Equil":
        scales = {}
        for p, v in reference.items():
            flat = v.flatten()
            order = flat.argsort(stable=True)
            reverse = torch.empty_like(flat)
            reverse[order] = flat[order].flip(0)
            scales[p] = reverse.view_as(v)
    gains = torch.cat([v.flatten() for v in scales.values()])
    base = torch.cat([v.flatten() for v in reference.values()])
    quantiles = gains.quantile(gains.new_tensor([0., .1, .5, .9, 1.])).tolist()
    stats = dict(zip(("gain_min", "gain_p10", "gain_median", "gain_p90", "gain_max"), quantiles))
    # Pearson correlation of average ranks is tie-aware Spearman; constant
    # DP-SGD gains have undefined alignment, represented by NaN.
    ranks = pd.Series(gains.cpu().numpy()).rank()
    base_ranks = pd.Series(base.cpu().numpy()).rank()
    alignment = float("nan") if ranks.nunique() == 1 else ranks.corr(base_ranks)
    stats.update(gain_log_std=gains.log().std(correction=0).item(),
                 gain_p90_p10_ratio=quantiles[3] / quantiles[1],
                 lower_cap_fraction=(gains <= SCALE_CLAMP[0]).float().mean().item(),
                 upper_cap_fraction=(gains >= SCALE_CLAMP[1]).float().mean().item(),
                 alignment_spearman=alignment)
    return scales, stats


def build_preconditioner(model, method, tau, epoch, device, shared_statistics):
    params = [p for p in model.parameters() if p.requires_grad]
    if method == "Full-Equil" and tau == 1.0:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(SEED + 10000 + epoch)
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(SEED + 30000 + epoch)
                probes = rademacher(params, PROBES)
            _, statistic = fisher_statistics(model, pink_batches(PRECONDITIONER_BATCHES, device),
                                             device, probes)
        shared_statistics[epoch] = [statistic[p].cpu() for p in params]
    else:
        statistic = {p: v.to(device) for p, v in zip(params, shared_statistics[epoch])}
    scales, diagnostics = applied_scales(statistic, method, tau, epoch, device)
    return ({} if method == "DP-SGD" else dict(scales=scales)), diagnostics


@torch.no_grad()
def mean_gradient_diagnostics(params, factors):
    raw = torch.cat([p.grad_sample.flatten(1).mean(0) for p in params])
    clipped = torch.cat([(p.grad_sample.flatten(1) * factors[:, None]).mean(0) for p in params])
    return ((clipped.norm() / raw.norm()).item(),
            F.cosine_similarity(raw, clipped, dim=0).item())


def run(method, tau, train, test, sigma, device, shared_statistics):
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
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        start = timestamp(device)
        state, gains = build_preconditioner(model, method, tau, epoch, device, shared_statistics)
        build_seconds = timestamp(device) - start
        start = timestamp(device)
        loss_sum, count = 0., 0
        norm_batches, factor_batches, retentions, cosines = [], [], [], []
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            with torch.no_grad():
                for p, scale in state.get("scales", {}).items():
                    p.grad_sample.mul_(scale)
                norms_sq = _compute_per_sample_norms_squared(params, len(x), device)
                factors = _compute_clip_factors(norms_sq, MAX_GRAD_NORM)
                norm_batches.append(norms_sq.sqrt().cpu())
                factor_batches.append(factors.cpu())
                retention, cosine = mean_gradient_diagnostics(params, factors)
                retentions.append(retention)
                cosines.append(cosine)
            clip_and_noise_gradients(model, sigma, MAX_GRAD_NORM, len(x))
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=BATCH_SIZE / len(train))
            loss_sum += loss.item()
            count += len(x)
        train_seconds = timestamp(device) - start
        norms, factors = torch.cat(norm_batches), torch.cat(factor_batches)
        p50, p90, p99 = norms.quantile(torch.tensor([.5, .9, .99])).tolist()
        test_loss, accuracy = evaluate(model, test_loader, device)
        rows.append(dict(method=method, tau=tau, preconditioner_batches=PRECONDITIONER_BATCHES,
            synthetic_samples=SYNTHETIC_BATCH_SIZE * PRECONDITIONER_BATCHES, seed=SEED, epoch=epoch,
            statistic_source="Full-Equil tau=1 shared epoch statistic", analysis_only=True,
            test_accuracy=accuracy, test_loss=test_loss, train_loss=loss_sum/count,
            epsilon_spent=accountant.get_epsilon(delta=DELTA),
            preclip_norm_mean=norms.mean().item(), preclip_norm_p50=p50,
            preclip_norm_p90=p90, preclip_norm_p99=p99,
            clip_fraction=(norms > MAX_GRAD_NORM).float().mean().item(),
            mean_clip_factor=factors.mean().item(),
            severe_clip_fraction_05=(factors < .5).float().mean().item(),
            severe_clip_fraction_025=(factors < .25).float().mean().item(),
            clip_mean_gradient_norm_retention=sum(retentions)/len(retentions),
            clip_mean_gradient_cosine=sum(cosines)/len(cosines),
            precond_build_seconds=build_seconds, private_train_seconds=train_seconds,
            total_epoch_seconds=build_seconds+train_seconds,
            preconditioner_storage_mb=storage_mb(state), **gains))
        print(f"{method} tau={tau:g} epoch={epoch} accuracy={accuracy:.4f} "
              f"build={build_seconds:.2f}s train={train_seconds:.2f}s", flush=True)
    model.remove_hooks()
    return rows


def save_csv(rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "metrics.csv", index=False, na_rep="NaN")
    final = frame[frame.epoch == EPOCHS].sort_values("test_accuracy", ascending=False)
    final.to_csv(OUTPUT / "summary.csv", index=False, na_rep="NaN")
    return final


def plot_results(final):
    final = final.copy()
    final["label"] = [f"{r.method} tau={r.tau:g}" if r.method == "Full-Equil" else r.method
                      for r in final.itertuples()]
    equil = final[final.method == "Full-Equil"].sort_values("tau")
    baseline = final.loc[final.method == "DP-SGD", "test_accuracy"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(equil.tau, equil.test_accuracy, marker="o")
    ax.axhline(baseline, color="black", linestyle="--", label="DP-SGD")
    ax.set(xscale="log", xlabel="Tau", ylabel="Epoch 5 test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "accuracy_vs_tau.png", dpi=160)
    plt.close(fig)
    part = final[(final.method != "Full-Equil") | (final.tau == 1.)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(part.label, part.test_accuracy)
    ax.set(ylabel="Epoch 5 test accuracy")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ablation_accuracy.png", dpi=160)
    plt.close(fig)
    for filename, metrics in (
        ("accuracy_vs_gain_log_std.png", [("test_accuracy", "Epoch 5 test accuracy")]),
        ("clipping_vs_gain_log_std.png", [("clip_fraction", "Clip fraction (analysis only)"),
                                         ("mean_clip_factor", "Mean clip factor (analysis only)")]),
        ("gain_dispersion.png", [("gain_log_std", "Gain log std"),
                                 ("gain_p90_p10_ratio", "Gain p90 / p10")]),
    ):
        fig, axes = plt.subplots(1, len(metrics), figsize=(8*len(metrics), 5), squeeze=False)
        for ax, (metric, label) in zip(axes.flat, metrics):
            if filename == "gain_dispersion.png":
                ax.bar(final.label, final[metric])
                ax.tick_params(axis="x", labelrotation=65)
            else:
                for row in final.itertuples():
                    ax.scatter(row.gain_log_std, getattr(row, metric), label=row.label)
                ax.set_xlabel("Gain log std")
                ax.legend(fontsize=7)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
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
    print(f"device={device}, noise_multiplier={sigma}, runs={len(CONFIGS)}, output={OUTPUT}\n"
          "Private-gradient diagnostics: analysis_only=True; not DP releases.\n"
          "All configurations share statistics from the Full-Equil tau=1 trajectory.", flush=True)
    rows, shared_statistics = [], {}
    for method, tau in CONFIGS:
        rows.extend(run(method, tau, train, test, sigma, device, shared_statistics))
        final = save_csv(rows)
    plot_results(final)


if __name__ == "__main__":
    main()

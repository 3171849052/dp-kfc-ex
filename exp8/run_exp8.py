"""Cheap rotations after full Fisher equilibration on full MNIST.

Run directly for the fixed twelve runs. Unnoised training diagnostics are
analysis-only; privacy_valid has the same training-mechanism meaning as Exp6/7.
Equil retains Exp6's normalize-then-clamp definition (clamping can shift GM).
"""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp7.run_exp7 import (
    torch, F, pd, plt, DataLoader, datasets, transforms, GradSampleModule,
    RDPAccountant, get_noise_multiplier, SimpleCNN, generate_pink_noise,
    rademacher, stabilized_scales, average_covariances, kfac_preconditioner,
    precondition_per_sample_gradients, clip_and_noise_gradients,
    _compute_per_sample_norms_squared, _compute_clip_factors, evaluate, timestamp,
)

METHODS = ("DP-SGD", "Full-Fisher Equil", "Equil + G-Corr", "Equil + LR-r4",
           "Equil + LR-r8", "KFAC-Full-GM")
SEEDS = (42, 7)
EPOCHS = 5
BATCH_SIZE = 256
PRECOND_STEPS = 10
PROBES = 8
DAMPING = 1e-3
LAYERS = ("conv1", "conv2", "fc1", "fc2")
LR_CONFIG = {"Equil + LR-r4": (4, 8), "Equil + LR-r8": (8, 12)}
ROTATION_FIELDS = ("method", "seed", "epoch", "operator_geomean_before_norm",
    "rank", "sketch_size", "top_eigenvalue", "rank_r_eigenvalue",
    "captured_trace_fraction", "layer", "corr_frobenius", "corr_condition_number")


def rng_devices(device):
    return [device.index] if device.type == "cuda" else []


def synthetic_batches(seed, epoch, batch_size, device):
    # Exactly one information budget, cached on CPU and replayed verbatim.
    with torch.random.fork_rng(devices=rng_devices(device)):
        torch.manual_seed(seed + 10000 + epoch)
        return [(generate_pink_noise(batch_size, (1, 28, 28), device).cpu(),
                 torch.randint(0, 10, (batch_size,), device=device).cpu())
                for _ in range(PRECOND_STEPS)]


def fisher_product(model, batches, vectors, device):
    """F V = G.T (G V) / M across all parameter blocks; also exact diag(F)."""
    products = {p: torch.zeros_like(v) for p, v in vectors.items()}
    diagonal = {p: torch.zeros_like(p).flatten() for p in vectors}
    q = next(iter(vectors.values())).shape[1]
    count = 0
    model.train()
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum").backward()
        with torch.no_grad():
            projection = torch.zeros(len(x), q, device=device)
            for p, v in vectors.items():
                projection.add_(p.grad_sample.flatten(1) @ v)
            for p in vectors:
                block = p.grad_sample.flatten(1)
                products[p].add_(block.T @ projection)
                diagonal[p].add_(block.square().sum(0))
        count += len(x)
    model.zero_grad(set_to_none=True)
    for p in vectors:
        products[p].div_(count)
        diagonal[p].div_(count)
    return products, diagonal


@torch.no_grad()
def geomean(blocks):
    return (sum(v.double().log().sum() for v in blocks) /
            sum(v.numel() for v in blocks)).exp().item()


@torch.no_grad()
def g_correction(model, scales, cov_g):
    corrections, stats, gains = {}, [], []
    for name in LAYERS:
        module = getattr(model._module, name)
        reg = cov_g[name] + DAMPING * torch.eye(len(cov_g[name]),
            device=module.weight.device, dtype=cov_g[name].dtype)
        d = reg.diagonal().rsqrt()
        corr = d[:, None] * reg * d[None, :]
        ev, vec = torch.linalg.eigh(corr)
        cg = ((vec * ev.rsqrt()) @ vec.T).to(module.weight.dtype)
        corrections[name] = cg
        norms = cg.double().norm(dim=0)
        gains.extend((scales[module.weight].flatten(1) * norms[:, None],
                      scales[module.bias] * norms))
        stats.append(dict(layer=name, corr_frobenius=(corr - torch.eye(
            len(corr), device=corr.device, dtype=corr.dtype)).norm().item(),
            corr_condition_number=(ev[-1] / ev[0]).item()))
    before = geomean(gains)
    for cg in corrections.values():
        cg.div_(before)
    return corrections, before, stats


def low_rank(model, batches, scales, diagonal, rank, q, seed, epoch, device):
    with torch.random.fork_rng(devices=rng_devices(device)):
        torch.manual_seed(seed + 40000 + epoch)
        omega = rademacher(scales, q)
    scaled = {p: s.flatten()[:, None] * omega[p] for p, s in scales.items()}
    del omega
    products, _ = fisher_product(model, batches, scaled, device)
    del scaled
    with torch.no_grad():
        y = torch.cat([s.flatten()[:, None] * products[p] for p, s in scales.items()])
        del products
        basis, _ = torch.linalg.qr(y, mode="reduced")
        del y
        blocks = dict(zip(scales, basis.split([p.numel() for p in scales])))
        scaled = {p: s.flatten()[:, None] * blocks[p] for p, s in scales.items()}
    products, _ = fisher_product(model, batches, scaled, device)
    del scaled
    with torch.no_grad():
        small = sum(blocks[p].double().T @
            (scales[p].flatten()[:, None] * products[p]).double() for p in scales)
        ev, vec = torch.linalg.eigh((small + small.T) / 2)
        eigenvalues = ev[-rank:].flip(0).to(basis.dtype)
        u = {p: (block @ vec[:, -rank:].flip(1).to(basis.dtype)).contiguous()
             for p, block in blocks.items()}
        alpha = eigenvalues.rsqrt() - 1
        # Double precision avoids cancellation in 1 + (lambda^-1 - 1) U^2.
        gains = [scales[p].flatten().double() * (1 +
            (block.double().square() * (2 * alpha.double() + alpha.double().square()))
            .sum(1)).sqrt() for p, block in u.items()]
        before = geomean(gains)
        trace = sum((scales[p].flatten().double().square() * diagonal[p].double()).sum()
                    for p in scales)
        stats = dict(rank=rank, sketch_size=q, top_eigenvalue=eigenvalues[0].item(),
            rank_r_eigenvalue=eigenvalues[-1].item(),
            captured_trace_fraction=(eigenvalues.double().sum() / trace).item())
    return u, eigenvalues, alpha, before, stats


def build_preconditioner(model, method, seed, epoch, batch_size, device):
    batches = synthetic_batches(seed, epoch, batch_size, device)
    state = {}
    if method == "DP-SGD":
        return state, [dict(operator_geomean_before_norm=1.)]
    if method == "KFAC-Full-GM":
        ca, cg = average_covariances(model, batches, device)
        a, g, before = kfac_preconditioner(model, ca, cg, method)
        return dict(a=a, g=g), [dict(operator_geomean_before_norm=before)]
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.random.fork_rng(devices=rng_devices(device)):
        torch.manual_seed(seed + 30000 + epoch)
        probes = rademacher(params, PROBES)
    products, diagonal = fisher_product(model, batches, probes, device)
    equil = {p: v.square().mean(1).sqrt().view_as(p) for p, v in products.items()}
    scales, before = stabilized_scales(equil)
    del probes, products, equil
    state["scales"] = scales
    stats = [{}]
    if method == "Equil + G-Corr":
        ca, cg = average_covariances(model, batches, device)
        del ca
        state["g"], before, stats = g_correction(model, scales, cg)
    elif method in LR_CONFIG:
        rank, q = LR_CONFIG[method]
        u, eigenvalues, alpha, before, detail = low_rank(
            model, batches, scales, diagonal, rank, q, seed, epoch, device)
        state.update(u=u, eigenvalues=eigenvalues, alpha=alpha, normalizer=before)
        stats = [detail]
    return state, [dict(operator_geomean_before_norm=before, **s) for s in stats]


@torch.no_grad()
def apply_preconditioner(model, method, state):
    if method == "DP-SGD":
        return
    if method == "KFAC-Full-GM":
        precondition_per_sample_gradients(model, state["a"], state["g"])
        return
    for p, s in state["scales"].items():
        p.grad_sample.mul_(s)
    if method == "Equil + G-Corr":
        for name, cg in state["g"].items():
            module = getattr(model._module, name)
            for p in (module.weight, module.bias):
                block = p.grad_sample.reshape(len(p.grad_sample), p.shape[0], -1)
                p.grad_sample = torch.einsum("oi,bik->bok", cg, block).reshape_as(p.grad_sample)
    elif method in LR_CONFIG:
        projection = sum(p.grad_sample.flatten(1) @ u for p, u in state["u"].items())
        projection.mul_(state["alpha"])
        for p, u in state["u"].items():
            p.grad_sample.add_((projection @ u.T).view_as(p.grad_sample))
            p.grad_sample.div_(state["normalizer"])


def storage_mb(state):
    tensors = []
    for value in state.values():
        if isinstance(value, dict):
            tensors.extend(value.values())
        elif torch.is_tensor(value):
            tensors.append(value)
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


def start_phase(device):
    start = timestamp(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return start


def finish_phase(start, device, phase):
    seconds = timestamp(device) - start
    allocated = reserved = float("nan")
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 2**20
        reserved = torch.cuda.max_memory_reserved(device) / 2**20
    return seconds, {f"peak_allocated_mb_{phase}": allocated,
                     f"peak_reserved_mb_{phase}": reserved}


def warmup(device):
    if device.type != "cuda":
        return
    with torch.random.fork_rng(devices=rng_devices(device)):
        torch.manual_seed(12345)
        model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            x = generate_pink_noise(BATCH_SIZE, (1, 28, 28), device)
            y = torch.randint(0, 10, (BATCH_SIZE,), device=device)
            F.cross_entropy(model(x), y, reduction="sum").backward()
            clip_and_noise_gradients(model, 1., 1., BATCH_SIZE)
            optimizer.step()
        torch.cuda.synchronize(device)
        model.remove_hooks()
    del model, optimizer, x, y
    torch.cuda.empty_cache()


def run(method, seed, train, test, epochs, batch_size, sigma, device):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=.1, momentum=.9)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=batch_size,
                             generator=torch.Generator().manual_seed(seed))
    accountant = RDPAccountant()
    rows, compute, rotation = [], [], []
    state = {}
    for epoch in range(1, epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        state.clear()
        start = start_phase(device)
        state, stats = build_preconditioner(model, method, seed, epoch, batch_size, device)
        build_seconds, build_memory = finish_phase(start, device, "build")
        common = dict(method=method, seed=seed, epoch=epoch)
        rotation.extend(dict(**common, **s) for s in stats)
        start = start_phase(device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            apply_preconditioner(model, method, state)
            norms_sq = _compute_per_sample_norms_squared(params, len(x), device)
            factors = _compute_clip_factors(norms_sq, 1.)
            clipped += (norms_sq.sqrt() > 1.).sum().item()
            factor_sum += factors.sum().item()
            # Precondition -> global per-example clipping -> isotropic noise -> SGD.
            clip_and_noise_gradients(model, sigma, 1., len(x))
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=batch_size / len(train))
            loss_sum += loss.item()
            count += len(x)
        train_seconds, train_memory = finish_phase(start, device, "train")
        total = build_seconds + train_seconds
        compute.append(dict(**common, precond_build_seconds=build_seconds,
            private_train_seconds=train_seconds, total_epoch_seconds=total,
            train_examples_per_second=len(train)/train_seconds,
            end_to_end_examples_per_second=len(train)/total,
            **build_memory, **train_memory, preconditioner_storage_mb=storage_mb(state)))
        test_loss, accuracy = evaluate(model, test_loader, device)
        rows.append(dict(**common, privacy_valid=True, train_loss=loss_sum/count,
            test_loss=test_loss, test_accuracy=accuracy,
            epsilon_spent=accountant.get_epsilon(delta=1e-5), clip_fraction=clipped/count,
            mean_clip_factor=factor_sum/count))
        print(f"{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} "
              f"build={build_seconds:.2f}s train={train_seconds:.2f}s", flush=True)
    model.remove_hooks()
    return rows, compute, rotation


def save_results(rows, compute, rotation, output):
    output.mkdir(parents=True, exist_ok=True)
    frame, costs = pd.DataFrame(rows), pd.DataFrame(compute)
    frame.to_csv(output / "metrics.csv", index=False)
    costs.to_csv(output / "compute.csv", index=False)
    pd.DataFrame(rotation, columns=ROTATION_FIELDS).to_csv(output / "rotation_stats.csv", index=False)
    final = frame.groupby(["method", "seed"], sort=False).tail(1)
    summary = final.groupby("method", sort=False).agg(
        mean_accuracy=("test_accuracy", "mean"), std_accuracy=("test_accuracy", "std"),
        mean_clipping=("clip_fraction", "mean"), mean_clip_factor=("mean_clip_factor", "mean"))
    fields = ("precond_build_seconds", "private_train_seconds", "total_epoch_seconds",
              "end_to_end_examples_per_second", "peak_allocated_mb_build",
              "peak_allocated_mb_train", "preconditioner_storage_mb")
    summary = summary.join(costs.groupby("method", sort=False)[list(fields)].mean().add_prefix("mean_"))
    summary.to_csv(output / "summary.csv")
    for metric, filename in (("test_accuracy", "accuracy.png"), ("clip_fraction", "clipping.png")):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, method in enumerate(METHODS):
            part = frame[frame.method == method]
            if part.empty:
                continue
            for _, seed_rows in part.groupby("seed"):
                ax.plot(seed_rows.epoch, seed_rows[metric], color=f"C{i}", alpha=.3)
            mean = part.groupby("epoch")[metric].mean()
            ax.plot(mean.index, mean.values, marker="o", color=f"C{i}", label=method)
        ax.set(xlabel="Epoch", ylabel=metric, xticks=sorted(frame.epoch.unique()))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary.index, summary.mean_precond_build_seconds, label="Build")
    ax.bar(summary.index, summary.mean_private_train_seconds,
           bottom=summary.mean_precond_build_seconds, label="Private train")
    ax.set_ylabel("Mean epoch seconds")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "runtime.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    memory = summary[["mean_peak_allocated_mb_build", "mean_peak_allocated_mb_train"]].dropna()
    if memory.empty:
        axes[0].set_axis_off()
    else:
        memory.rename(columns={"mean_peak_allocated_mb_build": "Build",
                               "mean_peak_allocated_mb_train": "Train"}).plot.bar(ax=axes[0], rot=25)
        axes[0].set_ylabel("Mean peak allocated VRAM (MB)")
    summary.mean_preconditioner_storage_mb.plot.bar(ax=axes[1], rot=25)
    axes[1].set_ylabel("Preconditioner storage (MB)")
    fig.tight_layout()
    fig.savefig(output / "memory.png", dpi=160)
    plt.close(fig)
    for field, label, filename in (
        ("mean_total_epoch_seconds", "Mean epoch seconds", "accuracy_vs_runtime.png"),
        ("mean_peak_allocated_mb_train", "Mean peak training VRAM (MB)", "accuracy_vs_memory.png")):
        fig, ax = plt.subplots(figsize=(10, 6))
        part = summary.dropna(subset=[field, "mean_accuracy"])
        if part.empty:
            ax.set_axis_off()
        else:
            for method, row in part.iterrows():
                ax.scatter(row[field], row.mean_accuracy, color=f"C{METHODS.index(method)}")
                ax.annotate(method, (row[field], row.mean_accuracy), xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set(xlabel=label, ylabel="Final mean accuracy")
            ax.margins(.2)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
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
        sample_rate=BATCH_SIZE/len(train), steps=EPOCHS*math.ceil(len(train)/BATCH_SIZE), accountant="rdp")
    output = ROOT / "exp8/results"
    warmup(device)
    print(f"device={device}, noise_multiplier={sigma}, output={output}", flush=True)
    rows, compute, rotation = [], [], []
    for seed in SEEDS:
        for method in METHODS:
            rr, rc, rs = run(method, seed, train, test, EPOCHS, BATCH_SIZE, sigma, device)
            rows.extend(rr)
            compute.extend(rc)
            rotation.extend(rs)
            save_results(rows, compute, rotation, output)


if __name__ == "__main__":
    main()

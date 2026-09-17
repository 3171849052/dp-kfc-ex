"""MNIST: non-normalized Full/A-only KFC, explicit clipping versus GD+GN+BK.

Each epoch uses one 256-example geometry batch. BK keeps raw factors and uses
fixed tiled Gram norms; it never constructs sample gradient matrices. The default
protocol runs BK at every epsilon and adds Explicit at epsilon=3. Only epsilon=3
is profiled. Timings synchronize CUDA, including the component timings
(profiling adds overhead).
Cache/workspace byte counts describe tensor storage; CUDA peaks include the
whole live allocator footprint. RDP accounting follows the original experiment's
sample-rate convention with shuffled, fixed-size, drop-last private batches.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from dp_kfac.covariance import compute_covariances, compute_inverse_sqrt
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.models import SimpleCNN
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.precondition import precondition_per_sample_gradients
from dp_kfac.trainer import evaluate, set_seed
from exp21.handlers import Record, nbytes
from exp22.methods import Clipper, _retained_bytes
from scripts.paper.exp_imdb_logreg_a import compute_a_covariance, compute_a_operator

# Training configuration. Keep these local so this experiment can be tuned
# without modifying the original exp_cnn_mnist.py script.
EPSILONS = [0.5, 3.0, 8.0]
PROFILE_EPSILON = 3.0
SEEDS = [42, 7, 91, 23, 58]
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

# KFC configuration.
A_POWER = 0.4
DAMPING = 1e-3
GHOST_TILE = 64
CONDITIONS = [("base", "none")] + [
    (geometry, source)
    for geometry in ("full", "a_only")
    for source in ("match", "mismatch", "pink")
]
TIMINGS = (
    "differentiation_seconds", "factor_precondition_seconds",
    "sample_gradient_precondition_seconds", "norm_clip_seconds",
    "aggregate_seconds", "aggregate_precondition_seconds", "noise_optimizer_seconds",
)
COUNTERS = (
    "sample_gradient_matrices_preconditioned", "sample_gradient_elements_preconditioned",
    "factor_elements_preconditioned_for_norm", "aggregate_matrices_preconditioned",
    "aggregate_elements_preconditioned",
)


def timestamp(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@contextmanager
def timed(stats, key, device, enabled=True):
    if not enabled:
        yield
        return
    start = timestamp(device)
    yield
    stats[key] += timestamp(device) - start


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        # Do not charge a later BK run for an earlier explicit run's cached blocks.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peaks(device, prefix):
    return {
        f"{prefix}_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        f"{prefix}_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }


def affine_layers(model):
    layers = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
    assert {id(p) for m in layers.values() for p in m.parameters()} == {
        id(p) for p in model.parameters()
    }
    assert all(m.bias is not None for m in layers.values())
    return layers


def capture(layers, activations, backprops, anchors=None):
    """Output hooks capture d(sum loss)/d(output), as in exp21 Ghost Differentiation."""
    def hook(name):
        def forward(module, inputs, output):
            assert name not in activations, "SimpleCNN must call each layer once"
            activations[name] = inputs[0].detach()
            if anchors is not None and not output.requires_grad:
                output.requires_grad_(True)
                anchors.append(output)
            if output.requires_grad:
                output.register_hook(lambda b: backprops.__setitem__(name, b.detach()))
        return forward
    return [module.register_forward_hook(hook(name)) for name, module in layers.items()]


def build_geometry(model, layers, geometry, source, public, device, seed, epoch):
    if geometry == "base":
        return {}, {}
    # Geometry RNG is independent of private shuffling and DP Gaussian noise.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 10000 + epoch)
        if source == "pink":
            x = generate_pink_noise(BATCH_SIZE, (1, 28, 28), device)
            y = torch.randint(10, (BATCH_SIZE,), device=device) if geometry == "full" else None
        else:
            batch = next(iter(public[source]))
            x = batch[0].to(device)
            y = batch[1].reshape(-1).long().to(device) if geometry == "full" else None
        assert x.shape == (BATCH_SIZE, 1, 28, 28)
        activations, backprops = {}, {}
        handles = capture(layers, activations, backprops)
        if isinstance(model, GradSampleModule):
            model.disable_hooks()
        if geometry == "full":
            F.cross_entropy(model(x), y, reduction="sum").backward()
        else:
            with torch.no_grad():
                model(x)
        for handle in handles:
            handle.remove()
        if isinstance(model, GradSampleModule):
            model.enable_hooks()
        model.zero_grad(set_to_none=True)
        assert set(activations) == set(layers)
        if geometry == "full":
            assert set(backprops) == set(layers)
            covariance = compute_covariances(model, activations, backprops)
            return compute_inverse_sqrt(covariance, damping=DAMPING)
        operators = {}
        for name, module in layers.items():
            a = activations[name]
            if isinstance(module, nn.Conv2d):
                a = F.unfold(a, module.kernel_size, padding=module.padding,
                             stride=module.stride).transpose(1, 2).reshape(-1, module.weight[0].numel())
            operators[name] = compute_a_operator(compute_a_covariance(a), A_POWER, DAMPING)
        return operators, {}


@torch.no_grad()
def explicit_clip(model, layers, ua, ug, stats, device, batch_size, profile):
    def sample_bytes():
        return _retained_bytes([p.grad_sample for p in model.parameters()])

    if profile:
        stats["grad_sample_peak_bytes"] = max(stats["grad_sample_peak_bytes"], sample_bytes())
    if ua:
        with timed(stats, "sample_gradient_precondition_seconds", device, enabled=profile):
            if ug:
                precondition_per_sample_gradients(model, ua, ug)
            else:
                for name, module in layers.items():
                    g = torch.cat((module.weight.grad_sample.flatten(2),
                                   module.bias.grad_sample.unsqueeze(-1)), dim=-1)
                    g = g @ ua[name]
                    module.weight.grad_sample = g[..., :-1].reshape_as(module.weight.grad_sample)
                    module.bias.grad_sample = g[..., -1]
            if profile:
                for module in layers.values():
                    stats["sample_gradient_matrices_preconditioned"] += batch_size
                    stats["sample_gradient_elements_preconditioned"] += batch_size * (
                        module.weight.numel() + module.bias.numel()
                    )
    if profile:
        stats["grad_sample_peak_bytes"] = max(stats["grad_sample_peak_bytes"], sample_bytes())
    with timed(stats, "norm_clip_seconds", device, enabled=profile):
        sq = sum(torch.linalg.vector_norm(p.grad_sample, dim=tuple(range(1, p.grad_sample.ndim))).square()
                 for p in model.parameters())
        norms = sq.clamp_min(0).sqrt()
        factors = (MAX_GRAD_NORM / (norms + 1e-6)).clamp(max=1)
    with timed(stats, "aggregate_seconds", device, enabled=profile):
        for p in model.parameters():
            p.grad = torch.einsum("b,b...->...", factors, p.grad_sample)
            p.grad_sample = None
    return norms, factors


def bk_differentiate(model, layers, x, y, stats, profile):
    """Disable parameter differentiation; only differentiate through output anchors."""
    activations, backprops, anchors = {}, {}, []
    for p in model.parameters():
        assert not hasattr(p, "grad_sample")
        p.requires_grad_(False)
    handles = capture(layers, activations, backprops, anchors)
    losses = F.cross_entropy(model(x.detach()), y, reduction="none")
    torch.autograd.grad(losses.sum(), anchors)
    for handle in handles:
        handle.remove()
    for p in model.parameters():
        p.requires_grad_(True)
        assert p.grad is None
    assert set(activations) == set(backprops) == set(layers)
    trainable = {id(p) for p in model.parameters()}
    records = []
    if profile:
        cache_peak = _retained_bytes(list(activations.values()) + list(backprops.values()))
    for name, module in layers.items():
        # operator=None is essential: the persistent record contains RAW Z/B.
        record = Record(name, module, activations.pop(name), backprops.pop(name), None, trainable)
        records.append(record)
        if profile:
            cache_peak = max(cache_peak, _retained_bytes(
                list(activations.values()) + list(backprops.values()) +
                [t for r in records for t in (r.z, r.b)]
            ))
    if profile:
        stats["bk_cache_peak_bytes"] = max(stats["bk_cache_peak_bytes"], cache_peak)
    return losses.detach().sum(), records


@torch.no_grad()
def bk_clip(records, ua, ug, stats, device, profile):
    sq = records[0].b.new_zeros(len(records[0].b))
    for record in records:
        z, b = record.z, record.b
        transformed_bytes = 0
        if ua:
            with timed(stats, "factor_precondition_seconds", device, enabled=profile):
                z = z @ ua[record.name]
                if profile:
                    stats["factor_elements_preconditioned_for_norm"] += z.numel()
                    transformed_bytes += nbytes(z)
                if ug:
                    b = b @ ug[record.name].T
                    if profile:
                        stats["factor_elements_preconditioned_for_norm"] += b.numel()
                        transformed_bytes += nbytes(b)
        with timed(stats, "norm_clip_seconds", device, enabled=profile):
            # Tiled Gram identity from exp22: no [batch, output, input] gradient.
            layer_sq, workspace = Clipper._linear_norm_squared(z, b, tile=GHOST_TILE)
            sq.add_(layer_sq)
            if profile:
                stats["bk_temporary_peak_bytes"] = max(
                    stats["bk_temporary_peak_bytes"], transformed_bytes + workspace + nbytes(sq) + nbytes(layer_sq)
                )
            del z, b, layer_sq  # transformed factors never reach aggregation
    with timed(stats, "norm_clip_seconds", device, enabled=profile):
        norms = sq.clamp_min(0).sqrt()
        factors = (MAX_GRAD_NORM / (norms + 1e-6)).clamp(max=1)
    for record in records:
        with timed(stats, "aggregate_seconds", device, enabled=profile):
            weighted_b = torch.empty(record.b.shape, device=device, dtype=record.b.dtype)
            torch.mul(record.b, factors[:, None, None], out=weighted_b)
            h = weighted_b.reshape(-1, weighted_b.shape[-1]).T @ record.z.reshape(-1, record.z.shape[-1])
            if profile:
                stats["bk_temporary_peak_bytes"] = max(
                    stats["bk_temporary_peak_bytes"], nbytes(weighted_b) + nbytes(h)
                )
            del weighted_b
        if ua:
            with timed(stats, "aggregate_precondition_seconds", device, enabled=profile):
                if ug:
                    h = ug[record.name] @ h
                h = h @ ua[record.name]
                if profile:
                    stats["aggregate_matrices_preconditioned"] += 1
                    stats["aggregate_elements_preconditioned"] += h.numel()
                    stats["bk_temporary_peak_bytes"] = max(stats["bk_temporary_peak_bytes"], 2 * nbytes(h))
        with timed(stats, "aggregate_seconds", device, enabled=profile):
            for p, grad in record.split(h).items():
                p.grad = grad
        del h
    records.clear()
    return norms, factors


def run_one(train, test, public, geometry, source, engine, epsilon, seed, epochs, device, output_dir, profile: bool):
    set_seed(seed)
    train.generator = torch.Generator().manual_seed(seed)
    train.sampler.generator = train.generator
    raw_model = SimpleCNN(in_channels=1, num_classes=10).to(device)
    layers = affine_layers(raw_model)
    model = GradSampleModule(raw_model, loss_reduction="sum") if engine == "explicit" else raw_model
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sample_rate = BATCH_SIZE / len(train.dataset)
    sigma = get_noise_multiplier(target_epsilon=epsilon, target_delta=DELTA,
                                 sample_rate=sample_rate, steps=epochs * len(train), accountant="rdp")
    accountant = RDPAccountant()
    noise_rng = torch.Generator(device=device).manual_seed(seed + 20000)
    rows = []
    ua, ug = {}, {}
    method = {"base": "DP-Adam", "full": "DP-KFC", "a_only": "DP-KFC-A"}[geometry]
    for epoch in range(1, epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        ua.clear()
        ug.clear()
        stats = {key: 0.0 if profile else float("nan") for key in TIMINGS}
        stats.update({key: 0 if profile else float("nan") for key in (
            *COUNTERS, "grad_sample_peak_bytes", "bk_cache_peak_bytes", "bk_temporary_peak_bytes",
        )})
        if profile:
            reset_peak(device)
        started = timestamp(device) if profile else float("nan")
        ua, ug = build_geometry(model, layers, geometry, source, public, device, seed, epoch)
        stats["geometry_build_seconds"] = timestamp(device) - started if profile else float("nan")
        stats.update(peaks(device, "geometry") if profile else {
            "geometry_peak_allocated_bytes": float("nan"),
            "geometry_peak_reserved_bytes": float("nan"),
        })
        if profile:
            reset_peak(device)
        started = timestamp(device) if profile else float("nan")
        all_norms, all_factors = [], []
        total_loss, examples = 0.0, 0
        steps = optimizer_steps = noise_events = accountant_steps = 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            assert len(x) == BATCH_SIZE
            model.zero_grad(set_to_none=True)
            with timed(stats, "differentiation_seconds", device, enabled=profile):
                if engine == "explicit":
                    loss = F.cross_entropy(model(x), y, reduction="sum")
                    loss.backward()
                    loss = loss.detach()
                else:
                    loss, records = bk_differentiate(model, layers, x, y, stats, profile)
            if engine == "explicit":
                norms, factors = explicit_clip(model, layers, ua, ug, stats, device, len(x), profile)
            else:
                norms, factors = bk_clip(records, ua, ug, stats, device, profile)
                assert all(not hasattr(p, "grad_sample") for p in model.parameters())
            with timed(stats, "noise_optimizer_seconds", device, enabled=profile):
                for p in model.parameters():
                    noise = torch.randn(p.shape, device=device, dtype=p.dtype, generator=noise_rng)
                    p.grad.add_(noise, alpha=sigma * MAX_GRAD_NORM).div_(len(x))
                noise_events += 1  # one full-model Gaussian mechanism per logical batch
                optimizer.step()
                optimizer_steps += 1
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                accountant_steps += 1
            steps += 1
            total_loss += loss.item()
            examples += len(x)
            if profile:
                all_norms.append(norms.cpu())
                all_factors.append(factors.cpu())
        stats["private_train_seconds"] = timestamp(device) - started if profile else float("nan")
        stats.update(peaks(device, "private") if profile else {
            "private_peak_allocated_bytes": float("nan"),
            "private_peak_reserved_bytes": float("nan"),
        })
        stats["algorithm_seconds"] = stats["geometry_build_seconds"] + stats["private_train_seconds"]
        started = timestamp(device) if profile else float("nan")
        accuracy, test_loss = evaluate(model, test, device)
        stats["evaluation_seconds"] = timestamp(device) - started if profile else float("nan")
        if profile:
            norms, factors = torch.cat(all_norms), torch.cat(all_factors)
        row = dict(method=method, geometry=geometry, source=source, engine=engine, profiled=profile,
                   epsilon_target=epsilon, epsilon_spent=accountant.get_epsilon(DELTA), delta=DELTA,
                   noise_multiplier=sigma, seed=seed, epoch=epoch, train_loss=total_loss / examples,
                   test_loss=test_loss, accuracy=accuracy, **stats,
                   clip_fraction=(factors < 1).float().mean().item() if profile else float("nan"),
                   mean_clip_factor=factors.mean().item() if profile else float("nan"),
                   norm_p50=norms.quantile(.5).item() if profile else float("nan"),
                   norm_p90=norms.quantile(.9).item() if profile else float("nan"),
                   norm_p99=norms.quantile(.99).item() if profile else float("nan"),
                   norm_max=norms.max().item() if profile else float("nan"),
                   logical_steps=steps, optimizer_steps=optimizer_steps, noise_events=noise_events,
                   accountant_steps=accountant_steps)
        assert steps == optimizer_steps == noise_events == accountant_steps
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / f"{geometry}_{source}_{engine}_eps{epsilon:g}_seed{seed}.csv", index=False)
        print(f"{geometry}/{source}/{engine} eps={epsilon:g} seed={seed} epoch={epoch} "
              f"accuracy={accuracy:.4f} algorithm={stats['algorithm_seconds']:.2f}s", flush=True)
    if engine == "explicit":
        model.remove_hooks()
    summary = rows[-1].copy()
    for key in ("private_train_seconds", "algorithm_seconds", "geometry_build_seconds", "evaluation_seconds", *TIMINGS,
                "logical_steps", "optimizer_steps", "noise_events", "accountant_steps", *COUNTERS):
        summary[key] = sum(row[key] for row in rows)
    for key in ("private_peak_allocated_bytes", "private_peak_reserved_bytes", "geometry_peak_allocated_bytes",
                "geometry_peak_reserved_bytes", "grad_sample_peak_bytes", "bk_cache_peak_bytes", "bk_temporary_peak_bytes"):
        summary[key] = max(row[key] for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="All 14 conditions, seed=42, epsilon=3, one full epoch with profiling")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--engine", choices=("auto", "explicit", "bk", "all"), default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("results/cnn_mnist_a"))
    args = parser.parse_args()
    if args.fast:
        seeds, epsilons, epochs = [42], [PROFILE_EPSILON], 1
    else:
        seeds = [args.seed] if args.seed is not None else SEEDS
        epsilons = [args.epsilon] if args.epsilon is not None else EPSILONS
        epochs = EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seeds={seeds}; epsilons={epsilons}; epochs={epochs}; engine={'all' if args.fast else args.engine}", flush=True)
    train, test, _ = get_mnist_loaders(BATCH_SIZE)
    match, _, _ = get_fashionmnist_loaders(BATCH_SIZE)
    from medmnist import PathMNIST
    mismatch = DataLoader(PathMNIST(root="./data", split="train", download=True, transform=transforms.Compose([
        transforms.Resize((28, 28)), transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    public = {"match": match, "mismatch": mismatch}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for epsilon in epsilons:
        profile = epsilon == PROFILE_EPSILON
        if args.fast or args.engine == "all" or (args.engine == "auto" and profile):
            engines = ("bk", "explicit")
        else:
            engines = ("bk",) if args.engine == "auto" else (args.engine,)
        for seed in seeds:
            for geometry, source in CONDITIONS:
                for engine in engines:
                    summaries.append(run_one(train, test, public, geometry, source, engine,
                                             epsilon, seed, epochs, device, args.output_dir, profile=profile))
                    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()

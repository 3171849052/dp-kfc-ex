"""Run one fresh Exp22 method/seed subprocess."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
for key, relative in {
    "XDG_CACHE_HOME": ".cache",
    "MPLCONFIGDIR": ".cache/matplotlib",
    "CUDA_CACHE_PATH": ".cache/cuda",
    "TMPDIR": ".cache/tmp",
}.items():
    path = HERE / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import pandas as pd
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from exp22 import config as cfg
from exp22.analyze import accuracy_auc
from exp22.geometry import build_from_cache, synthetic_cache
from exp22.methods import Clipper
from exp22.model import initialize


def load_data(download: bool = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = HERE / "data"
    return (
        datasets.CIFAR10(root=root, train=True, download=download, transform=transform),
        datasets.CIFAR10(root=root, train=False, download=download, transform=transform),
    )


def private_loader(dataset, seed: int):
    return DataLoader(
        dataset,
        batch_size=cfg.LOGICAL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return total_loss / count, correct / count


def _sigma(steps: int, sample_count: int, smoke: bool) -> float:
    if smoke:
        return 0.0
    return get_noise_multiplier(
        target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA,
        sample_rate=cfg.LOGICAL_BATCH_SIZE / sample_count,
        steps=steps,
        accountant="rdp",
    )


def _finite_model(model):
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def run(method: str, seed: int, smoke: bool, output: Path, smoke_batches: int = cfg.SMOKE_LOGICAL_BATCHES):
    if method not in cfg.METHODS:
        raise ValueError(method)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test = load_data(download=True)
    if smoke:
        train = Subset(train, range(min(len(train), smoke_batches * cfg.LOGICAL_BATCH_SIZE)))
        test = Subset(test, range(min(len(test), cfg.SMOKE_TEST_SAMPLES)))
    epochs = cfg.SMOKE_EPOCHS if smoke else cfg.EPOCHS
    sample_count = len(train)
    logical_steps = sample_count // cfg.LOGICAL_BATCH_SIZE
    total_steps = epochs * logical_steps
    sigma = _sigma(total_steps, cfg.TRAIN_SAMPLES if not smoke else sample_count, smoke)

    model = initialize(seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS,
        eps=cfg.ADAM_EPS, weight_decay=cfg.WEIGHT_DECAY,
    )
    loader = private_loader(train, seed)
    test_loader = DataLoader(test, batch_size=cfg.LOGICAL_BATCH_SIZE, shuffle=False, num_workers=0)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 40000)
    accountant = RDPAccountant()
    run_dir = output / "runs" / f"{method}_{seed}"
    if not smoke and (run_dir / "metrics.csv").exists():
        raise FileExistsError(f"formal run already exists; choose a fresh output: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accuracies = []
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if method == "dp_sgd":
            cache = []
            operator, builder = None, {
                "builder_forward_calls": 0, "builder_vjp_calls": 0,
                "builder_logical_batches": 0,
                "builder_reverse_vectors": 0, "builder_samples": 0,
                "preconditioned_layers": [], "operator_state_bytes": 0,
            }
            builder_seconds = 0.0
        else:
            build_start = time.perf_counter()
            cache = synthetic_cache(seed, epoch, device)
            operator, builder = build_from_cache(model, method, cache, seed, epoch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            builder_seconds = time.perf_counter() - build_start
        del cache

        clipper = Clipper(model, operator, method="bk", max_grad_norm=cfg.MAX_GRAD_NORM)
        private_start = time.perf_counter()
        loss_total = 0.0
        all_norms, all_factors = [], []
        layer_values, group_values = {}, {}
        batches, max_cache, max_temp, max_fallback_temp, backward_calls = 0, 0, 0, 0, 0
        cache_empty = True
        layer_strategies = {}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, norms, factors, layer_sq, stats = clipper.aggregate_logical(
                x, y, cfg.PHYSICAL_BATCH_SIZE
            )
            clipper.step(optimizer, sigma, len(x), noise_generator)
            accountant.step(
                noise_multiplier=sigma,
                sample_rate=cfg.LOGICAL_BATCH_SIZE / (cfg.TRAIN_SAMPLES if not smoke else sample_count),
            )
            loss_total += loss.item()
            all_norms.append(norms.cpu())
            all_factors.append(factors.cpu())
            for name, value in layer_sq.items():
                layer_values.setdefault(name, []).append(value.cpu())
            for name, value in stats["layer_group_sq"].items():
                group_values.setdefault(name, []).append(value.cpu())
            max_temp = max(max_temp, stats["temporary_per_sample_grad_bytes"])
            max_cache = max(max_cache, stats["bk_cache_bytes"])
            max_fallback_temp = max(max_fallback_temp, stats["fallback_temporary_grad_bytes"])
            cache_empty = cache_empty and stats["cache_empty_after_step"]
            layer_strategies.update(stats["layer_strategies"])
            backward_calls += stats["backward_calls"]
            batches += 1
            del x, y, loss, norms, factors, layer_sq
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        private_seconds = time.perf_counter() - private_start
        memory = {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        }
        clipper.remove()
        norms, factors = torch.cat(all_norms), torch.cat(all_factors)
        test_loss, accuracy = evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        group_contribution = {
            name: torch.cat(values).mean().sqrt().item() for name, values in group_values.items()
        }
        row = {
            "method": method, "seed": seed, "epoch": epoch,
            "train_loss": loss_total / (batches * cfg.LOGICAL_BATCH_SIZE),
            "test_loss": test_loss, "test_accuracy": accuracy,
            "best_accuracy": max(accuracies),
            "accuracy_auc": accuracy_auc(range(1, len(accuracies) + 1), accuracies),
            "noise_multiplier": sigma,
            "epsilon": accountant.get_epsilon(delta=cfg.DELTA),
            "accountant_steps": sum(v[2] for v in accountant.history),
            "logical_steps": batches, "samples": batches * cfg.LOGICAL_BATCH_SIZE,
            "clip_fraction": (norms > cfg.MAX_GRAD_NORM).float().mean().item(),
            "mean_clip_factor": factors.mean().item(),
            "transformed_norm_p50": torch.quantile(norms, 0.50).item(),
            "transformed_norm_p90": torch.quantile(norms, 0.90).item(),
            "transformed_norm_p99": torch.quantile(norms, 0.99).item(),
            "transformed_norm_max": norms.max().item(),
            "builder_seconds": builder_seconds, "private_train_seconds": private_seconds,
            "algorithm_seconds": builder_seconds + private_seconds,
            "logical_steps_per_second": batches / private_seconds,
            "samples_per_second": batches * cfg.LOGICAL_BATCH_SIZE / private_seconds,
            "preconditioner_build_seconds": builder_seconds,
            "logical_batch_size": cfg.LOGICAL_BATCH_SIZE,
            "physical_batch_size": cfg.PHYSICAL_BATCH_SIZE,
            "accumulation_steps": cfg.ACCUMULATION_STEPS,
            "backward_calls": backward_calls,
            "optimizer_steps": clipper.optimizer_steps, "noise_events": clipper.noise_events,
            "bk_cache_bytes": max_cache, "temporary_per_sample_grad_bytes": max_temp,
            "fallback_temporary_grad_bytes": max_fallback_temp,
            "cache_empty_after_step": cache_empty,
            "layer_strategies": layer_strategies,
            "parameters_finite": _finite_model(model),
            "parameters_updated": any(
                not torch.equal(initial_parameters[name], parameter.detach())
                for name, parameter in model.named_parameters()
            ),
            "preconditioned_layers": clipper.preconditioned_layers,
            "identity_geometry_layers": clipper.identity_geometry_layers,
            **memory, **builder,
        }
        for name, value in group_contribution.items():
            row[f"group_norm_{name}"] = value
        for name, values in layer_values.items():
            row[f"layer_norm_{name.replace('.', '_')}"] = torch.cat(values).mean().sqrt().item()
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        print(f"{method} seed={seed} epoch={epoch}: accuracy={accuracy:.4f} loss={row['train_loss']:.5f} logical_steps={batches}", flush=True)

    configuration = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper() and name != "ROOT"}
    configuration.update({
        "method": method, "seed": seed, "smoke": smoke, "epochs": epochs,
        "total_steps": total_steps, "noise_multiplier": sigma,
        "accounting": "RDP fixed logical-batch convention",
        "rng": {"initialization": "seed", "private_shuffle": "seed",
                "synthetic_x": "seed+10000+epoch", "synthetic_y": "seed+20000+epoch",
                "dp_noise": "seed+40000"},
    })
    (run_dir / "config.json").write_text(json.dumps(configuration, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=cfg.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=cfg.SEEDS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=cfg.SMOKE_LOGICAL_BATCHES)
    parser.add_argument("--output", type=Path, default=cfg.RESULTS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.method, args.seed, args.smoke, args.output.resolve(), args.smoke_batches)


if __name__ == "__main__":
    main()

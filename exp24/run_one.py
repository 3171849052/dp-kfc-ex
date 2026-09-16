"""Run one Exp24 method/seed pair."""

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
    "HF_HOME": ".cache/huggingface",
    "TORCH_HOME": ".cache/torch",
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

from exp24 import config as cfg
from exp24.analyze import accuracy_auc
from exp24.geometry import build_from_batches, synthetic_stream
from exp24.methods import Clipper, make_optimizer
from exp24.model import initialize


def data_transform():
    return transforms.Compose([
        transforms.Resize(
            (cfg.IMG_SIZE, cfg.IMG_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_data(download: bool = True):
    transform = data_transform()
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


def _sigma(steps: int, sample_count: int, smoke: bool = False) -> float:
    """Calibrate sigma solely with Opacus RDP for this exact schedule."""

    return get_noise_multiplier(
        target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA,
        sample_rate=cfg.LOGICAL_BATCH_SIZE / sample_count,
        steps=steps,
        accountant="rdp",
    )


def _finite_model(model):
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _frozen_snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _head_snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _frozen_unchanged(model, snapshot):
    return all(torch.equal(snapshot[name], parameter.detach()) for name, parameter in model.named_parameters() if name in snapshot)


def _frozen_have_no_grad(model):
    return all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


def _builder_zero_stats():
    return {
        "builder_forward_calls": 0,
        "builder_logical_batches": 0,
        "builder_vjp_calls": 0,
        "builder_reverse_vectors": 0,
        "builder_samples": 0,
        "preconditioned_layers": [],
        "operator_state_bytes": 0,
    }


def run(
    method: str,
    seed: int,
    smoke: bool,
    output: Path,
    smoke_batches: int = cfg.SMOKE_LOGICAL_BATCHES,
):
    if method not in cfg.METHODS:
        raise ValueError(method)
    if seed not in cfg.SEEDS:
        raise ValueError(seed)
    output = output.resolve()
    run_dir = output / "runs" / f"{method}_{seed}"
    if not smoke and (run_dir / "metrics.csv").exists():
        raise FileExistsError(f"formal run already exists; choose a fresh output: {run_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize(seed, device)
    train, test = load_data(download=True)
    if smoke:
        train = Subset(train, range(smoke_batches * cfg.LOGICAL_BATCH_SIZE))
        test = Subset(test, range(cfg.SMOKE_TEST_SAMPLES))
    epochs = cfg.SMOKE_EPOCHS if smoke else cfg.EPOCHS
    sample_count = len(train)
    logical_steps = sample_count // cfg.LOGICAL_BATCH_SIZE
    total_steps = epochs * logical_steps
    sigma = _sigma(total_steps, cfg.TRAIN_SAMPLES if not smoke else sample_count, smoke)

    optimizer = make_optimizer(model)
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != head_ids:
        raise AssertionError("optimizer must contain exactly the CIFAR head parameters")

    loader = private_loader(train, seed)
    test_loader = DataLoader(
        test,
        batch_size=cfg.PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    noise_generator = torch.Generator(device=device).manual_seed(seed + 40000)
    accountant = RDPAccountant()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accuracies = []
    initial_frozen = _frozen_snapshot(model)
    initial_head = _head_snapshot(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if trainable_names != {"head.weight", "head.bias"}:
        raise AssertionError(f"unexpected trainable names: {sorted(trainable_names)}")

    for epoch in range(1, epochs + 1):
        model.eval()
        if model.training:
            raise AssertionError("private training must stay in eval mode")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if method == "dp_sgd":
            operator, builder = None, _builder_zero_stats()
            builder_seconds = 0.0
        else:
            build_start = time.perf_counter()
            operator, builder = build_from_batches(
                model,
                method,
                synthetic_stream(seed, epoch, device),
                seed,
                epoch,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            builder_seconds = time.perf_counter() - build_start

        clipper = Clipper(model, operator, method=method, max_grad_norm=cfg.MAX_GRAD_NORM)
        if method == "dp_sgd" and clipper.preconditioned_layers:
            raise AssertionError("DP-SGD must use identity geometry")
        if method != "dp_sgd" and clipper.preconditioned_layers != ["head"]:
            raise AssertionError("KFC geometry must precondition only head")
        if clipper.linear_modules.keys() != {"head"} or clipper.norm_modules:
            raise AssertionError("head-only Clipper discovery failed")

        private_start = time.perf_counter()
        loss_total = 0.0
        all_norms, all_factors = [], []
        batches = backward_calls = 0
        max_cache = max_temp = max_fallback_temp = 0
        cache_empty = True
        layer_strategies = {}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, norms, factors, _, stats = clipper.aggregate_logical(
                x,
                y,
                cfg.PHYSICAL_BATCH_SIZE,
            )
            if not _frozen_have_no_grad(model):
                raise AssertionError("frozen parameters received a gradient")
            clipper.step(optimizer, sigma, len(x), noise_generator)
            if not _frozen_have_no_grad(model):
                raise AssertionError("frozen parameters received noise/gradient state")
            accountant.step(
                noise_multiplier=sigma,
                sample_rate=cfg.LOGICAL_BATCH_SIZE / (cfg.TRAIN_SAMPLES if not smoke else sample_count),
            )
            loss_total += loss.item()
            all_norms.append(norms.cpu())
            all_factors.append(factors.cpu())
            max_temp = max(max_temp, stats["temporary_per_sample_grad_bytes"])
            max_cache = max(max_cache, stats["bk_cache_bytes"])
            max_fallback_temp = max(max_fallback_temp, stats["fallback_temporary_grad_bytes"])
            cache_empty = cache_empty and stats["cache_empty_after_step"]
            layer_strategies.update(stats["layer_strategies"])
            backward_calls += stats["backward_calls"]
            batches += 1
            del x, y, loss, norms, factors

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

        trainable_updated = any(
            not torch.equal(initial_head[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        frozen_unchanged = _frozen_unchanged(model, initial_frozen)
        if not trainable_updated or not frozen_unchanged:
            raise AssertionError("parameter update/frozen-backbone validation failed")
        if not _frozen_have_no_grad(model):
            raise AssertionError("frozen parameter gradients are not None")

        accountant_steps = sum(value[2] for value in accountant.history)
        row = {
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_total / (batches * cfg.LOGICAL_BATCH_SIZE),
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "best_accuracy": max(accuracies),
            "accuracy_auc": accuracy_auc(range(1, len(accuracies) + 1), accuracies),
            "trainable_parameter_names": sorted(trainable_names),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "trainable_parameters_updated": trainable_updated,
            "frozen_parameters_unchanged": frozen_unchanged,
            "noise_multiplier": sigma,
            "epsilon": accountant.get_epsilon(delta=cfg.DELTA),
            "accountant_steps": accountant_steps,
            "logical_steps": batches,
            "samples": batches * cfg.LOGICAL_BATCH_SIZE,
            "clip_fraction": (norms > cfg.MAX_GRAD_NORM).float().mean().item(),
            "mean_clip_factor": factors.mean().item(),
            "transformed_norm_p50": torch.quantile(norms, 0.50).item(),
            "transformed_norm_p90": torch.quantile(norms, 0.90).item(),
            "transformed_norm_p99": torch.quantile(norms, 0.99).item(),
            "transformed_norm_max": norms.max().item(),
            "builder_seconds": builder_seconds,
            "private_train_seconds": private_seconds,
            "algorithm_seconds": builder_seconds + private_seconds,
            "builder_time": builder_seconds,
            "private_time": private_seconds,
            "algorithm_time": builder_seconds + private_seconds,
            "logical_steps_per_second": batches / private_seconds,
            "samples_per_second": batches * cfg.LOGICAL_BATCH_SIZE / private_seconds,
            "logical_batch_size": cfg.LOGICAL_BATCH_SIZE,
            "physical_batch_size": cfg.PHYSICAL_BATCH_SIZE,
            "accumulation_steps": cfg.ACCUMULATION_STEPS,
            "backward_calls": backward_calls,
            "optimizer_steps": clipper.optimizer_steps,
            "noise_events": clipper.noise_events,
            "bk_cache_bytes": max_cache,
            "temporary_per_sample_grad_bytes": max_temp,
            "fallback_temporary_grad_bytes": max_fallback_temp,
            "cache_empty_after_step": cache_empty,
            "layer_strategies": layer_strategies,
            "parameters_finite": _finite_model(model),
            "preconditioned_layers": clipper.preconditioned_layers,
            "identity_geometry_layers": clipper.identity_geometry_layers,
            "optimizer_parameter_names": sorted(name for name, parameter in model.named_parameters() if id(parameter) in optimizer_ids),
            **memory,
            "peak_allocated_cuda_bytes": memory["cuda_peak_allocated_bytes"],
            "peak_reserved_cuda_bytes": memory["cuda_peak_reserved_bytes"],
            **builder,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        print(
            f"{method} seed={seed} epoch={epoch}: accuracy={accuracy:.4f} "
            f"loss={row['train_loss']:.5f} logical_steps={batches}",
            flush=True,
        )

    configuration = {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if name.isupper() and name not in {"ROOT", "RESULTS"}
    }
    configuration.update({
        "method": method,
        "seed": seed,
        "smoke": smoke,
        "epochs": epochs,
        "total_steps": total_steps,
        "noise_multiplier": sigma,
        "formal_accountant_steps": cfg.FORMAL_ACCOUNTANT_STEPS,
        "trainable_parameter_names": sorted(trainable_names),
        "trainable_parameter_count": cfg.TRAINABLE_PARAMETER_COUNT,
        "data_config": {
            "input_size": [3, cfg.IMG_SIZE, cfg.IMG_SIZE],
            "resize": [cfg.IMG_SIZE, cfg.IMG_SIZE],
            "interpolation": "bicubic",
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
            "augmentation": None,
        },
        "accounting": "Opacus RDP fixed logical-batch convention",
        "rng": {
            "initialization": "seed",
            "private_shuffle": "seed",
            "synthetic_x": "seed+10000+epoch",
            "synthetic_y": "seed+20000+epoch",
            "dp_noise": "seed+40000",
        },
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
    run(args.method, args.seed, args.smoke, args.output, args.smoke_batches)


if __name__ == "__main__":
    main()

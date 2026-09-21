"""One fresh full run; training orchestration follows exp22.run_one."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33c import config as cfg  # Configure caches before third-party imports.

# Exp22.model sets HF/Torch paths at import using config.ROOT. Redirect only
# that process-local path, leaving every Exp22 algorithm/protocol value intact.
from exp22 import config as exp22_cfg
exp22_cfg.ROOT = cfg.ROOT
from exp22.model import initialize

import pandas as pd
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from exp22.analyze import accuracy_auc
from exp33c.wiener import build_wiener, private_step, make_optimizer
from exp22.methods import Clipper


def data_transform():
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_data():
    transform = data_transform()
    # Reuse the CIFAR-10 archive prepared by Exp30.  It is read-only input;
    # all Exp33c-generated files remain under exp33c/.
    root = cfg.DATA_ROOT
    return (
        datasets.CIFAR10(root=root, train=True, download=False, transform=transform),
        datasets.CIFAR10(root=root, train=False, download=False, transform=transform),
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


def run(method: str):
    spec = cfg.RUNS[method]
    gamma = spec["gamma"]
    run_dir = cfg.RESULTS / "runs" / method
    run_dir.mkdir(parents=True, exist_ok=False)
    seed, epochs = cfg.SEED, cfg.EPOCHS
    device = torch.device("cuda:0")
    model = initialize(seed, device)
    assert sum(isinstance(m, torch.nn.Linear) for m in model.modules()) == 74
    assert all(p.requires_grad for p in model.parameters())
    train, test = load_data()
    sample_count = len(train)
    assert sample_count == cfg.TRAIN_SAMPLES
    logical_steps = sample_count // cfg.LOGICAL_BATCH_SIZE
    total_steps = epochs * logical_steps
    assert logical_steps == 195 and total_steps == 975
    sigma = get_noise_multiplier(
        target_epsilon=cfg.EPSILON, target_delta=cfg.DELTA,
        sample_rate=cfg.LOGICAL_BATCH_SIZE / sample_count,
        steps=total_steps, accountant="rdp",
    )

    optimizer = make_optimizer(model)
    loader = private_loader(train, seed)
    test_loader = DataLoader(test, batch_size=cfg.PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=0)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 40000)
    accountant = RDPAccountant()
    rows = []
    accuracies = []
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    configuration = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper() and name != "ROOT"}
    configuration.update({
        "method": method, "synthetic_physical_batch_size": 128,
        "gamma": gamma, "lr": cfg.LEARNING_RATE, "seed": seed, "epochs": epochs,
        "total_steps": total_steps, "noise_multiplier": sigma,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data_config": {"input_size": [3, cfg.IMG_SIZE, cfg.IMG_SIZE],
                        "resize": [cfg.IMG_SIZE, cfg.IMG_SIZE], "interpolation": "bicubic",
                        "mean": [0.5] * 3, "std": [0.5] * 3, "augmentation": None},
        "accounting": "RDP fixed logical-batch convention",
        "rng": {"initialization": "seed", "private_shuffle": "seed",
                "synthetic_x": "seed+10000+epoch", "synthetic_y": "seed+20000+epoch",
                "dp_noise": "seed+40000"},
    })
    (run_dir / "config.json").write_text(json.dumps(configuration, indent=2, default=str) + "\n")

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        build_start = time.perf_counter()
        wiener, builder = build_wiener(model, seed, epoch, sigma, gamma)
        torch.cuda.synchronize(device)
        builder_seconds = time.perf_counter() - build_start

        clipper = Clipper(model, None, method="bk", max_grad_norm=cfg.MAX_GRAD_NORM)
        private_start = time.perf_counter()
        loss_total = 0.0
        step_diagnostics = []
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
            step_diagnostics.append(private_step(clipper, optimizer, sigma, len(x), noise_generator, wiener))
            accountant.step(
                noise_multiplier=sigma,
                sample_rate=cfg.LOGICAL_BATCH_SIZE / cfg.TRAIN_SAMPLES,
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
            "method": method, "gamma": gamma, "lr": cfg.LEARNING_RATE, "seed": seed, "epoch": epoch,
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
            "raw_private_norm_p50": torch.quantile(norms, 0.50).item(),
            "raw_private_norm_p90": torch.quantile(norms, 0.90).item(),
            "raw_private_norm_p99": torch.quantile(norms, 0.99).item(),
            "raw_private_norm_max": norms.max().item(),
            "builder_seconds": builder_seconds, "private_train_seconds": private_seconds,
            "algorithm_seconds": builder_seconds + private_seconds,
            "logical_steps_per_second": batches / private_seconds,
            "samples_per_second": batches * cfg.LOGICAL_BATCH_SIZE / private_seconds,
            "preconditioner_build_seconds": builder_seconds,
            "logical_batch_size": cfg.LOGICAL_BATCH_SIZE,
            "physical_batch_size": cfg.PHYSICAL_BATCH_SIZE,
            "accumulation_steps": cfg.ACCUMULATION_STEPS,
            "backward_calls": backward_calls,
            "optimizer_steps": (epoch - 1) * logical_steps + clipper.optimizer_steps, "noise_events": (epoch - 1) * logical_steps + clipper.noise_events,
            "bk_cache_bytes": max_cache, "temporary_per_sample_grad_bytes": max_temp,
            "fallback_temporary_grad_bytes": max_fallback_temp,
            "cache_empty_after_step": cache_empty,
            "layer_strategies": layer_strategies,
            "parameters_finite": all(torch.isfinite(p).all().item() for p in model.parameters()),
            "all_parameters_updated": all(
                not torch.equal(initial_parameters[name], parameter.detach())
                for name, parameter in model.named_parameters()
            ),
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
        row.update(pd.DataFrame(step_diagnostics).mean().to_dict())
        if wiener is not None:
            ratios = torch.tensor([d["wiener_norm_ratio"] for d in step_diagnostics])
            for q in (10, 50, 90):
                row[f"wiener_norm_ratio_p{q}"] = ratios.quantile(q / 100).item()
            row["wiener_norm_ratio_mean"] = ratios.mean().item()
        assert batches == clipper.optimizer_steps == clipper.noise_events == 195
        assert row["accountant_steps"] == row["optimizer_steps"] == row["noise_events"] == epoch * 195
        assert row["effective_gain_min"] >= 1 - gamma - 1e-7
        assert row["effective_gain_max"] <= 1 + 1e-7
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        print(f"{method} gamma={gamma} seed={seed} epoch={epoch}: accuracy={accuracy:.4f} loss={row['train_loss']:.5f} logical_steps={batches}", flush=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(cfg.RUNS), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.method)


if __name__ == "__main__":
    main()

"""Execute one fresh ExpM1 formal run in one CUDA process."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from expm1 import config as cfg
from expm1 import data
from expm1.bk import BKClipper
from expm1.geometry import build_factors, geometry_rows
from expm1.mechanism import Shape, add_noise_and_step


def _task_module(task: str):
    if task == "mnist":
        from expm1 import mnist

        return mnist
    assert task == "vit"
    from expm1 import vit

    return vit


def _synchronize(device: torch.device) -> None:
    assert device.type == "cuda"
    torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    assert count > 0
    return loss_sum / count, correct / count


def _mean_rows(accumulator: dict[tuple[str, str, str], dict[str, float]], count: int):
    rows = []
    for (level, layer, group), values in accumulator.items():
        row = {"level": level, "layer": layer, "group": group}
        row.update({key: value / count for key, value in values.items()})
        rows.append(row)
    return rows


def _merge_geometry(
    shape: Shape,
    factors: dict[str, dict] | None,
    oracle: dict[str, dict] | None,
    spec: cfg.RunSpec,
    epoch: int,
) -> list[dict[str, object]]:
    alignment = (
        geometry_rows(
            factors,
            oracle,
            task=spec.task,
            method=spec.method,
            source=spec.source,
            beta=spec.beta,
            seed=spec.seed,
            epoch=epoch,
        )
        if factors is not None
        else []
    )
    by_layer = {row["layer"]: row for row in alignment}
    rows = []
    for diagnostics in shape.diagnostic_rows(factors):
        name = diagnostics["layer"]
        row = by_layer.get(name, {}) | diagnostics
        row.update(
            task=spec.task,
            method=spec.method,
            source=spec.source,
            beta=spec.beta,
            seed=spec.seed,
            epoch=epoch,
            research_only=True,
        )
        rows.append(row)
    assert math.isclose(
        sum(float(row["trace_S"]) for row in rows),
        shape.d_total,
        rel_tol=2e-10,
        abs_tol=2e-10 * shape.d_total,
    )
    return rows


def run(spec: cfg.RunSpec) -> None:
    protocol = cfg.task_config(spec.task)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible == str(spec.gpu), (
        f"{spec.run_name} is statically assigned to physical GPU {spec.gpu}, got {visible!r}"
    )
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    run_dir = cfg.RESULTS_ROOT / "runs" / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    module = _task_module(spec.task)
    train_data, test_data = module.load_data()
    assert (len(train_data), len(test_data)) == (
        protocol.train_samples,
        protocol.test_samples,
    )
    model = module.initialize(spec.seed, device)
    optimizer = module.build_optimizer(model)
    train_loader = data.private_loader(train_data, spec.task, spec.seed)
    evaluation_loader = data.test_loader(test_data, spec.task)
    total_steps = protocol.accountant_steps
    sigma = get_noise_multiplier(
        target_epsilon=protocol.epsilon,
        target_delta=protocol.delta,
        sample_rate=protocol.sample_rate,
        steps=total_steps,
        accountant="rdp",
    )
    assert sigma > 0
    accountant = RDPAccountant()
    noise_generator = torch.Generator(device=device).manual_seed(
        spec.seed + cfg.NOISE_SEED_OFFSET
    )
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    configuration = {
        "task": spec.task,
        "method": spec.method,
        "source": spec.source,
        "beta": spec.beta,
        "seed": spec.seed,
        "physical_gpu": spec.gpu,
        "process_device": "cuda:0",
        "epochs": protocol.epochs,
        "logical_batch_size": protocol.logical_batch_size,
        "physical_batch_size": protocol.physical_batch_size,
        "train_samples": protocol.train_samples,
        "steps_per_epoch": protocol.steps_per_epoch,
        "total_steps": total_steps,
        "epsilon_target": protocol.epsilon,
        "delta": protocol.delta,
        "noise_multiplier": sigma,
        "max_grad_norm": protocol.max_grad_norm,
        "damping": protocol.damping,
        "optimizer": protocol.optimizer,
        "learning_rate": protocol.learning_rate,
        "momentum": protocol.momentum,
        "weight_decay": protocol.weight_decay,
        "betas": protocol.betas,
        "optimizer_eps": protocol.optimizer_eps,
        "geometry_batches": cfg.AUXILIARY_BATCHES,
        "geometry_batch_size": cfg.AUXILIARY_BATCH_SIZE,
        "geometry_rebuild_every_epochs": 1,
        "oracle_batches": cfg.ORACLE_BATCHES,
        "oracle_batch_size": cfg.ORACLE_BATCH_SIZE,
        "download": False,
        "research_only": cfg.RESEARCH_ONLY,
        "rng": {
            "initialization": spec.seed,
            "private_shuffle": spec.seed,
            "public_or_pink": f"seed+10000+epoch",
            "pink_labels": f"seed+20000+epoch",
            "dp_noise": spec.seed + cfg.NOISE_SEED_OFFSET,
            "oracle_selection": cfg.ORACLE_SEED,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")

    metric_rows: list[dict[str, object]] = []
    all_geometry_rows: list[dict[str, object]] = []
    all_layer_rows: list[dict[str, object]] = []
    accuracies: list[float] = []
    optimizer_steps = noise_events = 0
    run_started = time.perf_counter()

    for epoch in range(1, protocol.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        need_g = spec.method in ("dp_kfc", "dp_kfm")
        factors = None
        builder_stats: dict[str, float | int] = {
            "geometry_build_seconds": 0.0,
            "builder_logical_batches": 0,
            "builder_forward_calls": 0,
            "builder_backward_calls": 0,
            "builder_samples": 0,
            "factor_state_bytes": 0,
        }
        if spec.method != "dp_sgd":
            source_batches = (
                module.public_calibration(spec.seed, epoch, device)
                if spec.source == "public"
                else module.pink_calibration(spec.seed, epoch, device)
            )
            _synchronize(device)
            builder_started = time.perf_counter()
            factors, builder_stats = build_factors(
                model,
                data.geometry_batches(source_batches),
                need_g=need_g,
                physical_batch_size=protocol.physical_batch_size,
            )
            _synchronize(device)
            builder_stats["geometry_build_seconds"] = time.perf_counter() - builder_started

        shape = Shape(model, spec.method, factors, spec.beta, protocol.damping)
        oracle = None
        oracle_seconds = 0.0
        if factors is not None:
            _synchronize(device)
            oracle_started = time.perf_counter()
            oracle, _ = build_factors(
                model,
                data.oracle_batches(module.oracle_calibration(device)),
                need_g=need_g,
                physical_batch_size=protocol.physical_batch_size,
            )
            _synchronize(device)
            oracle_seconds = time.perf_counter() - oracle_started
        epoch_geometry = _merge_geometry(shape, factors, oracle, spec, epoch)
        all_geometry_rows.extend(epoch_geometry)
        del oracle
        assert all(parameter.grad is None for parameter in model.parameters())

        clipper = BKClipper(model, shape, protocol.max_grad_norm)
        _synchronize(device)
        private_started = time.perf_counter()
        loss_sum = 0.0
        sample_count = logical_steps = 0
        matched_norms: list[torch.Tensor] = []
        raw_norms: list[torch.Tensor] = []
        clip_factors: list[torch.Tensor] = []
        distortion_values: dict[str, list[float]] = defaultdict(list)
        layer_accumulator: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        max_cache = max_temporary = 0
        for x, y in train_loader:
            assert len(x) == protocol.logical_batch_size
            x, y = x.to(device), y.to(device)
            aggregate = clipper.aggregate_logical(x, y, protocol.physical_batch_size)
            distortion_stats, noise_rows = add_noise_and_step(
                shape,
                optimizer,
                aggregate.raw_sum,
                aggregate.clipped_sum,
                sigma=sigma,
                bound=protocol.max_grad_norm,
                logical_batch_size=protocol.logical_batch_size,
                generator=noise_generator,
            )
            accountant.step(noise_multiplier=sigma, sample_rate=protocol.sample_rate)
            optimizer_steps += 1
            noise_events += 1
            logical_steps += 1
            sample_count += len(x)
            loss_sum += float(aggregate.loss_sum)
            matched_norms.append(aggregate.matched_norms.cpu())
            raw_norms.append(aggregate.raw_norms.cpu())
            clip_factors.append(aggregate.clip_factors.cpu())
            for key in (
                "clip_cos",
                "clip_rel_error",
                "update_cos",
                "update_rel_error",
                "total_noise_rms",
            ):
                distortion_values[key].append(float(distortion_stats[key]))
            for noise_row in noise_rows:
                index = (
                    str(noise_row["level"]),
                    str(noise_row["layer"]),
                    str(noise_row["group"]),
                )
                for key in (
                    "noise_rms",
                    "noise_energy_share",
                    "expected_noise_energy",
                    "signal_norm",
                    "snr",
                ):
                    layer_accumulator[index][key] += float(noise_row[key])
            max_cache = max(max_cache, int(aggregate.stats["bk_cache_bytes"]))
            max_temporary = max(
                max_temporary, int(aggregate.stats["temporary_per_sample_bytes"])
            )
        _synchronize(device)
        private_train_seconds = time.perf_counter() - private_started
        clipper.remove()
        assert logical_steps == protocol.steps_per_epoch
        assert sample_count == protocol.steps_per_epoch * protocol.logical_batch_size
        matched = torch.cat(matched_norms)
        raw = torch.cat(raw_norms)
        clips = torch.cat(clip_factors)
        assert torch.isfinite(matched).all() and torch.isfinite(raw).all()

        test_loss, test_accuracy = evaluate(model, evaluation_loader, device)
        accuracies.append(test_accuracy)
        auc = float(np.trapezoid(accuracies, dx=1.0))
        epoch_layer_rows = _mean_rows(layer_accumulator, logical_steps)
        for row in epoch_layer_rows:
            row.update(
                task=spec.task,
                method=spec.method,
                source=spec.source,
                beta=spec.beta,
                seed=spec.seed,
                epoch=epoch,
                research_only=True,
            )
        all_layer_rows.extend(epoch_layer_rows)
        _synchronize(device)
        epoch_wall = time.perf_counter() - epoch_started
        row: dict[str, object] = {
            "task": spec.task,
            "method": spec.method,
            "source": spec.source,
            "beta": spec.beta,
            "seed": spec.seed,
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_accuracy": max(accuracies),
            "accuracy_auc": auc,
            "epsilon": accountant.get_epsilon(protocol.delta),
            "noise_multiplier": sigma,
            "accountant_steps": sum(entry[2] for entry in accountant.history),
            "logical_steps": logical_steps,
            "samples": sample_count,
            "optimizer_steps": optimizer_steps,
            "noise_events": noise_events,
            "logical_batch_size": protocol.logical_batch_size,
            "physical_batch_size": protocol.physical_batch_size,
            "clip_fraction": float((clips < 1).float().mean()),
            "mean_clip_factor": float(clips.mean()),
            "matched_norm_p50": float(matched.quantile(0.50)),
            "matched_norm_p90": float(matched.quantile(0.90)),
            "matched_norm_p99": float(matched.quantile(0.99)),
            "matched_norm_max": float(matched.max()),
            "raw_norm_p50": float(raw.quantile(0.50)),
            "raw_norm_p90": float(raw.quantile(0.90)),
            "raw_norm_p99": float(raw.quantile(0.99)),
            "geometry_build_seconds": float(builder_stats["geometry_build_seconds"]),
            "oracle_seconds": oracle_seconds,
            "private_train_seconds": private_train_seconds,
            "wall_time_seconds": epoch_wall,
            "run_wall_time_seconds": time.perf_counter() - run_started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "bk_cache_bytes": max_cache,
            "temporary_per_sample_bytes": max_temporary,
            "operator_state_bytes": shape.operator_state_bytes,
            "factor_state_bytes": int(builder_stats["factor_state_bytes"]),
            "trace_S": shape.trace_s,
            "d_total": shape.d_total,
            "tau": shape.tau,
            "research_only": True,
            **{key: float(np.mean(values)) for key, values in distortion_values.items()},
        }
        metric_rows.append(row)
        pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(all_geometry_rows).to_csv(run_dir / "geometry.csv", index=False)
        pd.DataFrame(all_layer_rows).to_csv(run_dir / "layer_groups.csv", index=False)
        print(
            f"{spec.run_name} epoch={epoch} accuracy={test_accuracy:.4f} "
            f"epsilon={row['epsilon']:.4f}",
            flush=True,
        )
        del factors, shape, matched, raw, clips, matched_norms, raw_norms, clip_factors

    assert optimizer_steps == noise_events == protocol.accountant_steps
    updated = {
        name: not torch.equal(initial[name], parameter.detach())
        for name, parameter in model.named_parameters()
    }
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(updated.values())
    completion = {
        "run_name": spec.run_name,
        "epochs": protocol.epochs,
        "accountant_steps": sum(entry[2] for entry in accountant.history),
        "optimizer_steps": optimizer_steps,
        "noise_events": noise_events,
        "all_parameters_updated": True,
    }
    (run_dir / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")


def parse_args() -> cfg.RunSpec:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=cfg.TASKS, required=True)
    parser.add_argument("--method", choices=cfg.METHODS, required=True)
    parser.add_argument("--source", choices=("none", *cfg.SOURCES), required=True)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--seed", type=int, choices=cfg.SEEDS, required=True)
    args = parser.parse_args()
    spec = cfg.RunSpec(args.task, args.method, args.source, args.beta, args.seed)
    assert spec in cfg.FORMAL_GRID
    return spec


if __name__ == "__main__":
    run(parse_args())


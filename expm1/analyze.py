#!/usr/bin/env python3
"""Validate the complete formal grid, then write descriptive summaries and plots."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
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
    os.environ[key] = str(HERE / relative)
tempfile.tempdir = str(HERE / ".cache" / "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expm1 import config as cfg


RESULTS = Path(cfg.RESULTS_ROOT)
RUNS = RESULTS / "runs"
SEEDS = (42,)
BETAS = (0.25, 0.5, 0.75, 1.0)
SOURCES = ("pink", "public")
METHODS = ("dp_sgd", "dp_kfc", "dp_kfm", "dp_kfm_a")

CSV_OUTPUTS = (
    "all_metrics.csv",
    "final_summary.csv",
    "beta_summary.csv",
    "paired_summary.csv",
    "geometry_summary.csv",
    "layer_group_summary.csv",
)
PLOT_OUTPUTS = (
    "mnist_accuracy_vs_beta.png",
    "vit_accuracy_vs_beta.png",
    "final_accuracy_curves.png",
    "clipping_vs_beta.png",
    "clip_distortion_vs_beta.png",
    "signal_distortion_vs_beta.png",
    "update_distortion_vs_beta.png",
    "layer_group_snr.png",
    "oracle_error_utility_gap.png",
    "geometry_anisotropy_vs_beta.png",
    "runtime_memory_comparison.png",
)

METRIC_COLUMNS = (
    "task",
    "method",
    "source",
    "beta",
    "seed",
    "epoch",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "best_accuracy",
    "accuracy_auc",
    "epsilon",
    "noise_multiplier",
    "accountant_steps",
    "logical_steps",
    "samples",
    "optimizer_steps",
    "noise_events",
    "logical_batch_size",
    "physical_batch_size",
    "clip_fraction",
    "mean_clip_factor",
    "matched_norm_p50",
    "matched_norm_p90",
    "matched_norm_p99",
    "matched_norm_max",
    "raw_norm_p50",
    "raw_norm_p90",
    "raw_norm_p99",
    "clip_cos",
    "clip_rel_error",
    "signal_cos",
    "signal_rel_error",
    "update_cos",
    "update_rel_error",
    "realized_batch_size_mean",
    "realized_batch_size_min",
    "realized_batch_size_max",
    "sample_rate",
    "expected_batch_size",
    "total_noise_rms",
    "geometry_build_seconds",
    "private_train_seconds",
    "wall_time_seconds",
    "peak_cuda_allocated_bytes",
    "bk_cache_bytes",
    "temporary_per_sample_bytes",
    "operator_state_bytes",
    "factor_state_bytes",
    "trace_S",
    "d_total",
    "tau",
    "research_only",
)

GEOMETRY_COLUMNS = (
    "task",
    "method",
    "source",
    "beta",
    "seed",
    "epoch",
    "layer",
    "group",
    "trace_S",
    "d_total",
    "tau",
    "operator_state_bytes",
    "research_only",
)

LAYER_GROUP_COLUMNS = (
    "task",
    "method",
    "source",
    "beta",
    "seed",
    "epoch",
    "level",
    "layer",
    "group",
    "noise_rms",
    "noise_energy_share",
    "snr",
    "research_only",
)

FINAL_METRICS = (
    "test_accuracy",
    "best_accuracy",
    "accuracy_auc",
    "test_loss",
    "clip_fraction",
    "mean_clip_factor",
    "matched_norm_p50",
    "matched_norm_p90",
    "matched_norm_p99",
    "matched_norm_max",
    "raw_norm_p50",
    "raw_norm_p90",
    "raw_norm_p99",
    "clip_cos",
    "clip_rel_error",
    "update_cos",
    "update_rel_error",
    "total_noise_rms",
    "epsilon",
    "noise_multiplier",
    "peak_cuda_allocated_bytes",
    "bk_cache_bytes",
    "temporary_per_sample_bytes",
    "operator_state_bytes",
)


def _record(value):
    if is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, dict):
        result = dict(value)
    else:
        result = {
            name: getattr(value, name)
            for name in ("task", "method", "source", "beta", "seed")
        }
    if "gpu" not in result:
        result["gpu"] = int(getattr(value, "gpu"))
    if "name" not in result:
        result["name"] = str(getattr(value, "name"))
    return result


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _method(value) -> str:
    result = str(value).strip().lower().replace("-", "_")
    aliases = {"sgd": "dp_sgd", "kfc": "dp_kfc", "kfm": "dp_kfm", "kfm_a": "dp_kfm_a"}
    return aliases.get(result, result)


def _source(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none"
    result = str(value).strip().lower()
    return "none" if result in ("", "none", "null", "source_independent") else result


def _beta(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "nan"):
        return None
    return float(value)


def _condition(spec) -> tuple[str, str, float | None]:
    return _method(spec["method"]), _source(spec["source"]), _beta(spec["beta"])


def _task_kind(task: str) -> str:
    lowered = task.lower()
    if "mnist" in lowered:
        return "mnist"
    if "vit" in lowered or "cifar" in lowered:
        return "vit"
    raise AssertionError(f"unrecognized task name: {task!r}")


def _expected_conditions() -> set[tuple[str, str, float | None]]:
    conditions = {("dp_sgd", "none", None)}
    conditions.update(("dp_kfc", source, None) for source in SOURCES)
    for method in ("dp_kfm", "dp_kfm_a"):
        conditions.update((method, source, beta) for source in SOURCES for beta in BETAS)
    return conditions


def _validate_grid(specs: list[dict]) -> None:
    assert len(specs) == 38, f"formal grid must contain 38 runs, found {len(specs)}"
    assert len({spec["name"] for spec in specs}) == 38, "run names must be unique"
    keys = [(spec["task"], *_condition(spec), int(spec["seed"])) for spec in specs]
    assert len(set(keys)) == 38, "formal grid contains duplicate conditions"
    tasks = sorted({str(spec["task"]) for spec in specs})
    assert len(tasks) == 2 and {_task_kind(task) for task in tasks} == {"mnist", "vit"}
    expected = _expected_conditions()
    for task in tasks:
        task_specs = [spec for spec in specs if spec["task"] == task]
        assert len(task_specs) == 19, f"{task} must contain 19 runs"
        assert {int(spec["seed"]) for spec in task_specs} == set(SEEDS)
        for seed in SEEDS:
            actual = {_condition(spec) for spec in task_specs if int(spec["seed"]) == seed}
            assert actual == expected, f"bad condition grid for {task}, seed={seed}: {actual ^ expected}"
    gpu_counts = {gpu: sum(int(spec["gpu"]) == gpu for spec in specs) for gpu in (1, 2, 3)}
    assert gpu_counts == {1: 13, 2: 13, 3: 12}, f"grid is not statically balanced: {gpu_counts}"
    assert all(int(spec["gpu"]) in (1, 2, 3) for spec in specs)


def _assert_columns(frame: pd.DataFrame, required, path: Path) -> None:
    missing = set(required) - set(frame.columns)
    assert not missing, f"{path} is missing columns: {sorted(missing)}"


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value is True or value == 1 or str(value).strip().lower() == "true")


def _assert_finite(frame: pd.DataFrame, columns, path: Path) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    assert np.isfinite(values).all(), f"{path} contains non-finite values in {list(columns)}"


def _assert_identity(frame: pd.DataFrame, spec: dict, path: Path) -> None:
    assert frame["task"].astype(str).eq(str(spec["task"])).all(), f"task mismatch in {path}"
    assert frame["method"].map(_method).eq(_method(spec["method"])).all(), f"method mismatch in {path}"
    assert frame["source"].map(_source).eq(_source(spec["source"])).all(), f"source mismatch in {path}"
    actual_beta = frame["beta"].map(_beta)
    assert actual_beta.map(lambda value: value == _beta(spec["beta"])).all(), f"beta mismatch in {path}"
    assert pd.to_numeric(frame["seed"], errors="raise").eq(int(spec["seed"])).all(), f"seed mismatch in {path}"


def _config_number(configuration: dict, key: str) -> float:
    assert key in configuration, f"config.json missing {key}"
    value = float(configuration[key])
    assert math.isfinite(value)
    return value


def _validate_configuration(configuration: dict, spec: dict, task_cfg) -> None:
    assert str(configuration["task"]) == str(spec["task"])
    assert _method(configuration["method"]) == _method(spec["method"])
    assert _source(configuration["source"]) == _source(spec["source"])
    assert _beta(configuration["beta"]) == _beta(spec["beta"])
    assert int(configuration["seed"]) == int(spec["seed"])
    assert int(configuration["physical_gpu"]) == int(spec["gpu"])

    kind = _task_kind(str(spec["task"]))
    fixed = {
        "mnist": dict(train_samples=60_000, logical_batch_size=256, physical_batch_size=256, epsilon_target=1.0),
        "vit": dict(train_samples=50_000, logical_batch_size=256, physical_batch_size=128, epsilon_target=3.0),
    }[kind]
    assert int(_field(task_cfg, "epochs")) == 5
    assert int(_field(task_cfg, "train_samples")) == fixed["train_samples"]
    assert int(_field(task_cfg, "logical_batch_size")) == fixed["logical_batch_size"]
    assert int(_field(task_cfg, "physical_batch_size")) == fixed["physical_batch_size"]
    assert math.isclose(float(_field(task_cfg, "epsilon")), fixed["epsilon_target"], rel_tol=0, abs_tol=0)
    assert math.isclose(float(_field(task_cfg, "delta")), 1e-5, rel_tol=0, abs_tol=0)
    for key, expected in {
        "epochs": 5,
        **fixed,
        "delta": 1e-5,
        "max_grad_norm": 1.0,
        "damping": 1e-3,
        "geometry_batches": 10,
        "geometry_batch_size": 256,
    }.items():
        assert math.isclose(_config_number(configuration, key), expected, rel_tol=0, abs_tol=1e-12), key
    steps_per_epoch = fixed["train_samples"] // fixed["logical_batch_size"]
    assert int(configuration["steps_per_epoch"]) == steps_per_epoch
    assert int(configuration["total_steps"]) == 5 * steps_per_epoch
    assert int(configuration["geometry_rebuild_every_epochs"]) == 1
    assert int(configuration["oracle_batches"]) == 10
    assert int(configuration["oracle_batch_size"]) == 256
    assert configuration["process_device"] == "cuda:0"
    assert configuration["optimizer"] == _field(task_cfg, "optimizer")
    assert configuration["download"] is False
    assert str(configuration["research_only"]).startswith("research-only")


def _auc_prefix(epochs: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    if len(values) > 1:
        result[1:] = np.cumsum((values[:-1] + values[1:]) * 0.5 * np.diff(epochs))
    return result


def _validate_metrics(frame: pd.DataFrame, spec: dict, task_cfg, path: Path) -> pd.DataFrame:
    _assert_columns(frame, METRIC_COLUMNS, path)
    _assert_identity(frame, spec, path)
    epochs = int(_field(task_cfg, "epochs"))
    train_samples = int(_field(task_cfg, "train_samples"))
    logical_batch = int(_field(task_cfg, "logical_batch_size"))
    expected_steps = train_samples // logical_batch
    assert len(frame) == epochs, f"{path} must contain exactly {epochs} rows"
    frame = frame.sort_values("epoch").reset_index(drop=True)
    assert frame["epoch"].astype(int).tolist() == list(range(1, epochs + 1)), f"bad epochs in {path}"
    assert _truthy(frame["research_only"]).all(), f"private diagnostics are not marked research_only in {path}"

    numeric = set(METRIC_COLUMNS) - {"task", "method", "source", "beta", "research_only"}
    _assert_finite(frame, numeric, path)
    assert frame["accountant_steps"].astype(int).tolist() == [expected_steps * epoch for epoch in range(1, epochs + 1)]
    assert frame["optimizer_steps"].astype(int).tolist() == [expected_steps * epoch for epoch in range(1, epochs + 1)]
    assert frame["noise_events"].astype(int).tolist() == [expected_steps * epoch for epoch in range(1, epochs + 1)]
    assert frame["logical_steps"].astype(int).eq(expected_steps).all()
    assert frame["samples"].astype(int).eq(expected_steps * logical_batch).all()
    assert frame["logical_batch_size"].astype(int).eq(logical_batch).all()
    physical = 256 if _task_kind(str(spec["task"])) == "mnist" else 128
    assert frame["physical_batch_size"].astype(int).eq(physical).all()

    for column in ("test_accuracy", "best_accuracy", "clip_fraction", "mean_clip_factor"):
        assert frame[column].between(0, 1).all(), f"{column} outside [0,1] in {path}"
    for column in ("clip_cos", "update_cos"):
        assert frame[column].between(-1, 1).all(), f"{column} outside [-1,1] in {path}"
    nonnegative = (
        "train_loss", "test_loss", "accuracy_auc", "epsilon", "noise_multiplier",
        "matched_norm_p50", "matched_norm_p90", "matched_norm_p99", "matched_norm_max",
        "raw_norm_p50", "raw_norm_p90", "raw_norm_p99", "clip_rel_error",
        "update_rel_error", "total_noise_rms", "geometry_build_seconds",
        "private_train_seconds", "wall_time_seconds", "peak_cuda_allocated_bytes",
        "bk_cache_bytes", "temporary_per_sample_bytes", "operator_state_bytes",
        "factor_state_bytes", "trace_S", "d_total", "tau",
    )
    assert (frame[list(nonnegative)] >= 0).all().all(), f"negative metric in {path}"
    assert frame["noise_multiplier"].gt(0).all() and frame["noise_multiplier"].nunique() == 1
    assert frame["epsilon"].gt(0).all() and frame["epsilon"].is_monotonic_increasing
    epsilon_target = 1.0 if _task_kind(str(spec["task"])) == "mnist" else 3.0
    assert frame["epsilon"].iloc[-1] <= epsilon_target * (1 + 1e-6)

    assert (frame["matched_norm_p50"] <= frame["matched_norm_p90"]).all()
    assert (frame["matched_norm_p90"] <= frame["matched_norm_p99"]).all()
    assert (frame["matched_norm_p99"] <= frame["matched_norm_max"]).all()
    assert (frame["raw_norm_p50"] <= frame["raw_norm_p90"]).all()
    assert (frame["raw_norm_p90"] <= frame["raw_norm_p99"]).all()

    accuracy = frame["test_accuracy"].to_numpy(float)
    epoch_values = frame["epoch"].to_numpy(float)
    np.testing.assert_allclose(frame["best_accuracy"], np.maximum.accumulate(accuracy), rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame["accuracy_auc"], _auc_prefix(epoch_values, accuracy), rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame["trace_S"], frame["d_total"], rtol=2e-10,
                               atol=2e-10 * frame["d_total"].max())
    if _method(spec["method"]) == "dp_sgd":
        assert frame["geometry_build_seconds"].eq(0).all()
        assert frame["operator_state_bytes"].eq(0).all()
        assert frame["factor_state_bytes"].eq(0).all()
        assert frame["tau"].eq(1).all()
    else:
        assert frame["operator_state_bytes"].gt(0).all()
        assert frame["factor_state_bytes"].gt(0).all()
        assert frame["geometry_build_seconds"].gt(0).all()
        if _method(spec["method"]) == "dp_kfc":
            assert frame["tau"].eq(1).all()
        else:
            assert frame["tau"].gt(0).all()
    frame["run_name"] = spec["name"]
    frame["physical_gpu"] = int(spec["gpu"])
    return frame


def _validate_geometry(frame: pd.DataFrame, spec: dict, task_cfg, path: Path) -> pd.DataFrame:
    _assert_columns(frame, GEOMETRY_COLUMNS, path)
    _assert_identity(frame, spec, path)
    epochs = int(_field(task_cfg, "epochs"))
    assert not frame.empty
    assert set(frame["epoch"].astype(int)) == set(range(1, epochs + 1))
    assert frame.groupby("epoch").size().gt(0).all()
    assert _truthy(frame["research_only"]).all(), f"oracle diagnostics are not research_only in {path}"
    method = _method(spec["method"])
    _assert_finite(frame, ("trace_S", "d_total", "tau", "operator_state_bytes"), path)
    totals = frame.groupby("epoch").agg(trace_S=("trace_S", "sum"), d_total=("d_total", "first"))
    np.testing.assert_allclose(totals["trace_S"], totals["d_total"], rtol=2e-10, atol=2e-10 * totals["d_total"].max())
    if method == "dp_sgd":
        factor_columns = {
            "condition_S", "log_eigenvalue_spread_S", "effective_rank_S",
            "A_condition_raw", "A_eigenvalue_min", "A_eigenvalue_max", "cosA", "relative_error_A",
            "G_condition_raw", "G_eigenvalue_min", "G_eigenvalue_max", "cosG", "relative_error_G",
        }
        assert not (factor_columns & set(frame.columns)), f"DP-SGD contains factor diagnostics in {path}"
        assert frame["tau"].eq(1).all()
        assert frame["operator_state_bytes"].eq(0).all()
        frame["run_name"] = spec["name"]
        return frame

    required_factor_columns = (
        "condition_S", "log_eigenvalue_spread_S", "effective_rank_S",
        "A_condition_raw", "A_eigenvalue_min", "A_eigenvalue_max", "cosA", "relative_error_A",
    )
    _assert_columns(frame, required_factor_columns, path)
    affine = frame[frame["layer"].ne("identity")]
    _assert_finite(affine, required_factor_columns, path)
    assert affine["condition_S"].ge(1).all()
    assert affine["log_eigenvalue_spread_S"].ge(0).all()
    assert affine["effective_rank_S"].gt(0).all()
    assert affine["A_eigenvalue_min"].ge(0).all()
    assert (affine["A_eigenvalue_max"] >= affine["A_eigenvalue_min"]).all()
    assert affine["cosA"].between(-1, 1).all()
    assert affine["relative_error_A"].ge(0).all()

    g_columns = ("G_condition_raw", "G_eigenvalue_min", "G_eigenvalue_max", "cosG", "relative_error_G")
    if method == "dp_kfm_a":
        present = list(set(g_columns) & set(frame.columns))
        assert not present or frame[present].isna().all().all(), f"DP-KFM-A used G in {path}"
    else:
        _assert_columns(frame, g_columns, path)
        _assert_finite(affine, g_columns, path)
        assert affine["G_eigenvalue_min"].ge(0).all()
        assert (affine["G_eigenvalue_max"] >= affine["G_eigenvalue_min"]).all()
        assert affine["cosG"].between(-1, 1).all()
        assert affine["relative_error_G"].ge(0).all()
    if method in ("dp_kfm", "dp_kfm_a"):
        assert frame["tau"].gt(0).all()
    else:
        assert frame["tau"].eq(1).all()
    frame["run_name"] = spec["name"]
    return frame


def _validate_layer_groups(frame: pd.DataFrame, spec: dict, task_cfg, path: Path) -> pd.DataFrame:
    _assert_columns(frame, LAYER_GROUP_COLUMNS, path)
    _assert_identity(frame, spec, path)
    epochs = int(_field(task_cfg, "epochs"))
    assert not frame.empty
    assert set(frame["epoch"].astype(int)) == set(range(1, epochs + 1))
    assert set(frame["level"].astype(str)) == {"layer", "group"}
    assert _truthy(frame["research_only"]).all(), f"private SNR is not research_only in {path}"
    _assert_finite(frame, ("noise_rms", "noise_energy_share", "snr"), path)
    assert frame["noise_rms"].ge(0).all() and frame["snr"].ge(0).all()
    assert frame["noise_energy_share"].between(0, 1).all()
    groups = frame[frame["level"].eq("group")]
    shares = groups.groupby("epoch")["noise_energy_share"].sum()
    np.testing.assert_allclose(shares, np.ones(epochs), rtol=1e-5, atol=1e-5)
    if _task_kind(str(spec["task"])) == "vit":
        required = {"attention_qkv", "attention_out", "mlp_fc1", "mlp_fc2", "patch_head", "identity"}
        for epoch, part in groups.groupby("epoch"):
            assert required <= set(part["group"].astype(str)), f"missing ViT groups at epoch {epoch} in {path}"
    frame["run_name"] = spec["name"]
    return frame


def _validate_completion(completion: dict, spec: dict, task_cfg) -> None:
    expected_steps = int(_field(task_cfg, "epochs")) * (
        int(_field(task_cfg, "train_samples")) // int(_field(task_cfg, "logical_batch_size"))
    )
    assert completion == {
        "run_name": spec["name"],
        "epochs": int(_field(task_cfg, "epochs")),
        "accountant_steps": expected_steps,
        "optimizer_steps": expected_steps,
        "noise_events": expected_steps,
        "all_parameters_updated": True,
    }


def _load_and_validate():
    specs = [_record(spec) for spec in cfg.formal_runs()]
    _validate_grid(specs)
    metrics, geometries, layer_groups = [], [], []
    for spec in specs:
        directory = RUNS / str(spec["name"])
        assert directory.is_dir(), f"missing formal run directory: {directory}"
        required = (
            directory / "config.json",
            directory / "metrics.csv",
            directory / "geometry.csv",
            directory / "layer_groups.csv",
            directory / "complete.json",
        )
        assert all(path.is_file() for path in required), f"incomplete formal run: {directory}"
        task_cfg = cfg.task_config(spec["task"])
        configuration = json.loads((directory / "config.json").read_text())
        _validate_configuration(configuration, spec, task_cfg)
        completion = json.loads((directory / "complete.json").read_text())
        _validate_completion(completion, spec, task_cfg)
        metric = _validate_metrics(pd.read_csv(directory / "metrics.csv"), spec, task_cfg, directory / "metrics.csv")
        np.testing.assert_allclose(metric["noise_multiplier"], float(configuration["noise_multiplier"]), rtol=0, atol=0)
        metrics.append(metric)
        groups = pd.read_csv(directory / "layer_groups.csv")
        layer_groups.append(_validate_layer_groups(groups, spec, task_cfg, directory / "layer_groups.csv"))
        geometry_path = directory / "geometry.csv"
        geometries.append(_validate_geometry(pd.read_csv(geometry_path), spec, task_cfg, geometry_path))
    metric_frame = pd.concat(metrics, ignore_index=True)
    geometry_frame = pd.concat(geometries, ignore_index=True)
    layer_group_frame = pd.concat(layer_groups, ignore_index=True)
    assert len(metric_frame) == 38 * 5
    return specs, metric_frame, geometry_frame, layer_group_frame


def _aggregate(frame: pd.DataFrame, groups, values) -> pd.DataFrame:
    grouped = frame.groupby(list(groups), dropna=False, sort=True)[list(values)].first()
    result = grouped.reset_index()
    result["n"] = frame.groupby(list(groups), dropna=False).size().to_numpy()
    return result


def _paired(final: pd.DataFrame) -> pd.DataFrame:
    values = {}
    for row in final.itertuples(index=False):
        key = (row.task, int(row.seed), _method(row.method), _source(row.source), _beta(row.beta))
        assert key not in values
        values[key] = row

    rows = []
    tasks = sorted(final["task"].unique())

    def add(task, seed, metric, comparison, left, right, method, source, beta):
        rows.append({
            "task": task,
            "metric": metric,
            "comparison": comparison,
            "method": method,
            "source": source,
            "beta": beta,
            "seed": seed,
            "delta": float(getattr(values[left], metric) - getattr(values[right], metric)),
        })

    for task in tasks:
        for seed in SEEDS:
            sgd = (task, seed, "dp_sgd", "none", None)
            for source in SOURCES:
                kfc = (task, seed, "dp_kfc", source, None)
                for beta in BETAS:
                    kfm = (task, seed, "dp_kfm", source, beta)
                    kfm_a = (task, seed, "dp_kfm_a", source, beta)
                    for metric in ("test_accuracy", "best_accuracy", "accuracy_auc"):
                        add(task, seed, metric, "KFM - SGD", kfm, sgd, "dp_kfm", source, beta)
                        add(task, seed, metric, "KFM - KFC", kfm, kfc, "dp_kfm", source, beta)
                        add(task, seed, metric, "KFM-A - KFM", kfm_a, kfm, "dp_kfm_a", source, beta)
            for method in ("dp_kfc", "dp_kfm", "dp_kfm_a"):
                betas = (None,) if method == "dp_kfc" else BETAS
                for beta in betas:
                    public = (task, seed, method, "public", beta)
                    pink = (task, seed, method, "pink", beta)
                    for metric in ("test_accuracy", "best_accuracy", "accuracy_auc"):
                        add(task, seed, metric, "public - pink", public, pink, method, "public-pink", beta)
    paired = pd.DataFrame(rows)
    group_columns = ["task", "metric", "comparison", "method", "source", "beta"]
    return paired


def _condition_label(method, source, beta) -> str:
    method = _method(method).replace("dp_", "").upper().replace("_", "-")
    source = _source(source)
    suffix = "" if source == "none" else f"/{source}"
    if _beta(beta) is not None:
        suffix += f"/b={float(beta):g}"
    return method + suffix


def _save(stage: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(stage / name, dpi=180)
    plt.close()


def _accuracy_beta_plot(final: pd.DataFrame, task: str, stage: Path, filename: str) -> None:
    data = final[(final["task"] == task) & final["method"].isin(("dp_kfm", "dp_kfm_a"))]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (method, source), part in data.groupby(["method", "source"]):
        curve = part.groupby("beta")["test_accuracy"].agg(["mean", "std"]).sort_index()
        ax.errorbar(curve.index, curve["mean"], yerr=curve["std"], marker="o", capsize=3,
                    label=_condition_label(method, source, None))
    ax.set(xlabel="beta", ylabel="final test accuracy", title=f"{task}: accuracy vs beta")
    ax.set_xticks(BETAS)
    ax.legend(fontsize=8)
    _save(stage, filename)


def _final_accuracy_curves(metrics: pd.DataFrame, stage: Path) -> None:
    tasks = sorted(metrics["task"].unique(), key=_task_kind)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        data = metrics[metrics["task"] == task]
        for keys, part in data.groupby(["method", "source", "beta"], dropna=False):
            curve = part.groupby("epoch")["test_accuracy"].mean()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1, label=_condition_label(*keys))
        ax.set(xlabel="epoch", ylabel="test accuracy", title=task)
        ax.legend(fontsize=5, ncol=2)
    _save(stage, "final_accuracy_curves.png")


def _beta_diagnostic(metrics: pd.DataFrame, stage: Path, filename: str, columns, title: str) -> None:
    final = metrics.sort_values("epoch").groupby("run_name", as_index=False).tail(1)
    tasks = sorted(final["task"].unique(), key=_task_kind)
    fig, axes = plt.subplots(len(tasks), len(columns), figsize=(6 * len(columns), 4.5 * len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        data = final[(final["task"] == task) & final["method"].isin(("dp_kfm", "dp_kfm_a"))]
        for col, metric in enumerate(columns):
            ax = axes[row, col]
            for (method, source), part in data.groupby(["method", "source"]):
                curve = part.groupby("beta")[metric].agg(["mean", "std"]).sort_index()
                ax.errorbar(curve.index, curve["mean"], yerr=curve["std"], marker="o", capsize=2,
                            label=_condition_label(method, source, None))
            ax.set(xlabel="beta", ylabel=metric, title=f"{task}: {metric}")
            ax.set_xticks(BETAS)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle(title + " — research-only private diagnostics", fontsize=12)
    _save(stage, filename)


def _layer_group_snr(layer_groups: pd.DataFrame, stage: Path) -> None:
    final_epoch = layer_groups["epoch"] == layer_groups.groupby("run_name")["epoch"].transform("max")
    data = layer_groups[final_epoch & layer_groups["level"].eq("group")].copy()
    tasks = sorted(data["task"].unique(), key=_task_kind)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        part = data[data["task"] == task].copy()
        part["condition"] = [_condition_label(m, s, b) for m, s, b in zip(part.method, part.source, part.beta)]
        table = part.pivot_table(index="condition", columns="group", values="snr", aggfunc="mean").sort_index()
        image = ax.imshow(np.log10(table.to_numpy() + 1e-12), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(table.index)), table.index, fontsize=6)
        ax.set_title(f"{task}: log10 group SNR")
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.suptitle("Research-only clipped-signal SNR")
    _save(stage, "layer_group_snr.png")


def _oracle_gap(final: pd.DataFrame, geometry: pd.DataFrame, stage: Path) -> None:
    kfm = final[final["method"].eq("dp_kfm")][["task", "seed", "source", "beta", "test_accuracy"]]
    kfma = final[final["method"].eq("dp_kfm_a")][["task", "seed", "source", "beta", "test_accuracy"]]
    gap = kfma.merge(kfm, on=["task", "seed", "source", "beta"], suffixes=("_a", "_full"), validate="one_to_one")
    gap["utility_gap"] = gap["test_accuracy_a"] - gap["test_accuracy_full"]
    last = geometry[geometry["epoch"] == geometry.groupby("run_name")["epoch"].transform("max")]
    errors = last[last["method"].eq("dp_kfm")].groupby(
        ["task", "seed", "source", "beta"], as_index=False
    )[["relative_error_A", "relative_error_G"]].mean()
    data = gap.merge(errors, on=["task", "seed", "source", "beta"], validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ("relative_error_A", "relative_error_G"), ("A oracle error", "G oracle error")):
        for (task, source), part in data.groupby(["task", "source"]):
            ax.scatter(part[metric], part["utility_gap"], label=f"{task}/{source}", alpha=0.8)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(xlabel=label, ylabel="KFM-A - KFM final accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("Research-only oracle mismatch and paired utility gap")
    _save(stage, "oracle_error_utility_gap.png")


def _geometry_anisotropy(geometry: pd.DataFrame, stage: Path) -> None:
    data = geometry[geometry["method"].isin(("dp_kfm", "dp_kfm_a"))]
    data = data[data["epoch"] == data.groupby("run_name")["epoch"].transform("max")]
    tasks = sorted(data["task"].unique(), key=_task_kind)
    metrics = ("condition_S", "log_eigenvalue_spread_S", "effective_rank_S")
    fig, axes = plt.subplots(len(tasks), len(metrics), figsize=(17, 8), squeeze=False)
    for row, task in enumerate(tasks):
        part = data[data["task"] == task]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for (method, source), group in part.groupby(["method", "source"]):
                curve = group.groupby("beta")[metric].mean().sort_index()
                ax.plot(curve.index, curve.values, marker="o", label=_condition_label(method, source, None))
            ax.set(xlabel="beta", ylabel=metric, title=f"{task}: {metric}")
            ax.set_xticks(BETAS)
    axes[0, 0].legend(fontsize=7)
    _save(stage, "geometry_anisotropy_vs_beta.png")


def _runtime_memory(metrics: pd.DataFrame, stage: Path) -> None:
    per_run = metrics.groupby(["run_name", "task", "method"], as_index=False).agg(
        wall_time_seconds=("wall_time_seconds", "sum"),
        peak_cuda_allocated_bytes=("peak_cuda_allocated_bytes", "max"),
    )
    tasks = sorted(per_run["task"].unique(), key=_task_kind)
    fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 8), squeeze=False)
    for row, task in enumerate(tasks):
        part = per_run[per_run["task"] == task]
        order = list(METHODS)
        runtime = part.groupby("method")["wall_time_seconds"].mean().reindex(order)
        memory = part.groupby("method")["peak_cuda_allocated_bytes"].mean().reindex(order) / 2**30
        axes[row, 0].bar(range(len(order)), runtime)
        axes[row, 1].bar(range(len(order)), memory)
        for ax, ylabel in zip(axes[row], ("wall time / run (s)", "peak CUDA allocated (GiB)")):
            ax.set_xticks(range(len(order)), [m.replace("dp_", "").upper() for m in order], rotation=20)
            ax.set_ylabel(ylabel)
            ax.set_title(task)
    _save(stage, "runtime_memory_comparison.png")


def _write_outputs(metrics: pd.DataFrame, geometry: pd.DataFrame, layer_groups: pd.DataFrame, stage: Path) -> None:
    sort = ["task", "method", "source", "beta", "seed", "epoch"]
    metrics.sort_values(sort, na_position="first").to_csv(stage / "all_metrics.csv", index=False)
    final = metrics.sort_values("epoch").groupby("run_name", as_index=False).tail(1).copy()
    final_summary = _aggregate(final, ("task", "method", "source", "beta"), FINAL_METRICS)
    assert final_summary["n"].eq(1).all()
    final_summary.to_csv(stage / "final_summary.csv", index=False)
    beta = final[final["method"].isin(("dp_kfm", "dp_kfm_a"))]
    beta_summary = _aggregate(beta, ("task", "method", "source", "beta"), FINAL_METRICS)
    assert beta_summary["n"].eq(1).all()
    beta_summary.to_csv(stage / "beta_summary.csv", index=False)
    _paired(final).to_csv(stage / "paired_summary.csv", index=False)

    geometry_values = (
        "trace_S", "d_total", "tau", "condition_S", "log_eigenvalue_spread_S", "effective_rank_S",
        "A_condition_raw", "G_condition_raw",
        "A_eigenvalue_min", "A_eigenvalue_max", "G_eigenvalue_min", "G_eigenvalue_max",
        "cosA", "relative_error_A", "cosG", "relative_error_G",
    )
    geometry_summary = _aggregate(
        geometry, ("task", "method", "source", "beta", "epoch", "layer", "group"), geometry_values
    )
    geometry_summary.to_csv(stage / "geometry_summary.csv", index=False)
    layer_group_summary = _aggregate(
        layer_groups, ("task", "method", "source", "beta", "epoch", "level", "layer", "group"),
        ("noise_rms", "noise_energy_share", "snr"),
    )
    layer_group_summary.to_csv(stage / "layer_group_summary.csv", index=False)

    tasks = {_task_kind(task): task for task in final["task"].unique()}
    _accuracy_beta_plot(final, tasks["mnist"], stage, "mnist_accuracy_vs_beta.png")
    _accuracy_beta_plot(final, tasks["vit"], stage, "vit_accuracy_vs_beta.png")
    _final_accuracy_curves(metrics, stage)
    _beta_diagnostic(metrics, stage, "clipping_vs_beta.png", ("clip_fraction", "mean_clip_factor"), "Clipping")
    _beta_diagnostic(metrics, stage, "clip_distortion_vs_beta.png", ("clip_cos", "clip_rel_error"), "Clipping distortion")
    _beta_diagnostic(metrics, stage, "signal_distortion_vs_beta.png", ("signal_cos", "signal_rel_error"), "Mechanism signal distortion")
    _beta_diagnostic(metrics, stage, "update_distortion_vs_beta.png", ("update_cos", "update_rel_error"), "DP update distortion")
    _layer_group_snr(layer_groups, stage)
    _oracle_gap(final, geometry, stage)
    _geometry_anisotropy(geometry, stage)
    _runtime_memory(metrics, stage)


def analyze() -> None:
    _, metrics, geometry, layer_groups = _load_and_validate()
    outputs = [RESULTS / name for name in (*CSV_OUTPUTS, *PLOT_OUTPUTS)]
    collisions = [path for path in outputs if path.exists()]
    assert not collisions, f"analysis outputs already exist; refusing to overwrite: {collisions}"
    with tempfile.TemporaryDirectory(prefix=".analysis-", dir=RESULTS) as temporary:
        stage = Path(temporary)
        _write_outputs(metrics, geometry, layer_groups, stage)
        missing = [name for name in (*CSV_OUTPUTS, *PLOT_OUTPUTS) if not (stage / name).is_file()]
        assert not missing, f"analysis failed to produce: {missing}"
        for name in (*CSV_OUTPUTS, *PLOT_OUTPUTS):
            target = RESULTS / name
            assert not target.exists(), f"refusing to overwrite {target}"
            (stage / name).rename(target)
    print(f"Validated 38 complete formal runs and wrote analysis to {RESULTS}")


if __name__ == "__main__":
    analyze()

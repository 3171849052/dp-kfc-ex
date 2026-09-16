"""Summarize Exp22 run CSVs and paired descriptive deltas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))

import numpy as np
import pandas as pd

from exp22 import config as cfg


METRICS = (
    "train_loss", "test_loss", "test_accuracy", "best_accuracy", "accuracy_auc",
    "clip_fraction", "mean_clip_factor", "transformed_norm_p50",
    "transformed_norm_p90", "transformed_norm_p99", "transformed_norm_max",
    "algorithm_seconds", "logical_steps_per_second", "samples_per_second",
)
COMPARISONS = (
    ("dp_kfc_a_bk", "dp_sgd"),
    ("dp_kfc", "dp_sgd"),
    ("dp_kfc_a_bk", "dp_kfc"),
)


def accuracy_auc(epochs, accuracies):
    """Exp20 definition: trapezoidal integral over observed epoch points."""
    return float(np.trapezoid(np.asarray(accuracies, dtype=float), np.asarray(epochs, dtype=float)))


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int = 20_000):
    if len(values) < 2:
        raise ValueError("paired bootstrap requires at least two values")
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def analyze(output: Path, smoke: bool = False):
    run_files = sorted((output / "runs").glob("*/metrics.csv"))
    if not run_files:
        raise FileNotFoundError(f"no run metrics under {output / 'runs'}")
    frames = [pd.read_csv(path) for path in run_files]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    final = metrics.sort_values("epoch").groupby(["method", "seed"], as_index=False).tail(1).copy()
    final.to_csv(output / "summary.csv", index=False)

    numeric = [name for name in METRICS if name in final]
    method_rows = []
    for method, frame in final.groupby("method", sort=False):
        row = {"method": method, "n_seeds": len(frame)}
        for name in numeric:
            row[f"{name}_mean"] = frame[name].mean()
            row[f"{name}_sample_std"] = frame[name].std(ddof=1) if len(frame) > 1 else np.nan
        method_rows.append(row)
    pd.DataFrame(method_rows).to_csv(output / "method_summary.csv", index=False)

    rng = np.random.default_rng(2209)
    paired_rows = []
    for first, second in COMPARISONS:
        left = final[final.method == first].set_index("seed")
        right = final[final.method == second].set_index("seed")
        common = sorted(set(left.index) & set(right.index))
        if not common:
            continue
        for name in numeric:
            deltas = (left.loc[common, name] - right.loc[common, name]).to_numpy(dtype=float)
            if len(deltas) >= 2:
                ci_low, ci_high = _bootstrap(deltas, rng)
                sample_std = deltas.std(ddof=1)
                bootstrap_seed = 2209
            else:
                ci_low = ci_high = sample_std = bootstrap_seed = np.nan
            paired_rows.append({
                "method_a": first, "method_b": second, "metric": name,
                "n_seeds": len(deltas), "mean_delta_a_minus_b": deltas.mean(),
                "sample_std": sample_std,
                "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
                "bootstrap_seed": bootstrap_seed,
            })
    pd.DataFrame(paired_rows).to_csv(output / "paired_summary.csv", index=False)

    configuration = {
        name: getattr(cfg, name) for name in dir(cfg)
        if name.isupper() and name not in {"ROOT", "RESULTS"}
    }
    configuration.update({"smoke": smoke, "bootstrap_seed": 2209, "run_count": len(run_files)})
    (output / "config.json").write_text(json.dumps(configuration, indent=2, default=str) + "\n")
    print(f"analyzed {len(run_files)} runs -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or (cfg.RESULTS / "smoke" if args.smoke else cfg.RESULTS)).resolve()
    analyze(output, args.smoke)


if __name__ == "__main__":
    main()

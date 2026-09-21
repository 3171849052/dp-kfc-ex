"""Require all 14 complete runs, then summarize final epochs and plot."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp30 import config as cfg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED = (
    "method", "damping", "epoch", "train_loss", "test_loss", "test_accuracy",
    "best_accuracy", "accuracy_auc", "clip_fraction", "mean_clip_factor",
    "transformed_norm_p50", "transformed_norm_p90", "transformed_norm_p99",
    "transformed_norm_max", "noise_multiplier", "epsilon", "logical_steps",
    "optimizer_steps", "noise_events", "accountant_steps", "builder_seconds",
    "private_train_seconds", "operator_state_bytes",
)


def analyze(output=cfg.RESULTS):
    paths = [output / "runs" / cfg.run_name(method, damping) / "metrics.csv"
             for _, method, damping in cfg.grid()]
    if set((output / "runs").glob("*/metrics.csv")) != set(paths):
        raise ValueError("Expected exactly the 14 configured run CSVs")
    rows = []
    for (_, method, damping), path in zip(cfg.grid(), paths):
        frame = pd.read_csv(path)
        missing = set(REQUIRED) - set(frame.columns)
        if missing or not any(c.startswith("group_norm_") for c in frame.columns):
            raise ValueError(f"Missing metrics in {path}: {sorted(missing)}")
        if (frame.epoch.tolist() != list(range(1, cfg.EPOCHS + 1))
                or not frame.method.eq(method).all()
                or not np.isclose(frame.damping, damping, rtol=1e-12, atol=0).all()):
            raise ValueError(f"Incomplete or mismatched run: {path}")
        steps = cfg.TRAIN_SAMPLES // cfg.LOGICAL_BATCH_SIZE
        for column in ("logical_steps", "optimizer_steps", "noise_events"):
            if not frame[column].eq(steps).all():
                raise ValueError(f"Unexpected {column}: {path}")
        if not frame.accountant_steps.eq(frame.epoch * steps).all():
            raise ValueError(f"Unexpected accountant steps: {path}")
        rows.append(frame.iloc[-1])
    summary = pd.DataFrame(rows).reset_index(drop=True)
    summary.to_csv(output / "summary.csv", index=False)
    for method in cfg.METHODS:
        ranking = summary[summary.method == method].sort_values(
            ["test_accuracy", "damping"], ascending=[False, True])
        ranking.to_csv(output / f"{method}_ranking.csv", index=False)
    paired = summary.pivot(index="damping", columns="method", values="test_accuracy")
    paired = paired.rename(columns={m: f"{m}_accuracy" for m in cfg.METHODS})
    paired["delta_accuracy"] = paired.dp_kfc_a_accuracy - paired.dp_kfc_accuracy
    paired.reset_index().to_csv(output / "paired.csv", index=False)

    for suffix, metric in (
        ("accuracy", "test_accuracy"), ("test_loss", "test_loss"),
        ("clip_fraction", "clip_fraction"), ("mean_clip_factor", "mean_clip_factor"),
        ("norm_tail", None), ("delta_accuracy", "delta_accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if metric == "delta_accuracy":
            ax.plot(paired.index, paired[metric], "o-", label="DP-KFC-A − DP-KFC")
            ax.axhline(0, color="gray", linewidth=0.8)
        else:
            for method, color in zip(cfg.METHODS, ("tab:blue", "tab:orange")):
                data = summary[summary.method == method].sort_values("damping")
                metrics = [(metric, "-", cfg.LABELS[method])] if metric else [
                    ("transformed_norm_p90", ":", f"{cfg.LABELS[method]} p90"),
                    ("transformed_norm_p99", "-", f"{cfg.LABELS[method]} p99"),
                    ("transformed_norm_max", "--", f"{cfg.LABELS[method]} max"),
                ]
                for column, style, label in metrics:
                    ax.plot(data.damping, data[column], marker="o", linestyle=style,
                            color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Damping")
        ax.set_ylabel(metric or "Transformed norm tail")
        ax.set_title("Exp30 · final epoch · seed 42")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"damping_vs_{suffix}.png", dpi=180)
        plt.close(fig)
    print(f"Analyzed {len(summary)} complete runs -> {output}")


if __name__ == "__main__":
    analyze()

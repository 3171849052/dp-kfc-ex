"""Validate and plot the five completed runs; test_loss denotes validation loss."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp29 import config as cfg


def validate(frame):
    assert len(frame) == 5
    assert not frame.damping.duplicated().any()
    assert set(frame.damping) == set(cfg.DAMPING_VALUES)
    assert np.allclose(frame.effective_damping, frame.damping + 1e-5)
    fixed = dict(method="DP-KFC-A", geometry=cfg.GEOMETRY, source=cfg.SOURCE,
                 engine=cfg.ENGINE, seed=cfg.SEED, epoch=cfg.EPOCHS,
                 epsilon_target=cfg.EPSILON, delta=cfg.DELTA, profiled=False,
                 collect_diagnostics=True, logical_batch_size=cfg.LOGICAL_BATCH_SIZE,
                 physical_batch_size=cfg.PHYSICAL_BATCH_SIZE,
                 geometry_batch_size=cfg.GEOMETRY_BATCH_SIZE,
                 geometry_physical_batch_size=cfg.GEOMETRY_PHYSICAL_BATCH_SIZE,
                 C=cfg.C, learning_rate=cfg.LEARNING_RATE, a_power=cfg.A_POWER, train_size=67349,
                 validation_size=872)
    for key, value in fixed.items():
        assert frame[key].eq(value).all(), key
    assert np.isfinite(frame[list(cfg.METRICS)].to_numpy()).all()
    for key in ("accuracy", "clip_fraction", "mean_clip_factor"):
        assert frame[key].between(0, 1).all(), key
    assert np.allclose(frame.noise_std, frame.noise_multiplier * frame.C)
    assert frame[list(cfg.STEP_FIELDS)].eq(cfg.EPOCHS * 66).all().all()
    assert frame.physical_steps.eq(cfg.EPOCHS * 527).all()
    assert np.allclose(frame.sample_rate, cfg.LOGICAL_BATCH_SIZE / 67349)


def plot_metrics(frame, metrics, output):
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for metric in metrics:
        ax.plot(frame.damping, frame[metric], marker="o", label=metric)
    ax.set_xscale("log")
    ax.set_xticks(cfg.DAMPING_VALUES, [f"{v:g}" for v in cfg.DAMPING_VALUES])
    ax.set(xlabel="Damping", ylabel=metrics[0] if len(metrics) == 1 else "Gradient norm")
    ax.grid(True, alpha=.3)
    if len(metrics) > 1:
        ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(results=cfg.RESULTS):
    frame = pd.read_csv(results / "summary.csv")
    validate(frame)
    ranking = cfg.ranked(frame)
    ranking.to_csv(results / "ranking.csv", index=False)
    frame = frame.sort_values("damping")
    for metric in ("accuracy", "test_loss", "clip_fraction", "mean_clip_factor"):
        plot_metrics(frame, [metric], results / f"damping_vs_{metric}.png")
    plot_metrics(frame, ["norm_p90", "norm_p99"], results / "damping_vs_norm_tail.png")
    best = ranking.iloc[0]
    print(f"Best: damping={best.damping:g}, validation accuracy={best.accuracy:.4%}")


if __name__ == "__main__":
    analyze()

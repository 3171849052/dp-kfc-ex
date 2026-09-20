"""Validate and plot the 16 completed runs; test_loss denotes validation loss."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp28 import config as cfg


def validate(frame):
    assert len(frame) == 16
    assert not frame.duplicated(["C", "learning_rate"]).any()
    assert set(frame[["C", "learning_rate"]].itertuples(index=False, name=None)) == set(cfg.grid())
    fixed = dict(method="DP-KFC-A", geometry=cfg.GEOMETRY, source=cfg.SOURCE,
                 engine=cfg.ENGINE, seed=cfg.SEED, epoch=cfg.EPOCHS,
                 epsilon_target=cfg.EPSILON, delta=cfg.DELTA, profiled=False,
                 collect_diagnostics=True, logical_batch_size=cfg.LOGICAL_BATCH_SIZE,
                 physical_batch_size=cfg.PHYSICAL_BATCH_SIZE,
                 geometry_batch_size=cfg.GEOMETRY_BATCH_SIZE,
                 geometry_physical_batch_size=cfg.GEOMETRY_PHYSICAL_BATCH_SIZE,
                 damping=cfg.DAMPING, a_power=cfg.A_POWER, train_size=67349,
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


def heatmap(frame, metric, ax):
    values = frame.pivot(index="C", columns="learning_rate", values=metric).reindex(
        index=cfg.C_VALUES, columns=cfg.LEARNING_RATES).to_numpy()
    plot = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(4), [f"{lr:g}" for lr in cfg.LEARNING_RATES])
    ax.set_yticks(range(4), [f"{c:g}" for c in cfg.C_VALUES])
    ax.set(xlabel="Learning rate", ylabel="C", title=metric)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:.4g}", ha="center", va="center", color="white",
                bbox=dict(facecolor="black", alpha=.45, edgecolor="none", pad=1))
    ax.figure.colorbar(plot, ax=ax)


def analyze(results=cfg.RESULTS):
    frame = pd.read_csv(results / "summary.csv")
    validate(frame)
    ranking = cfg.ranked(frame)
    ranking.to_csv(results / "ranking.csv", index=False)
    for metric in ("accuracy", "test_loss", "clip_fraction", "mean_clip_factor"):
        fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
        heatmap(frame, metric, ax)
        fig.savefig(results / f"{metric}_heatmap.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, metric in zip(axes, ("norm_p90", "norm_p99")):
        heatmap(frame, metric, ax)
    fig.savefig(results / "norm_p90_p99_heatmap.png", dpi=180)
    plt.close(fig)
    best = ranking.iloc[0]
    print(f"Best: C={best.C:g}, LR={best.learning_rate:g}, validation accuracy={best.accuracy:.4%}")


if __name__ == "__main__":
    analyze()

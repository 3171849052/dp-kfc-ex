"""Rank the complete grid and plot shared-scale accuracy response surfaces."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd

from exp27 import config as cfg


def validate(frame):
    assert len(frame) == 50, "Expected all 50 completed runs"
    expected = {(method, c, lr) for c, lr, _, method, _ in cfg.grid()}
    assert not frame.duplicated(["method", "C", "learning_rate"]).any()
    assert set(frame[["method", "C", "learning_rate"]].itertuples(index=False, name=None)) == expected
    for key, value in dict(seed=cfg.SEED, epoch=cfg.EPOCHS, epsilon_target=cfg.EPSILON,
                           delta=cfg.DELTA, source=cfg.SOURCE, engine=cfg.ENGINE,
                           profiled=False).items():
        assert frame[key].eq(value).all(), key
    for _, method, geometry in cfg.METHODS:
        assert frame.loc[frame.method == method, "geometry"].eq(geometry).all()
    numeric = ["accuracy", "test_loss", "train_loss", "epsilon_spent", "noise_multiplier", "noise_std"]
    assert np.isfinite(frame[numeric].to_numpy()).all()
    assert frame.accuracy.between(0, 1).all()
    assert np.allclose(frame.noise_std, frame.noise_multiplier * frame.C)
    assert frame[list(cfg.STEP_FIELDS)].eq(cfg.EPOCHS * (60000 // cfg.BATCH_SIZE)).all().all()


def analyze(results=cfg.RESULTS):
    frame = pd.read_csv(results / "summary.csv")
    validate(frame)
    rankings = {}
    for slug, method, _ in cfg.METHODS:
        ranked = frame.loc[frame.method == method].sort_values(
            ["accuracy", "C", "learning_rate"], ascending=[False, True, True]).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.to_csv(results / f"{slug}_ranking.csv", index=False)
        rankings[method] = ranked
        best = ranked.iloc[0]
        print(f"{method}: best C={best.C:g}, lr={best.learning_rate:g}, accuracy={best.accuracy:.4%}")
    paired = frame.pivot(index=["C", "learning_rate"], columns="method", values="accuracy").rename(
        columns={"DP-KFC": "dp_kfc_accuracy", "DP-KFC-A": "dp_kfc_a_accuracy"}).reset_index()
    paired["delta_accuracy"] = paired.dp_kfc_a_accuracy - paired.dp_kfc_accuracy
    paired.to_csv(results / "paired_grid.csv", index=False)

    low, high = frame.accuracy.min() * 100, frame.accuracy.max() * 100
    if low == high:
        low, high = low - 0.01, high + 0.01
    norm = Normalize(low, high)
    levels = np.linspace(low, high, 21)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained")
    x, y = np.meshgrid(cfg.C_VALUES, cfg.LEARNING_RATES)
    for ax, (method, ranked), panel in zip(axes, rankings.items(), ("a", "b")):
        z = ranked.pivot(index="learning_rate", columns="C", values="accuracy").reindex(
            index=cfg.LEARNING_RATES, columns=cfg.C_VALUES).to_numpy() * 100
        # Contours interpolate in log10 parameter space, without extrapolation.
        ax.contourf(np.log10(x), np.log10(y), z, levels=levels, norm=norm, cmap="viridis")
        ax.scatter(np.log10(x), np.log10(y), s=18, c="black", alpha=0.65, clip_on=False)
        best = ranked.iloc[0]
        bx, by = np.log10(best.C), np.log10(best.learning_rate)
        ax.scatter(bx, by, marker="*", s=230, c="white", edgecolors="black", linewidths=0.8,
                   zorder=5, clip_on=False)
        right = best.C >= 1
        top = best.learning_rate >= 1e-3
        ax.annotate(f"{best.accuracy:.2%}", (bx, by), xytext=(-10 if right else 10, -18 if top else 12),
                    textcoords="offset points", ha="right" if right else "left", color="white",
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2))
        for axis, values in ((ax.xaxis, cfg.C_VALUES), (ax.yaxis, cfg.LEARNING_RATES)):
            axis.set_major_locator(FixedLocator(np.log10(values)))
            axis.set_major_formatter(FixedFormatter([f"{v:g}" for v in values]))
        ax.set_title(f"({panel}) {method}")
        ax.set_xlabel("Clip Norm (log scale)")
        ax.set_ylabel("Learning Rate (log scale)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, label="Test accuracy (%)")
    fig.savefig(results / "accuracy_surface.png", dpi=300)
    fig.savefig(results / "accuracy_surface.pdf")
    plt.close(fig)


if __name__ == "__main__":
    analyze()

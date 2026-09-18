"""Rank epoch-three validation results and plot the fixed coarse grid."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp26 import config as cfg


def heatmap(rows, method, metric):
    grid = rows.pivot(index="C", columns="learning_rate", values=metric).reindex(
        index=cfg.CLIP_NORMS, columns=cfg.LEARNING_RATES
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(grid.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cfg.LEARNING_RATES)), [f"{lr:g}" for lr in cfg.LEARNING_RATES])
    ax.set_yticks(range(len(cfg.CLIP_NORMS)), [f"{c:g}" for c in cfg.CLIP_NORMS])
    ax.set_xlabel("Adam learning rate")
    ax.set_ylabel("Clipping norm C")
    ax.set_title(f"{method}: epoch {cfg.EPOCHS} {metric} (seed={cfg.SEED})")
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            value = grid.iloc[i, j]
            ax.text(j, i, f"{value:.4f}", ha="center", va="center",
                    color="white" if value < 0.5 else "black")
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS / f"{method}_{metric}_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    summary = pd.read_csv(cfg.RESULTS / "summary.csv")
    assert len(summary) == len(cfg.METHODS) * len(cfg.CLIP_NORMS) * len(cfg.LEARNING_RATES)
    assert (summary["epoch"] == cfg.EPOCHS).all()
    assert (summary["seed"] == cfg.SEED).all()
    rankings = []
    for method in cfg.METHODS:
        rows = summary.loc[summary["method"] == method].sort_values(
            ["accuracy", "test_loss", "C", "learning_rate"],
            ascending=[False, True, True, True],
        ).copy()
        assert set(zip(rows["C"], rows["learning_rate"])) == {
            (c, lr) for c in cfg.CLIP_NORMS for lr in cfg.LEARNING_RATES
        }
        assert rows[["accuracy", "test_loss", "clip_fraction"]].notna().all().all()
        rows.insert(0, "rank", range(1, len(rows) + 1))
        rows.to_csv(cfg.RESULTS / f"{method}_ranking.csv", index=False)
        rankings.append(rows)
        best = rows.iloc[0]
        print(f"{method}: best (C, LR)=({best['C']:g}, {best['learning_rate']:g}), "
              f"validation accuracy={best['accuracy']:.6f}, loss={best['test_loss']:.6f}")
        for metric in ("accuracy", "clip_fraction"):
            heatmap(rows, method, metric)
    pd.concat(rankings, ignore_index=True).to_csv(cfg.RESULTS / "ranking.csv", index=False)


if __name__ == "__main__":
    main()

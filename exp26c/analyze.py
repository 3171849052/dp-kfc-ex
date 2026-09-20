"""Rank final validation results and plot damping versus utility/diagnostics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp26c import config as cfg


def plot(rows, metrics, ylabel, filename, reference_lines=()):
    fig, ax = plt.subplots(figsize=(7, 5))
    for field, label in metrics:
        ax.plot(rows["damping"], rows[field], marker="o", label=label)
    for value, label, color in reference_lines:
        ax.axhline(value, linestyle="--", color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(cfg.DAMPINGS, [f"{d:g}" for d in cfg.DAMPINGS])
    ax.set_xlabel("Damping")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Full DP-KFC: epoch {cfg.EPOCHS}, seed {cfg.SEED}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.RESULTS / filename, dpi=180)
    plt.close(fig)


def main():
    rows = pd.read_csv(cfg.RESULTS / "summary.csv").sort_values("damping")
    assert rows["damping"].tolist() == cfg.DAMPINGS
    assert (rows["epoch"] == cfg.EPOCHS).all()
    assert (rows["seed"] == cfg.SEED).all()
    assert rows[["accuracy", "test_loss", "train_loss", "norm_p90", "norm_p99",
                 "clip_fraction"]].notna().all().all()
    rows["p90_ratio_to_dp_adam"] = rows["norm_p90"] / cfg.DP_ADAM_NORM_P90
    rows["p99_ratio_to_dp_adam"] = rows["norm_p99"] / cfg.DP_ADAM_NORM_P99
    ranking = rows.sort_values(["accuracy", "test_loss", "damping"], ascending=[False, True, True]).copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking.to_csv(cfg.RESULTS / "ranking.csv", index=False)
    best = ranking.iloc[0]
    print(f"Best damping={best['damping']:g}, validation accuracy={best['accuracy']:.6f}, "
          f"validation loss={best['test_loss']:.6f}")
    plot(rows, [("accuracy", "Validation accuracy")], "Accuracy", "damping_vs_accuracy.png",
         [(cfg.DP_ADAM_ACCURACY, "DP-Adam accuracy (Exp26)", "black")])
    plot(rows, [("train_loss", "Train loss"), ("test_loss", "Validation loss")],
         "Loss", "damping_vs_loss.png")
    plot(rows, [("norm_p90", "Norm p90"), ("norm_p99", "Norm p99")],
         "Preconditioned gradient norm", "damping_vs_norm_tail.png",
         [(cfg.DP_ADAM_NORM_P90, "DP-Adam p90 (Exp26)", "C0"),
          (cfg.DP_ADAM_NORM_P99, "DP-Adam p99 (Exp26)", "C1")])
    plot(rows, [("clip_fraction", "Clip fraction")], "Clip fraction", "damping_vs_clip_fraction.png")


if __name__ == "__main__":
    main()

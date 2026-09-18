"""Process and plot the formal CNN-MNIST-A experiment results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPSILON = 3.0
PINK_CONDITION_ORDER = [
    ("base", "none", "DP-Adam"),
    ("full", "pink", "DP-KFC-Pink"),
    ("a_only", "pink", "DP-KFC-A-Pink"),
]
CLIPPING_ORDER = [
    ("match", "full", "Match Full"),
    ("match", "a_only", "Match A-only"),
    ("mismatch", "full", "Mismatch Full"),
    ("mismatch", "a_only", "Mismatch A-only"),
    ("pink", "full", "Pink Full"),
    ("pink", "a_only", "Pink A-only"),
]

REQUIRED_COLUMNS = {
    "geometry",
    "source",
    "engine",
    "profiled",
    "epsilon_target",
    "seed",
    "test_loss",
    "accuracy",
    "private_train_seconds",
    "private_peak_allocated_bytes",
    "clip_fraction",
    "mean_clip_factor",
    "norm_p90",
    "norm_p99",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
})


def summarize_utility(results):
    """Summarize accuracy and test loss over BK seeds."""
    utility = results[results["engine"] == "bk"]
    summary = (
        utility.groupby(["epsilon_target", "geometry", "source"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            test_loss_mean=("test_loss", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["epsilon_target", "geometry", "source"])
        .reset_index(drop=True)
    )
    summary["accuracy_mean"] *= 100
    summary["accuracy_std"] *= 100
    return summary


def paired_efficiency(results):
    """Pair BK and Explicit profiling rows, then summarize seed-level metrics."""
    profiled = results[
        (results["epsilon_target"] == EPSILON)
        & (results["profiled"] == True)
        & results["engine"].isin(["bk", "explicit"])
    ]
    keys = ["geometry", "source", "seed"]
    bk = (
        profiled[profiled["engine"] == "bk"]
        [keys + ["private_train_seconds", "private_peak_allocated_bytes"]]
        .rename(columns={
            "private_train_seconds": "bk_private_train_seconds",
            "private_peak_allocated_bytes": "bk_private_peak_allocated_bytes",
        })
    )
    explicit = (
        profiled[profiled["engine"] == "explicit"]
        [keys + ["private_train_seconds", "private_peak_allocated_bytes"]]
        .rename(columns={
            "private_train_seconds": "explicit_private_train_seconds",
            "private_peak_allocated_bytes": "explicit_private_peak_allocated_bytes",
        })
    )
    paired = bk.merge(explicit, on=keys, how="inner", validate="one_to_one")
    paired["speedup"] = (
        paired["explicit_private_train_seconds"]
        / paired["bk_private_train_seconds"]
    )
    paired["memory_saving_pct"] = 100 * (
        1
        - paired["bk_private_peak_allocated_bytes"]
        / paired["explicit_private_peak_allocated_bytes"]
    )
    paired = paired[keys + [
        "bk_private_train_seconds",
        "explicit_private_train_seconds",
        "speedup",
        "bk_private_peak_allocated_bytes",
        "explicit_private_peak_allocated_bytes",
        "memory_saving_pct",
    ]].sort_values(keys).reset_index(drop=True)

    summary = (
        paired.groupby(["geometry", "source"], as_index=False)
        .agg(
            speedup_mean=("speedup", "mean"),
            speedup_std=("speedup", "std"),
            memory_saving_pct_mean=("memory_saving_pct", "mean"),
            memory_saving_pct_std=("memory_saving_pct", "std"),
            seed_count=("seed", "nunique"),
        )
    )
    return paired, summary


def summarize_memory_saving(paired):
    """Collapse source-equivalent memory measurements to one value per seed."""
    per_seed = (
        paired.groupby(["geometry", "seed"], as_index=False)
        .agg(memory_saving_pct=("memory_saving_pct", "mean"))
    )
    return (
        per_seed.groupby("geometry", as_index=False)
        .agg(
            memory_saving_pct_mean=("memory_saving_pct", "mean"),
            memory_saving_pct_std=("memory_saving_pct", "std"),
            seed_count=("seed", "nunique"),
        )
        .sort_values("geometry")
        .reset_index(drop=True)
    )


def summarize_clipping(results):
    """Summarize clipping diagnostics for profiled BK runs at epsilon=3."""
    clipping = results[
        (results["epsilon_target"] == EPSILON)
        & (results["engine"] == "bk")
        & (results["profiled"] == True)
        & results["geometry"].isin(["full", "a_only"])
        & results["source"].isin(["match", "mismatch", "pink"])
    ]
    return (
        clipping.groupby(["geometry", "source"], as_index=False)
        .agg(
            clip_fraction_mean=("clip_fraction", "mean"),
            clip_fraction_std=("clip_fraction", "std"),
            mean_clip_factor_mean=("mean_clip_factor", "mean"),
            mean_clip_factor_std=("mean_clip_factor", "std"),
            norm_p90_mean=("norm_p90", "mean"),
            norm_p99_mean=("norm_p99", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["geometry", "source"])
        .reset_index(drop=True)
    )


def build_proxy_gap_table(utility_summary):
    """Build accuracy gaps in percentage points from the utility means."""
    means = utility_summary.set_index(
        ["epsilon_target", "geometry", "source"]
    )["accuracy_mean"]
    rows = []
    for epsilon in sorted(utility_summary["epsilon_target"].unique()):
        for geometry in ("full", "a_only"):
            match = means.loc[(epsilon, geometry, "match")]
            mismatch = means.loc[(epsilon, geometry, "mismatch")]
            pink = means.loc[(epsilon, geometry, "pink")]
            rows.append({
                "epsilon_target": epsilon,
                "geometry": geometry,
                "match_minus_mismatch_pp": match - mismatch,
                "pink_minus_mismatch_pp": pink - mismatch,
                "match_minus_pink_pp": match - pink,
            })
    return pd.DataFrame(rows)


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_utility_condition(ax, utility_summary, geometry, source, label):
    condition = utility_summary[
        (utility_summary["geometry"] == geometry)
        & (utility_summary["source"] == source)
    ].sort_values("epsilon_target")
    ax.errorbar(
        condition["epsilon_target"],
        condition["accuracy_mean"],
        yerr=condition["accuracy_std"],
        fmt="-o",
        linewidth=1.2,
        markersize=3.5,
        capsize=2,
        capthick=0.8,
        label=label,
    )


def plot_utility(utility_summary, geometry, output_dir):
    """Plot utility curves for Full or A-only geometry."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    _plot_utility_condition(ax, utility_summary, "base", "none", "DP-Adam")
    for source in ("match", "mismatch", "pink"):
        _plot_utility_condition(
            ax, utility_summary, geometry, source,
            f"{'Full' if geometry == 'full' else 'A-only'} / {source.title()}",
        )
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(sorted(utility_summary["epsilon_target"].unique()))
    ax.legend(frameon=False, ncol=2, loc="best")
    _style_axes(ax)
    _save_figure(fig, output_dir, f"utility_{geometry}")


def plot_proxy_gaps(proxy_gap_summary, output_dir):
    """Plot proxy mismatch and Pink recovery gaps."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for geometry, label in (
        ("full", "Full"),
        ("a_only", "A-only"),
    ):
        condition = proxy_gap_summary[
            proxy_gap_summary["geometry"] == geometry
        ].sort_values("epsilon_target")
        ax.plot(
            condition["epsilon_target"],
            condition["match_minus_mismatch_pp"],
            "-o",
            linewidth=1.2,
            markersize=3.5,
            label=f"{label}: Match - Mismatch",
        )
        ax.plot(
            condition["epsilon_target"],
            condition["pink_minus_mismatch_pp"],
            "-o",
            linewidth=1.2,
            markersize=3.5,
            label=f"{label}: Pink - Mismatch",
        )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy difference (percentage points)")
    ax.set_xticks(sorted(proxy_gap_summary["epsilon_target"].unique()))
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
    )
    _style_axes(ax)
    _save_figure(fig, output_dir, "proxy_gap")


def _ordered_summary(summary, order=PINK_CONDITION_ORDER):
    order_index = {(geometry, source): index for index, (geometry, source, _) in enumerate(order)}
    ordered = summary.copy()
    ordered["_order"] = [order_index[(g, s)] for g, s in zip(ordered["geometry"], ordered["source"])]
    return ordered.sort_values("_order").reset_index(drop=True)


def plot_speedup(efficiency_summary, output_dir):
    """Plot Explicit/BK speedup for DP-Adam and the two Pink variants."""
    condition = efficiency_summary[
        ((efficiency_summary["geometry"] == "base")
         & (efficiency_summary["source"] == "none"))
        | ((efficiency_summary["geometry"].isin(["full", "a_only"]))
           & (efficiency_summary["source"] == "pink"))
    ]
    condition = _ordered_summary(condition)
    labels = [label for _, _, label in PINK_CONDITION_ORDER]
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    ax.bar(
        np.arange(len(condition)),
        condition["speedup_mean"],
        yerr=condition["speedup_std"],
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.axhline(1, color="0.5", linewidth=0.8)
    ax.set_xticks(np.arange(len(condition)), labels, rotation=30, ha="right")
    ax.set_ylabel("Speedup (Explicit / BK)")
    _style_axes(ax)
    _save_figure(fig, output_dir, "bk_speedup_eps3")


def plot_memory_saving(memory_summary, output_dir):
    """Plot source-independent BK peak allocated memory savings at epsilon=3."""
    order = ["base", "full", "a_only"]
    labels = ["DP-Adam", "Full", "A-only"]
    condition = memory_summary.set_index("geometry").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    ax.bar(
        np.arange(len(condition)),
        condition["memory_saving_pct_mean"],
        yerr=condition["memory_saving_pct_std"],
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.set_xticks(np.arange(len(condition)), labels, rotation=30, ha="right")
    ax.set_ylabel("Peak allocated memory saved by BK (%)")
    _style_axes(ax)
    _save_figure(fig, output_dir, "bk_memory_saving_eps3")


def plot_pink_utility(utility_summary, output_dir):
    """Plot utility for DP-Adam and the two Pink KFC variants."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    _plot_utility_condition(ax, utility_summary, "base", "none", "DP-Adam")
    _plot_utility_condition(ax, utility_summary, "full", "pink", "DP-KFC-Pink")
    _plot_utility_condition(ax, utility_summary, "a_only", "pink", "DP-KFC-A-Pink")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(sorted(utility_summary["epsilon_target"].unique()))
    ax.legend(frameon=False, ncol=1, loc="best")
    _style_axes(ax)
    _save_figure(fig, output_dir, "utility_pink_comparison")


def plot_clipping(clipping_summary, output_dir):
    """Plot clipped sample fractions at epsilon=3."""
    order_index = {(source, geometry): index for index, (source, geometry, _) in enumerate(CLIPPING_ORDER)}
    condition = clipping_summary.copy()
    condition["_order"] = [order_index[(s, g)] for g, s in zip(condition["geometry"], condition["source"])]
    condition = condition.sort_values("_order")
    labels = [label for _, _, label in CLIPPING_ORDER]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.bar(
        np.arange(len(condition)),
        100 * condition["clip_fraction_mean"],
        yerr=100 * condition["clip_fraction_std"],
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.set_xticks(np.arange(len(condition)), labels, rotation=30, ha="right")
    ax.set_ylabel("Clipped samples (%)")
    _style_axes(ax)
    _save_figure(fig, output_dir, "clipping_fraction_eps3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/cnn_mnist_a/summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/cnn_mnist_a/plots"),
    )
    args = parser.parse_args()

    results = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utility_summary = summarize_utility(results)
    proxy_gap_summary = build_proxy_gap_table(utility_summary)
    efficiency_paired, efficiency_summary = paired_efficiency(results)
    memory_summary = summarize_memory_saving(efficiency_paired)
    clipping_summary = summarize_clipping(results)

    utility_summary.to_csv(args.output_dir / "utility_summary.csv", index=False)
    proxy_gap_summary.to_csv(args.output_dir / "proxy_gap_summary.csv", index=False)
    efficiency_paired.to_csv(args.output_dir / "efficiency_paired_eps3.csv", index=False)
    efficiency_summary.to_csv(args.output_dir / "efficiency_summary_eps3.csv", index=False)
    clipping_summary.to_csv(args.output_dir / "clipping_summary_eps3.csv", index=False)

    plot_utility(utility_summary, "full", args.output_dir)
    plot_utility(utility_summary, "a_only", args.output_dir)
    plot_proxy_gaps(proxy_gap_summary, args.output_dir)
    plot_speedup(efficiency_summary, args.output_dir)
    plot_memory_saving(memory_summary, args.output_dir)
    plot_pink_utility(utility_summary, args.output_dir)
    plot_clipping(clipping_summary, args.output_dir)

    print("Utility summary:")
    print(utility_summary.to_string(index=False))
    print("\nEpsilon=3 speedup mean:")
    pink_efficiency = efficiency_summary[
        ((efficiency_summary["geometry"] == "base")
         & (efficiency_summary["source"] == "none"))
        | ((efficiency_summary["geometry"].isin(["full", "a_only"]))
           & (efficiency_summary["source"] == "pink"))
    ]
    print(
        _ordered_summary(pink_efficiency)[
            ["geometry", "source", "speedup_mean"]
        ].to_string(index=False)
    )
    print("\nEpsilon=3 memory saving mean:")
    print(memory_summary.to_string(index=False))


if __name__ == "__main__":
    main()

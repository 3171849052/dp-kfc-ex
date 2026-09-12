"""Plots and explicit integrity checks for exp10b."""
from itertools import combinations
import numpy as np
from exp10.run_exp10 import plt
from exp10b.config import METHODS, STRUCTURED, LAYERS


def report(frame, final, approximation, output, epochs, seeds, smoke=False):
    assert len(frame) == len(METHODS)*len(seeds)*epochs
    assert not frame.duplicated(["method", "seed", "epoch"]).any()
    assert set(zip(final.method, final.seed)) == {(m, s) for m in METHODS for s in seeds}
    assert (final.epoch == epochs).all()
    numeric = frame.select_dtypes(include="number").drop(columns=["log_scale_pearson", "log_scale_spearman"])
    if smoke:
        numeric = numeric.drop(columns="epsilon_spent")  # sigma=0 intentionally has infinite epsilon.
    assert np.isfinite(numeric.to_numpy()).all()
    for table in (frame, approximation):
        defined = table.correlation_defined
        assert table.loc[defined, ["log_scale_pearson", "log_scale_spearman"]].notna().all().all()
        assert table.loc[~defined, ["log_scale_pearson", "log_scale_spearman"]].isna().all().all()
        assert (table.gain_min > 0).all()
    assert len(approximation) == len(seeds)*epochs*50  # 1 full + 3 sources x 3 projections, 5 scopes
    assert not approximation.duplicated(["source_method", "method", "seed", "epoch", "module"]).any()
    for _, group in approximation[approximation.source_method.isin(STRUCTURED)].groupby(
            ["source_method", "seed", "epoch"]):
        assert group.statistic_sha256.nunique() == group.target_sha256.nunique() == 1
        assert set(group.method) == set(STRUCTURED)
    global_rows = approximation[approximation.module == "__global__"]
    assert np.allclose(global_rows.gain_geomean, 1, atol=2e-7, rtol=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(METHODS):
        part = frame[frame.method == method]
        for _, run in part.groupby("seed"):
            ax.plot(run.epoch, run.test_accuracy*100, color=f"C{i}", alpha=.25)
        mean = part.groupby("epoch").test_accuracy.mean()
        ax.plot(mean.index, mean*100, marker="o", color=f"C{i}", label=method)
    ax.set(xlabel="Epoch", ylabel="Test accuracy (%)", xticks=range(1, epochs+1))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in STRUCTURED:
        part = final[final.method == method]
        std = part.test_accuracy.std(ddof=1) if len(seeds)>1 else 0.
        ax.errorbar(part.operator_dof.iloc[0], part.test_accuracy.mean()*100,
                    yerr=std*100, fmt="o", capsize=4, label=method)
    ax.set(xscale="log", xlabel="Effective operator DOF (global GM constraint removed)",
           ylabel="Final test accuracy (%)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"complexity_vs_accuracy.png", dpi=160)
    plt.close(fig)

    # Compare projections on the SAME module-elementwise trajectory, not different models.
    ref = approximation[(approximation.source_method == STRUCTURED[0]) &
                        (approximation.module != "__global__")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("log_scale_rmse", "log_scale_pearson", "log_scale_spearman")):
        for method in STRUCTURED:
            part = ref[ref.method == method].groupby("module")[metric].mean().reindex(LAYERS)
            ax.plot(range(len(LAYERS)), part, marker="o", label=method)
        ax.set(xticks=range(len(LAYERS)), xticklabels=LAYERS, ylabel=metric)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same module-elementwise targets; mean over seeds/epochs. Scalar correlations undefined.")
    fig.tight_layout()
    fig.savefig(output/"approximation_by_layer.png", dpi=160)
    plt.close(fig)

    test_log = output.parent/"tests.log" if smoke else output/"tests.log"
    assert "\nOK\n" in test_log.read_text(), "Quick tests did not pass"
    if not smoke:
        assert "Integrity checks: PASS" in (output/"smoke/verification.txt").read_text()
    from PIL import Image
    for filename in ("accuracy.png", "complexity_vs_accuracy.png", "approximation_by_layer.png"):
        with Image.open(output/filename) as image:
            image.load()
            assert image.width >= 800 and image.height >= 500
            assert np.asarray(image).std() > 1
    lines = ["exp10b verification", "Integrity checks: PASS",
             f"Completed: {len(final)}/{len(METHODS)*len(seeds)} runs; {epochs} epochs each.",
             "Quick tests: " + ("PASS" if (output.parent/"tests.log" if smoke else output/"tests.log").exists()
                                else "See separate test command/log."),
             "Final accuracy mean +/- sample std (percentage points):"]
    for method in METHODS:
        part = final[final.method == method]
        std = part.test_accuracy.std(ddof=1) if len(seeds)>1 else 0.
        lines.append(f"  {method}: {part.test_accuracy.mean()*100:.4f} +/- {std*100:.4f}")
    paired = final.pivot(index="seed", columns="method", values="test_accuracy")
    lines.append("Paired final accuracy differences (first minus second, percentage points):")
    for a, b in combinations(STRUCTURED, 2):
        delta = (paired[a]-paired[b])*100
        lines.append(f"  {a} - {b}: {delta.to_dict()}; mean={delta.mean():+.4f}")
    lines.append("Effective operator DOF (subtract global GM=1 constraint):")
    for method in STRUCTURED:
        lines.append(f"  {method}: {int(final[final.method == method].operator_dof.iloc[0])}")
    lines.append("Approximation on shared Module-Elementwise trajectory (global mean over builds):")
    ref_global = global_rows[global_rows.source_method == STRUCTURED[0]]
    for method in STRUCTURED:
        part = ref_global[ref_global.method == method]
        lines.append(f"  {method}: RMSE={part.log_scale_rmse.mean():.6f}, "
                     f"Pearson={part.log_scale_pearson.mean():.6f}, Spearman={part.log_scale_spearman.mean():.6f}")
    lines.extend([
        f"Maximum global GM error: {abs(global_rows.gain_geomean-1).max():.3g}",
        f"Applied gain range: [{frame.gain_min.min():.6g}, {frame.gain_max.max():.6g}]; "
        f"maximum log10(max/min)={frame.log10_gain_range.max():.6f}",
        "NaN/Inf or training failure: none in required finite metrics/model parameters.",
        "Undefined constant-vector correlations are blank by definition (Layer-Scalar per module; DP-SGD).",
        "sigma=0 smoke epsilon is +Inf by definition." if smoke else
        "Full experiment epsilon values are finite; accounting follows exp10 unchanged.",
        "Gain anomaly flag (range > 6 decades): " + str(bool((frame.log10_gain_range > 6).any())),
        "Build time includes same-target approximation diagnostics; private train time excludes diagnostics.",
        "All three projections share identical statistic/target hashes within each source model build.",
    ])
    (output/"verification.txt").write_text("\n".join(lines)+"\n")

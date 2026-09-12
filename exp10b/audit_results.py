"""Final audit of persisted full-run results; no training or RNG changes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from exp10b import config as cfg

output = ROOT/"exp10b/results"
metrics = pd.read_csv(output/"metrics.csv")
approx = pd.read_csv(output/"operator_approximation.csv")
summary = pd.read_csv(output/"summary.csv")
assert len(metrics) == 50 and len(summary) == 10 and len(approx) == 500
assert "28 passed" in (output/"standalone_tests.log").read_text()
assert "\nOK\n" in (output/"tests.log").read_text()
assert "Integrity checks: PASS" in (output/"smoke/verification.txt").read_text()
for frame in (metrics, approx):
    numeric = frame.select_dtypes(include="number").drop(columns=["log_scale_pearson", "log_scale_spearman"])
    assert np.isfinite(numeric.to_numpy()).all()
    corr = frame.loc[frame.correlation_defined, ["log_scale_pearson", "log_scale_spearman"]]
    assert np.isfinite(corr.to_numpy()).all()
assert metrics.test_accuracy.between(0, 1).all()
assert metrics.clip_fraction.between(0, 1).all()
assert metrics.mean_clip_factor.between(0, 1).all()
assert metrics.epsilon_spent.max() <= cfg.EPSILON
assert (summary.epoch == cfg.EPOCHS).all()
for seed in cfg.SEEDS:
    initial = approx[(approx.seed == seed) & (approx.epoch == 1) &
                     approx.source_method.isin(cfg.STRUCTURED)]
    assert initial.statistic_sha256.nunique() == initial.target_sha256.nunique() == 1
    assert set(initial.source_method) == set(cfg.STRUCTURED)
for filename in ("accuracy.png", "complexity_vs_accuracy.png", "approximation_by_layer.png"):
    assert (output/filename).stat().st_size > 10000
text = (output/"verification.txt").read_text().split("\nPersisted-results audit:")[0].rstrip()
text += ("\nPersisted-results audit: PASS; 7 numerical tests + 28 standalone regression tests + "
         "five-method sigma=0 smoke passed.\n"
         "All 500 approximation rows have finite non-correlation diagnostics; undefined correlations are intentional.\n"
         "Epoch-1 statistic/target hashes match across all three structured training methods for both seeds.\n"
         f"Final epsilon: {summary.epsilon_spent.min():.8f} to {summary.epsilon_spent.max():.8f}.\n"
         "Existing exp9/data cache absent: used the repository's exp1/data MNIST cache, with identical preprocessing.\n")
(output/"verification.txt").write_text(text)
print(text)

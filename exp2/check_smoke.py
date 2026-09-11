"""Check the tiny run's schedule, paired sensors, privacy labels and artifacts."""
from pathlib import Path
import pandas as pd
import numpy as np
from run_exp2 import METHODS, LAYERS, interval_ends

output = Path(__file__).resolve().parent / "results/smoke"
m = pd.read_csv(output / "metrics.csv")
c = pd.read_csv(output / "controller_metrics.csv")
assert interval_ends(234, 4) == [58, 117, 175, 234]
assert interval_ends(234, 1) == [234]
assert len(m) == 32 and len(c) == 80
assert set(m.method) == set(c.method) == set(METHODS)
for method in METHODS:
    frequency = 4 if "Fast" in method else 1
    part = c[c.method == method]
    assert set(part.sensor_type) == {"clean" if "Clean" in method else "noisy"}
    assert set(part.update_frequency) == {frequency}
    assert (m[m.method == method].privacy_valid == ("Noisy" in method)).all()
    for layer in LAYERS:
        p = part[part.layer == layer]
        assert p.global_update_index.tolist() == list(range(1, 2*frequency+1))
        for epoch in (1, 2):
            e = p[p.epoch == epoch]
            assert e.interval.tolist() == list(range(1, frequency+1))
            assert e.interval_steps.tolist() == ([1, 2, 1, 2] if frequency == 4 else [6])
for side in ("A", "G"):
    fields = [f"NSR_{side}", f"cos_{side}", f"clean_{side}_norm", f"noisy_{side}_norm"]
    assert np.isfinite(c[fields].to_numpy()).all()
    assert (c[f"NSR_{side}"] >= 0).all()
    assert (c[f"cos_{side}"].abs() <= 1+1e-12).all()
    assert (c[[f"clean_{side}_norm", f"noisy_{side}_norm"]] > 0).all().all()
# Slow epoch 1 uses identity throughout, regardless of the pending sensor update.
# Exact equality also checks that clean shadow observation does not perturb RNG.
cols = [x for x in m.columns if x not in ("method", "privacy_valid")]
slow = [m[(m.method == method) & (m.epoch == 1)][cols].reset_index(drop=True)
        for method in ("CLW-Noisy-Slow", "CLW-Clean-Slow")]
pd.testing.assert_frame_equal(*slow)
for filename in ("summary.csv", "whitening_by_layer.png", "factor_whitening_fc.png",
                 "accuracy.png", "feedback_nsr.png", "feedback_cosine.png"):
    assert (output / filename).stat().st_size > 0
print("PASS: 32 metric rows, 80 feedback rows; four methods, interval counts, paired quality, privacy labels and all artifacts")

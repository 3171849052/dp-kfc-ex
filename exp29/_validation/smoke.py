"""CPU-only protocol, orchestration and plotting smoke checks; no training."""
import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from exp29 import config as cfg
from exp29 import run
from exp29 import analyze

root = Path(__file__).resolve().parent
for path in root.parent.rglob("*.py"):
    ast.parse(path.read_text())
assert tuple(cfg.grid()) == (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
assert len({cfg.run_dir(d) for d in cfg.grid()}) == 5
assert all(cfg.run_dir(d).is_relative_to(cfg.RESULTS) for d in cfg.grid())
constants = ("MAX_GRAD_NORM", "DELTA", "EPOCHS", "LOGICAL_BATCH_SIZE", "BK_PHYSICAL_BATCH_SIZE",
             "GEOMETRY_BATCH_SIZE", "GEOMETRY_PHYSICAL_BATCH_SIZE", "DAMPING", "A_POWER", "MAX_LENGTH")
original = {key: getattr(run.reference, key) for key in constants}
for damping in cfg.grid():
    with run.reference_constants(damping):
        actual = run.reference.matrix_power(torch.zeros(2, 2, dtype=torch.float64), cfg.A_POWER)
        expected = torch.eye(2, dtype=torch.float64) * (damping + 1e-5) ** (-cfg.A_POWER)
        torch.testing.assert_close(actual, expected)
assert all(getattr(run.reference, k) == v for k, v in original.items())
calls = []
raw_paths = []


def fake(**kwargs):
    damping = run.reference.DAMPING
    calls.append(damping)
    expected = dict(geometry="a_only", source="synthetic", engine="bk", epsilon=3,
                    seed=42, epochs=3, physical_batch_size=128, profile=False,
                    collect_diagnostics=True, lr=5e-4, profile_mode="practical")
    for key, value in expected.items():
        assert kwargs[key] == value, key
    assert run.reference.MAX_GRAD_NORM == 2
    assert run.reference.BK_PHYSICAL_BATCH_SIZE == 128
    for key in constants:
        if key not in ("MAX_GRAD_NORM", "BK_PHYSICAL_BATCH_SIZE", "DAMPING"):
            assert getattr(run.reference, key) == getattr(cfg, key), key
    assert kwargs["output_dir"] == cfg.run_dir(damping)
    raw = kwargs["output_dir"] / "a_only_synthetic_bk_practical_eps3_seed42.csv"
    raw.write_text("epoch,accuracy\n1,\n2,\n3,0.5\n")
    raw_paths.append(raw)
    row = dict(method="DP-KFC-A", geometry="a_only", source="synthetic", engine="bk",
               seed=42, epoch=3, epsilon_target=3, delta=1e-5, profiled=False,
               logical_batch_size=1024, physical_batch_size=128,
               geometry_batch_size=256, geometry_physical_batch_size=16,
               train_size=67349, validation_size=872, sample_rate=1024/67349,
               physical_steps=1581, noise_multiplier=1.2,
               **{key: 198 for key in cfg.STEP_FIELDS})
    row.update({key: .5 for key in cfg.METRICS if key not in row})
    return row


results = root / "results"
# Run the real orchestration with data/model training replaced by mocks.
with patch.object(cfg, "RESULTS", results), \
     patch.object(run.torch.cuda, "is_available", return_value=True), \
     patch.object(run.reference.AutoTokenizer, "from_pretrained", return_value=None), \
     patch.object(run.reference, "load_data", return_value=(range(67349), range(872))), \
     patch.object(run.reference, "run_one", side_effect=fake):
    run.main()
assert calls == list(cfg.DAMPING_VALUES)
assert all(getattr(run.reference, k) == v for k, v in original.items())
assert all(p.read_text() == "epoch,accuracy\n1,\n2,\n3,0.5\n" for p in raw_paths)
frame = pd.read_csv(results / "summary.csv")
assert np.allclose(frame.effective_damping, frame.damping + 1e-5)
assert np.allclose(frame.noise_std, 2.4)
# Exercise accuracy, loss and damping tie-breaks independently.
ties = pd.DataFrame(dict(accuracy=[.8, .9, .9, .9], test_loss=[.1, .3, .2, .2],
                         damping=[.001, .003, .1, .01]))
assert cfg.ranked(ties).damping.tolist() == [.01, .1, .003, .001]
from matplotlib.axes import Axes
original_scale = Axes.set_xscale
scales = []


def set_scale(self, value, **kwargs):
    scales.append(value)
    return original_scale(self, value, **kwargs)


with patch.object(Axes, "set_xscale", set_scale):
    analyze.analyze(results)
assert scales == ["log"] * 5
assert len(list(results.glob("damping_vs_*.png"))) == 5
with patch.object(cfg, "RESULTS", results), patch.object(run.reference, "run_one", side_effect=RuntimeError("smoke")):
    try:
        run.run_one(range(67349), range(872), None, .1, "cpu")
    except RuntimeError:
        pass
assert all(getattr(run.reference, k) == v for k, v in original.items())
print("PASS: five runs, protocol/restoration, effective damping, raw CSV preservation, ranking and five log-scale plots")

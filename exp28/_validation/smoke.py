"""Lightweight contract and plotting checks; never trains a model."""
import ast
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from exp28 import config as cfg
from exp28 import run
from exp28.analyze import analyze

root = Path(__file__).resolve().parent
for path in (root.parent).glob("*.py"):
    ast.parse(path.read_text())
assert len(list(cfg.grid())) == len(set(cfg.grid())) == 16
assert len({cfg.run_dir(c, lr) for c, lr in cfg.grid()}) == 16
assert all(path.is_relative_to(cfg.RESULTS) for path in [cfg.run_dir(c, lr) for c, lr in cfg.grid()])
original = {key: getattr(run.reference, key) for key in (
    "MAX_GRAD_NORM", "DELTA", "EPOCHS", "LOGICAL_BATCH_SIZE", "BK_PHYSICAL_BATCH_SIZE",
    "GEOMETRY_BATCH_SIZE", "GEOMETRY_PHYSICAL_BATCH_SIZE", "DAMPING", "A_POWER", "MAX_LENGTH")}
rows = []


def fake(**kwargs):
    assert kwargs["geometry"] == "a_only" and kwargs["source"] == "synthetic"
    assert kwargs["engine"] == "bk" and kwargs["seed"] == 42
    assert kwargs["epsilon"] == 3 and kwargs["epochs"] == 3
    assert kwargs["physical_batch_size"] == 128
    assert kwargs["profile"] is False and kwargs["collect_diagnostics"] is True
    assert run.reference.MAX_GRAD_NORM == c
    for name in ("DELTA", "EPOCHS", "LOGICAL_BATCH_SIZE", "GEOMETRY_BATCH_SIZE",
                 "GEOMETRY_PHYSICAL_BATCH_SIZE", "DAMPING", "A_POWER", "MAX_LENGTH"):
        assert getattr(run.reference, name) == getattr(cfg, name)
    row = dict(method="DP-KFC-A", geometry="a_only", source="synthetic", engine="bk",
               seed=42, epoch=3, epsilon_target=3, delta=1e-5, profiled=False,
               logical_batch_size=1024, physical_batch_size=128,
               geometry_batch_size=256, geometry_physical_batch_size=16,
               train_size=67349, validation_size=872, sample_rate=1024/67349,
               physical_steps=1581, noise_multiplier=1.2,
               **{key: 198 for key in cfg.STEP_FIELDS})
    row.update({key: .5 for key in cfg.METRICS if key not in row})
    row["test_loss"] = 1 / c  # Exercise loss tie-breaking, then C/LR ties.
    return row


with patch.object(cfg, "RESULTS", root / "results"), patch.object(run.reference, "run_one", side_effect=fake) as mocked:
    for c, lr in cfg.grid():
        rows.append(run.run_one(range(67349), range(872), None, c, lr, "cpu"))
        assert all(getattr(run.reference, k) == v for k, v in original.items())
    assert mocked.call_count == 16
    assert len({call.kwargs["output_dir"] for call in mocked.call_args_list}) == 16
    pd.DataFrame(rows).to_csv(cfg.RESULTS / "summary.csv", index=False)
    analyze(cfg.RESULTS)
    ranking = pd.read_csv(cfg.RESULTS / "ranking.csv")
    assert ranking.iloc[0].C == 4 and ranking.iloc[0].learning_rate == 1e-5
    assert len(list(cfg.RESULTS.glob("*.png"))) == 5
    with patch.object(run.reference, "run_one", side_effect=RuntimeError("smoke")):
        try:
            run.run_one(range(67349), range(872), None, 99, 1e-5, "cpu")
        except RuntimeError:
            pass
    assert all(getattr(run.reference, k) == v for k, v in original.items())
print("PASS: 16 grid points, reference contract/restoration, isolated paths, ranking, five plots")

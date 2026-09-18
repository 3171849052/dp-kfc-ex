"""Lightweight checks using synthetic summaries only; never trains a model."""
import ast
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from exp27 import config as cfg
from exp27 import run, analyze

for path in (cfg.ROOT / "exp27").glob("*.py"):
    ast.parse(path.read_text())
results = cfg.ROOT / "exp27" / "_validation" / "synthetic_results"
original = run.reference.LR, run.reference.MAX_GRAD_NORM
calls = []


def fake_run(**kwargs):
    i = len(calls)
    c, lr, slug, method, geometry = list(cfg.grid())[i]
    assert (run.reference.LR, run.reference.MAX_GRAD_NORM) == (lr, c)
    assert kwargs["geometry"] == geometry
    assert kwargs["source"] == "pink" and kwargs["engine"] == "bk"
    assert kwargs["seed"] == 42 and kwargs["epochs"] == 5
    assert kwargs["epsilon"] == 2 and kwargs["profile"] is False
    assert kwargs["public"] == {}
    assert kwargs["output_dir"].name == f"{slug}_C{c:g}_LR{lr:g}"
    assert len(pd.read_csv(results / "summary.csv")) == i
    calls.append(kwargs)
    return dict(method=method, geometry=geometry, source="pink", engine="bk", profiled=False,
                seed=42, epoch=5, epsilon_target=2., epsilon_spent=1.99, delta=1e-5,
                noise_multiplier=1.5, accuracy=0.5 + i / 120, train_loss=1., test_loss=1.,
                **{key: 1170 for key in cfg.STEP_FIELDS})


with patch.object(cfg, "RESULTS", results), patch.object(run.reference, "run_one", fake_run):
    run.run_grid(range(234), None, torch.device("cuda"))
assert len(calls) == 50
assert len({c["output_dir"] for c in calls}) == 50
assert (run.reference.LR, run.reference.MAX_GRAD_NORM) == original
analyze.analyze(results)
frame = pd.read_csv(results / "summary.csv")
for invalid in (frame.iloc[:-1], pd.concat([frame.iloc[:-1], frame.iloc[[0]]])):
    try:
        analyze.validate(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incomplete or duplicate grid was accepted")
paired = pd.read_csv(results / "paired_grid.csv")
assert len(paired) == 25
assert abs(paired.delta_accuracy - 1 / 120).max() < 1e-12
for slug, _, _ in cfg.METHODS:
    ranking = pd.read_csv(results / f"{slug}_ranking.csv")
    assert len(ranking) == 25 and ranking.accuracy.is_monotonic_decreasing
for suffix in ("png", "pdf"):
    assert (results / f"accuracy_surface.{suffix}").stat().st_size > 1000
with patch.object(cfg, "RESULTS", results / "failure_case"), patch.object(
        run.reference, "run_one", side_effect=RuntimeError("intentional validation failure")):
    try:
        run.run_grid(range(234), None, torch.device("cuda"))
    except RuntimeError as error:
        assert str(error) == "intentional validation failure"
    else:
        raise AssertionError("Training failure was swallowed")
assert (run.reference.LR, run.reference.MAX_GRAD_NORM) == original
print("PASS: AST, paired 50-call grid, unique directories, incremental CSV, global restoration,")
print("failure propagation, completeness checks, rankings, pairing, PNG/PDF generation.")
print("Synthetic data only: no training runs executed.")

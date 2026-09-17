"""Execution and instrumentation contracts, not BK/explicit equivalence tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from scripts.paper import exp_cnn_mnist_a as experiment


@pytest.mark.parametrize("geometry,source", experiment.CONDITIONS)
@pytest.mark.parametrize("engine", ["explicit", "bk"])
def test_run_instrumentation(monkeypatch, tmp_path, geometry, source, engine):
    # Small affine fixture exercises both Conv2d and Linear without dataset I/O.
    monkeypatch.setattr(experiment, "BATCH_SIZE", 4)
    monkeypatch.setattr(experiment, "SimpleCNN", lambda **kwargs: nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, stride=7, padding=1),
        nn.ReLU(), nn.Flatten(), nn.Linear(48, 10),
    ))
    torch.manual_seed(42)
    x = torch.randn(8, 1, 28, 28)
    y = torch.arange(8)
    train = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    test = DataLoader(TensorDataset(x, y), batch_size=4)
    # Invalid class labels must be ignored completely for A-only calibration.
    calibration_y = torch.full_like(y, 999) if geometry == "a_only" else y
    calibration = DataLoader(TensorDataset(x, calibration_y), batch_size=4)
    result = experiment.run_one(
        train, test, {"match": calibration, "mismatch": calibration},
        geometry, source, engine, 1.0, 42, 1, torch.device("cpu"), tmp_path,
    )
    assert result["logical_steps"] == result["optimizer_steps"] == 2
    assert result["noise_events"] == result["accountant_steps"] == 2
    assert 0 < result["epsilon_spent"] <= 1.01
    assert result["algorithm_seconds"] == result["private_train_seconds"] + result["geometry_build_seconds"]
    assert all(result[key] >= 0 for key in experiment.TIMINGS)
    assert 0 <= result["clip_fraction"] <= 1
    assert 0 < result["mean_clip_factor"] <= 1
    assert result["norm_p50"] <= result["norm_p90"] <= result["norm_p99"] <= result["norm_max"]
    assert len(pd.read_csv(next(tmp_path.glob("*.csv")))) == 1
    if engine == "bk":
        assert result["grad_sample_peak_bytes"] == 0
        assert result["sample_gradient_precondition_seconds"] == 0
        assert result["sample_gradient_matrices_preconditioned"] == 0
        assert result["bk_cache_peak_bytes"] > 0
        assert result["bk_temporary_peak_bytes"] > 0
        assert result["aggregate_matrices_preconditioned"] == (0 if geometry == "base" else 4)
    else:
        assert result["grad_sample_peak_bytes"] > 0
        assert result["aggregate_precondition_seconds"] == 0
        assert result["factor_precondition_seconds"] == 0
        assert result["aggregate_matrices_preconditioned"] == 0
        assert result["sample_gradient_matrices_preconditioned"] == (0 if geometry == "base" else 16)


def test_bk_keeps_raw_factors_and_avoids_parameter_backward():
    model = nn.Sequential(nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 2))
    layers = experiment.affine_layers(model)
    stats = {key: 0 for key in (*experiment.TIMINGS, *experiment.COUNTERS)}
    stats.update(bk_cache_peak_bytes=0, bk_temporary_peak_bytes=0)
    loss, records = experiment.bk_differentiate(
        model, layers, torch.randn(4, 5), torch.tensor([0, 1, 0, 1]), stats,
    )
    assert not loss.requires_grad
    assert all(p.grad is None and not hasattr(p, "grad_sample") for p in model.parameters())
    raw = [(r, r.z.clone(), r.b.clone()) for r in records]
    ua = {r.name: 2 * torch.eye(r.z.shape[-1]) for r in records}
    ug = {r.name: 3 * torch.eye(r.b.shape[-1]) for r in records}
    forbidden_shapes = {(4, r.b.shape[-1], r.z.shape[-1]) for r in records}

    class NoSampleMatrices(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            for tensor in tree_leaves(output):
                if isinstance(tensor, torch.Tensor):
                    assert tuple(tensor.shape) not in forbidden_shapes
            return output

    with NoSampleMatrices():
        experiment.bk_clip(records, ua, ug, stats, torch.device("cpu"))
    assert not records
    for record, z, b in raw:
        assert torch.equal(record.z, z)
        assert torch.equal(record.b, b)
    assert all(p.grad is not None and not hasattr(p, "grad_sample") for p in model.parameters())
    assert stats["aggregate_matrices_preconditioned"] == 2

#!/usr/bin/env python3
"""Lightweight CPU correctness and fixed-protocol checks for ExpM1.

Fixed protocol checks for ExpM1.
"""
from __future__ import annotations

import ast
import copy
import math
import os
import re
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

import torch
from torch import nn
from torch.nn import functional as F

from expm1 import CACHE_ROOT, ROOT
from expm1 import config as cfg
from expm1 import data
from expm1.bk import BKClipper, ghost_squared
from expm1.mechanism import add_noise_and_step
from expm1.geometry import (
    DAMPING,
    FactorBatch,
    affine_factors,
    augmented_width,
    build_factors,
    matrix_power,
    output_width,
)
from expm1.mechanism import Shape


torch.set_num_threads(1)
DTYPE = torch.float64


def _passed(label: str) -> None:
    print(f"[PASS] {label}", flush=True)


def _close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 2e-9,
    atol: float = 2e-10,
) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _parameter_values(
    model: nn.Module, values: dict[nn.Parameter, torch.Tensor]
) -> list[torch.Tensor]:
    return [values[parameter] for parameter in model.parameters() if parameter.requires_grad]


def _assert_parameter_dict_close(
    left_model: nn.Module,
    left: dict[nn.Parameter, torch.Tensor],
    right_model: nn.Module,
    right: dict[nn.Parameter, torch.Tensor],
    *,
    exact: bool = False,
) -> None:
    left_values = _parameter_values(left_model, left)
    right_values = _parameter_values(right_model, right)
    assert len(left_values) == len(right_values)
    for actual, expected in zip(left_values, right_values):
        if exact:
            assert torch.equal(actual, expected)
        else:
            _close(actual, expected)


def _spd(size: int, offset: float) -> torch.Tensor:
    diagonal = torch.linspace(0.45 + offset, 1.35 + offset, size, dtype=DTYPE)
    direction = torch.linspace(0.2, 0.7, size, dtype=DTYPE)
    return torch.diag(diagonal) + 0.11 * torch.outer(direction, direction)


def _factors(model: nn.Module) -> dict[str, dict[str, torch.Tensor | int]]:
    result: dict[str, dict[str, torch.Tensor | int]] = {}
    for index, (name, module) in enumerate(
        (entry for entry in model.named_modules() if isinstance(entry[1], (nn.Linear, nn.Conv2d)))
    ):
        width = augmented_width(module)
        out = output_width(module)
        result[name] = {
            "A": _spd(width, 0.13 * index),
            "G": _spd(out, 0.21 + 0.09 * index),
            "output_dimension": out,
        }
    assert result
    return result


class _RejectG(dict):
    """Factor mapping that turns any G observation into a hard test failure."""

    @staticmethod
    def _reject(key: object) -> None:
        if key == "G":
            raise AssertionError("DP-KFM-A attempted to access G")

    def __getitem__(self, key):
        self._reject(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._reject(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self._reject(key)
        return super().get(key, default)


def _a_only_factors(
    factors: dict[str, dict[str, torch.Tensor | int]],
) -> dict[str, _RejectG]:
    return {
        name: _RejectG(A=factor["A"], output_dimension=factor["output_dimension"])
        for name, factor in factors.items()
    }


class _LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=True, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _ConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=2, bias=True, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(-2, -1))


class _TraceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=True, dtype=DTYPE)
        self.identity_block = nn.Parameter(torch.zeros(5, dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _linear_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [0.4, -1.1, 0.3],
            [1.2, 0.2, -0.7],
            [-0.8, 0.6, 1.4],
            [0.1, 1.1, -0.2],
        ],
        dtype=DTYPE,
    )
    return x, torch.tensor([0, 2, 1, 0])


def _conv_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(913)
    x = torch.randn(4, 1, 4, 4, generator=generator, dtype=DTYPE)
    return x, torch.tensor([0, 2, 1, 2])


def _per_example_gradients(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> list[dict[nn.Parameter, torch.Tensor]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows: list[dict[nn.Parameter, torch.Tensor]] = []
    for index in range(len(x)):
        loss = F.cross_entropy(model(x[index : index + 1]), y[index : index + 1])
        gradients = torch.autograd.grad(loss, parameters)
        rows.append(
            {parameter: gradient.detach().clone() for parameter, gradient in zip(parameters, gradients)}
        )
    return rows


def _explicit_norms(
    shape: Shape,
    gradients: list[dict[nn.Parameter, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    metric_values = []
    raw_values = []
    for gradient in gradients:
        metric_squared = torch.zeros((), dtype=DTYPE)
        raw_squared = torch.zeros((), dtype=DTYPE)
        affine_ids: set[int] = set()
        for name, module in shape.modules.items():
            weight = gradient[module.weight].flatten(1)
            matrix = (
                torch.cat((weight, gradient[module.bias][:, None]), dim=1)
                if module.bias is not None
                else weight
            )
            operator = shape.data[name]
            transformed = operator.metric_g @ matrix @ operator.metric_a
            metric_squared += transformed.square().sum()
            raw_squared += matrix.square().sum()
            affine_ids.add(id(module.weight))
            if module.bias is not None:
                affine_ids.add(id(module.bias))
        for parameter in shape.parameters:
            if id(parameter) not in affine_ids:
                value = gradient[parameter]
                metric_squared += value.square().sum()
                raw_squared += value.square().sum()
        metric_values.append((metric_squared * shape.metric_scale).sqrt())
        raw_values.append(raw_squared.sqrt())
    return torch.stack(metric_values), torch.stack(raw_values)


def _weighted_sum(
    shape: Shape,
    gradients: list[dict[nn.Parameter, torch.Tensor]],
    weights: torch.Tensor,
) -> dict[nn.Parameter, torch.Tensor]:
    assert len(gradients) == len(weights)
    return {
        parameter: sum(
            (weights[index] * gradient[parameter] for index, gradient in enumerate(gradients)),
            torch.zeros_like(parameter),
        )
        for parameter in shape.parameters
    }


def _check_toy_backend(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    layer_name: str,
) -> tuple[Shape, object, list[dict[nn.Parameter, torch.Tensor]]]:
    torch.manual_seed(401)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=parameter.dtype) * 0.18)
    shape = Shape(model, "dp_kfm", _factors(model), 0.75)
    clipper = BKClipper(model, shape, bound=0.37)
    result = clipper.aggregate_logical(x, y, physical_batch_size=len(x))
    clipper.remove()
    gradients = _per_example_gradients(model, x, y)
    explicit_metric, explicit_raw = _explicit_norms(shape, gradients)
    _close(result.matched_norms, explicit_metric, rtol=3e-9, atol=3e-10)
    _close(result.raw_norms, explicit_raw, rtol=3e-9, atol=3e-10)

    ones = torch.ones(len(x), dtype=DTYPE)
    explicit_raw_sum = _weighted_sum(shape, gradients, ones)
    explicit_clipped_sum = _weighted_sum(shape, gradients, result.clip_factors)
    _assert_parameter_dict_close(model, result.raw_sum, model, explicit_raw_sum)
    _assert_parameter_dict_close(model, result.clipped_sum, model, explicit_clipped_sum)

    # KFM's signal branch must remain the explicit raw weighted gradient.
    unchanged = shape.transform_aggregate(result.clipped_sum)
    _assert_parameter_dict_close(model, unchanged, model, result.clipped_sum, exact=True)
    module = shape.modules[layer_name]
    raw_weight = result.clipped_sum[module.weight].flatten(1)
    raw_matrix = torch.cat((raw_weight, result.clipped_sum[module.bias][:, None]), dim=1)
    metric_matrix = shape.data[layer_name].metric_g @ raw_matrix @ shape.data[layer_name].metric_a
    assert not torch.allclose(metric_matrix, raw_matrix, rtol=1e-5, atol=1e-7)
    return shape, result, gradients


def check_bk_against_explicit_gradients() -> None:
    linear = _LinearClassifier()
    linear_x, linear_y = _linear_batch()
    _check_toy_backend(linear, linear_x, linear_y, "linear")
    _passed("toy Linear Mahalanobis norm and raw BK aggregates")

    conv = _ConvClassifier()
    conv_x, conv_y = _conv_batch()
    _check_toy_backend(conv, conv_x, conv_y, "conv")
    _passed("toy Conv Mahalanobis norm and raw BK aggregates")
    _passed("metric transforms do not enter DP-KFM raw aggregates")


def check_bias_augmentation() -> None:
    linear = nn.Linear(3, 2, bias=True, dtype=DTYPE)
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(3), dtype=DTYPE)
    backprop = torch.randn(4, 2, generator=torch.Generator().manual_seed(4), dtype=DTYPE)
    a, b = affine_factors(x, backprop, linear)
    assert b is not None and a.shape == (4, 1, 4) and b.shape == (4, 1, 2)
    assert torch.equal(a[..., -1], torch.ones_like(a[..., -1]))
    ghost, _ = ghost_squared(a, b)
    explicit = torch.stack(
        [torch.outer(backprop[index], torch.cat((x[index], x.new_ones(1)))).square().sum() for index in range(4)]
    )
    _close(ghost, explicit)

    conv = nn.Conv2d(1, 2, kernel_size=2, bias=True, dtype=DTYPE)
    image = torch.randn(3, 1, 4, 4, generator=torch.Generator().manual_seed(5), dtype=DTYPE)
    delta = torch.randn(3, 2, 3, 3, generator=torch.Generator().manual_seed(6), dtype=DTYPE)
    conv_a, conv_b = affine_factors(image, delta, conv)
    assert conv_b is not None and conv_a.shape[-1] == conv.weight[0].numel() + 1
    assert torch.equal(conv_a[..., -1], torch.ones_like(conv_a[..., -1]))
    ghost_conv, _ = ghost_squared(conv_a, conv_b)
    explicit_conv = torch.stack(
        [(conv_b[index].T @ conv_a[index]).square().sum() for index in range(len(image))]
    )
    _close(ghost_conv, explicit_conv)
    _passed("Linear/Conv bias uses the augmented constant-one coordinate")


def _beta_zero_result(method: str):
    torch.manual_seed(117)
    model = _LinearClassifier()
    factors = _factors(model)
    if method == "dp_sgd":
        shape = Shape(model, method, None, None)
    elif method == "dp_kfm_a":
        shape = Shape(model, method, _a_only_factors(factors), 0.0)
    else:
        shape = Shape(model, method, factors, 0.0)
    x, y = _linear_batch()
    clipper = BKClipper(model, shape, bound=0.42)
    aggregate = clipper.aggregate_logical(x, y, physical_batch_size=2)
    clipper.remove()
    noise, expected = shape.sample_noise(
        sigma=0.83,
        bound=1.0,
        generator=torch.Generator().manual_seed(81),
    )
    private = {
        parameter: aggregate.clipped_sum[parameter] + noise[parameter]
        for parameter in shape.parameters
    }
    return model, shape, aggregate, noise, expected, private


def check_beta_zero() -> None:
    reference = _beta_zero_result("dp_sgd")
    for method in ("dp_kfm", "dp_kfm_a"):
        candidate = _beta_zero_result(method)
        reference_model, reference_shape, reference_aggregate, reference_noise, reference_expected, reference_private = reference
        model, shape, aggregate, noise, expected, private = candidate
        assert shape.tau == reference_shape.tau == 1.0
        assert shape.trace_s == reference_shape.trace_s == shape.d_total
        _close(aggregate.matched_norms, reference_aggregate.matched_norms)
        _close(aggregate.raw_norms, reference_aggregate.raw_norms)
        _close(aggregate.clip_factors, reference_aggregate.clip_factors)
        assert aggregate.stats["backward_calls"] == reference_aggregate.stats["backward_calls"]
        assert aggregate.stats["physical_chunks"] == reference_aggregate.stats["physical_chunks"]
        _assert_parameter_dict_close(model, aggregate.raw_sum, reference_model, reference_aggregate.raw_sum)
        _assert_parameter_dict_close(model, aggregate.clipped_sum, reference_model, reference_aggregate.clipped_sum)
        _assert_parameter_dict_close(model, noise, reference_model, reference_noise)
        _assert_parameter_dict_close(model, private, reference_model, reference_private)
        assert expected == reference_expected
    _passed("beta=0 DP-KFM/DP-KFM-A is numerically identical to DP-SGD")


def check_trace_normalization() -> None:
    torch.manual_seed(219)
    model = _TraceModel()
    factors = _factors(model)
    beta = 0.75
    shape = Shape(model, "dp_kfm", factors, beta)
    factor = factors["linear"]
    trace_a = float(torch.trace(matrix_power(factor["A"], beta, DAMPING)))
    trace_g = float(torch.trace(matrix_power(factor["G"], beta, DAMPING)))
    raw_trace = model.identity_block.numel() + trace_a * trace_g
    expected_tau = shape.d_total / raw_trace
    assert math.isclose(shape.tau, expected_tau, rel_tol=2e-12, abs_tol=2e-12)
    assert math.isclose(shape.trace_s, shape.d_total, rel_tol=0, abs_tol=2e-12 * shape.d_total)
    rows = shape.trace_rows()
    assert {row["layer"] for row in rows} == {"linear", "identity"}
    assert math.isclose(
        sum(float(row["trace_S"]) for row in rows),
        shape.d_total,
        rel_tol=2e-12,
        abs_tol=2e-12 * shape.d_total,
    )
    assert not math.isclose(float(rows[0]["trace_S"]), model.linear.weight.numel() + model.linear.bias.numel())
    _passed("single global trace normalization, including identity blocks")


def check_shaped_noise_covariance() -> None:
    torch.manual_seed(302)
    model = nn.Sequential(nn.Linear(2, 2, bias=True, dtype=DTYPE))
    factors = _factors(model)
    beta = 0.75
    shape = Shape(model, "dp_kfm", factors, beta)
    module = model[0]
    factor = factors["0"]
    scale = 0.71 * 1.13
    covariance = (
        scale**2
        * shape.tau
        * torch.kron(
            matrix_power(factor["G"], beta, DAMPING),
            matrix_power(factor["A"], beta, DAMPING),
        )
    )
    generator = torch.Generator().manual_seed(777)
    sample_count = 30_000
    samples = torch.empty(sample_count, covariance.shape[0], dtype=DTYPE)
    last_expected: dict[str, float] | None = None
    for index in range(sample_count):
        noise, last_expected = shape.sample_noise(0.71, 1.13, generator)
        matrix = torch.cat((noise[module.weight], noise[module.bias][:, None]), dim=1)
        samples[index].copy_(matrix.flatten())
    assert last_expected is not None
    mean = samples.mean(0)
    centered = samples - mean
    empirical = centered.T @ centered / sample_count
    relative = float(torch.linalg.norm(empirical - covariance) / torch.linalg.norm(covariance))
    standardized_mean = mean.abs() / covariance.diag().sqrt()
    assert relative < 0.035, relative
    assert float(standardized_mean.max()) < 0.022
    assert math.isclose(
        sum(last_expected.values()),
        scale**2 * shape.d_total,
        rel_tol=2e-12,
        abs_tol=2e-12,
    )
    _passed(f"shaped-noise empirical covariance matches S (n={sample_count}, relerr={relative:.4f})")


def check_kfm_a_never_reads_g() -> None:
    torch.manual_seed(420)
    model = _LinearClassifier()
    factors = _a_only_factors(_factors(model))
    shape = Shape(model, "dp_kfm_a", factors, 0.5)
    raw_a = torch.randn(2, 1, 4, dtype=DTYPE)
    raw_b = torch.randn(2, 1, 3, dtype=DTYPE)
    shape.metric_factors("linear", raw_a, raw_b)
    shape.sample_noise(0.8, 1.0, torch.Generator().manual_seed(4))
    shape.diagnostic_rows(factors)

    batches = [
        FactorBatch(
            torch.randn(256, 3, generator=torch.Generator().manual_seed(500 + index), dtype=DTYPE),
            torch.arange(256) % 3,
            "public",
        )
        for index in range(10)
    ]
    built, statistics = build_factors(
        model,
        batches,
        need_g=False,
        physical_batch_size=256,
    )
    assert all("G" not in factor for factor in built.values())
    assert statistics["builder_backward_calls"] == 0

    worker_tree = ast.parse((HERE / "worker.py").read_text())
    need_g_assignments = [
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "need_g" for target in node.targets)
    ]
    assert len(need_g_assignments) == 1
    expression = need_g_assignments[0].value
    assert isinstance(expression, ast.Compare) and len(expression.ops) == 1
    assert isinstance(expression.ops[0], ast.In)
    assert isinstance(expression.comparators[0], (ast.Tuple, ast.List, ast.Set))
    methods = {ast.literal_eval(item) for item in expression.comparators[0].elts}
    assert methods == {"dp_kfc", "dp_kfm"}
    _passed("DP-KFM-A has no G lookup and its factor builder performs no backward pass")


def check_geometry_provenance() -> None:
    x = torch.zeros(2, 3, dtype=DTYPE)
    y = torch.zeros(2, dtype=torch.long)
    rejected_private = False
    try:
        FactorBatch(x, y, "private")
    except AssertionError:
        rejected_private = True
    assert rejected_private

    rejected_oracle = False
    try:
        next(data.geometry_batches(iter((FactorBatch(x, y, "oracle"),))))
    except AssertionError:
        rejected_oracle = True
    assert rejected_oracle
    for provenance in ("public", "pink"):
        accepted = next(data.geometry_batches(iter((FactorBatch(x, y, provenance),))))
        assert accepted.provenance == provenance

    worker_tree = ast.parse((HERE / "worker.py").read_text())
    factor_calls = [
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "build_factors"))
    ]
    batch_arguments = {ast.unparse(call.args[1]) for call in factor_calls if len(call.args) >= 2}
    assert "data.geometry_batches(source_batches)" in batch_arguments
    assert "data.oracle_batches(module.oracle_calibration(device))" in batch_arguments
    assert all(argument not in {"x", "(x, y)", "train_loader"} for argument in batch_arguments)
    _passed("training geometry accepts only public/pink provenance, never the current private batch")



def check_poisson_and_empty_step() -> None:
    population, expected, steps, seed = 17, 5, 4, 991
    sampler = data.FixedStepPoissonSampler(population, expected, steps, seed)
    assert sampler.sample_rate == expected / population and len(sampler) == steps
    reference = torch.Generator().manual_seed(seed)
    expected_first = (torch.rand(population, generator=reference) < expected / population).nonzero().flatten().tolist()
    assert next(iter(sampler)) == expected_first
    assert list(data.FixedStepPoissonSampler(population, expected, steps, seed)) == list(data.FixedStepPoissonSampler(population, expected, steps, seed))
    assert list(data.FixedStepPoissonSampler(population, expected, steps, seed)) != list(data.FixedStepPoissonSampler(population, expected, steps, seed + 1))
    model = nn.Linear(2, 1, bias=False, dtype=DTYPE)
    shape = Shape(model, "dp_sgd", None, None)
    clipper = BKClipper(model, shape, bound=1.0)
    empty = clipper.aggregate_logical(torch.empty(0, 2, dtype=DTYPE), torch.empty(0, dtype=torch.long), 2)
    assert empty.stats["backward_calls"] == 0 and empty.stats["physical_chunks"] == 0
    assert all(torch.equal(value, torch.zeros_like(value)) for value in empty.raw_sum.values())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = {p: p.detach().clone() for p in model.parameters()}
    stats, _ = add_noise_and_step(shape, optimizer, empty.raw_sum, empty.clipped_sum, sigma=0.5, bound=1.0, expected_batch_size=256, generator=torch.Generator().manual_seed(3))
    assert stats["total_noise_rms"] > 0
    assert any(not torch.equal(before[p], p) for p in model.parameters())
    clipper.remove()
    _passed("Poisson sampler determinism and empty-batch noisy optimizer step")


def check_one_backward_and_layernorm() -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.norm = nn.LayerNorm(3, dtype=DTYPE); self.out = nn.Linear(3, 2, dtype=DTYPE)
        def forward(self, x): return self.out(self.norm(x))
    torch.manual_seed(701); model = Model(); x = torch.randn(4, 3, dtype=DTYPE); y = torch.tensor([0, 1, 0, 1])
    shape = Shape(model, "dp_sgd", None, None); clipper = BKClipper(model, shape, bound=0.37)
    result = clipper.aggregate_logical(x, y, 2)
    assert result.stats["backward_calls"] == result.stats["physical_chunks"] == 2
    rows = _per_example_gradients(model, x, y)
    weights = result.clip_factors
    for parameter in model.parameters():
        explicit = sum((weights[i] * rows[i][parameter] for i in range(len(rows))), torch.zeros_like(parameter))
        torch.testing.assert_close(result.clipped_sum[parameter], explicit, rtol=2e-6, atol=2e-7)
    clipper.remove()
    _passed("one-backward-per-chunk and LayerNorm clipped aggregate")


def check_launcher_grid_assertion() -> None:
    source = (HERE / "run_all.sh").read_text()
    preflight = source[source.index("runs = tuple(cfg.formal_runs())"):source.index("PY\n\nmkdir", source.index("runs = tuple(cfg.formal_runs())"))]
    assert "assert len(runs) == 38" in preflight
    assert "== {1: 13, 2: 13, 3: 12}" in preflight
    worker_listing = source[source.index("gpu = int(sys.argv[1])"):]
    assert "assert len(runs) == {1: 13, 2: 13, 3: 12}[gpu]" in worker_listing
    _passed("launcher distinguishes full 38-run grid from per-GPU 13/13/12 subsets")

def check_formal_protocol() -> None:
    assert cfg.TASKS == ("mnist", "vit")
    assert cfg.METHODS == ("dp_sgd", "dp_kfc", "dp_kfm", "dp_kfm_a")
    assert cfg.SOURCES == ("pink", "public")
    assert cfg.BETAS == (0.25, 0.5, 0.75, 1.0)
    assert cfg.SEEDS == (42,)
    assert cfg.EPOCHS == 5
    assert cfg.AUXILIARY_BATCHES == cfg.ORACLE_BATCHES == 10
    assert cfg.AUXILIARY_BATCH_SIZE == cfg.ORACLE_BATCH_SIZE == 256
    assert cfg.DAMPING == 1e-3 and cfg.MAX_GRAD_NORM == 1.0 and cfg.DELTA == 1e-5

    fixed = {
        "mnist": (60_000, 5, 256, 256, 1.0, 1e-5, 1),
        "vit": (50_000, 5, 256, 128, 3.0, 1e-5, 2),
    }
    for task, expected in fixed.items():
        protocol = cfg.task_config(task)
        actual = (
            protocol.train_samples,
            protocol.epochs,
            protocol.logical_batch_size,
            protocol.physical_batch_size,
            protocol.epsilon,
            protocol.delta,
            protocol.accumulation_steps,
        )
        assert actual == expected
        assert protocol.accountant_steps == protocol.epochs * (
            protocol.train_samples // protocol.logical_batch_size
        )
        assert protocol.sample_rate == protocol.logical_batch_size / protocol.train_samples

    expected_conditions = {
        ("dp_sgd", "none", None),
        *(("dp_kfc", source, None) for source in cfg.SOURCES),
        *((method, source, beta)
          for method in ("dp_kfm", "dp_kfm_a")
          for source in cfg.SOURCES
          for beta in cfg.BETAS),
    }
    grid = cfg.formal_grid()
    assert grid == cfg.FORMAL_GRID and len(grid) == 38
    assert len({run.run_name for run in grid}) == len(grid)
    assert all(run.beta != 0 for run in grid)
    for task in cfg.TASKS:
        task_runs = [run for run in grid if run.task == task]
        assert len(task_runs) == 19
        for seed in cfg.SEEDS:
            actual = {
                (run.method, run.source, run.beta)
                for run in task_runs
                if run.seed == seed
            }
            assert actual == expected_conditions
    assert set(cfg.GPU_BY_RUN) == {run.run_name for run in grid}
    assert tuple(cfg.GPU_BY_RUN[run.run_name] for run in grid) == tuple((1, 2, 3)[i % 3] for i in range(38))
    assert {gpu: len(cfg.GPU_RUNS[gpu]) for gpu in cfg.PHYSICAL_GPUS} == {1: 13, 2: 13, 3: 12}
    _passed("formal grid, seeds, epochs, logical/physical batches, and privacy protocol")


def check_launcher() -> None:
    launcher = HERE / "run_all.sh"
    assert launcher.is_file(), f"missing launcher: {launcher}"
    source = launcher.read_text()
    assert "nvidia-smi" not in source and "RANDOM" not in source and "shuf" not in source
    assert "worker.py" in source and "analyze.py" in source
    assert re.search(r"\bwait\b", source)
    assert re.search(r"status|failed|failure", source, flags=re.IGNORECASE)

    direct = {int(value) for value in re.findall(r"CUDA_VISIBLE_DEVICES\s*=\s*['\"]?([0-9]+)", source)}
    function_calls = {
        int(value)
        for value in re.findall(r"^\s*(?:run_gpu|worker|launch_gpu)\s+([0-9]+)\b", source, flags=re.MULTILINE)
    }
    physical = direct | function_calls
    assert physical == {1, 2, 3}, f"launcher physical GPUs are not exactly 1/2/3: {physical}"
    numeric_assignments = {
        int(value) for value in re.findall(r"CUDA_VISIBLE_DEVICES[^\n]*?([0-9]+)", source)
    }
    assert numeric_assignments <= {1, 2, 3}

    worker_source = (HERE / "worker.py").read_text()
    assert 'assert visible == str(spec.gpu)' in worker_source
    assert "cfg.GPU_RUNS" in source or "GPU_BY_RUN" in source or "--gpu" not in source
    _passed("launcher has a static balanced mapping and only physical GPUs 1/2/3")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def check_source_boundaries() -> None:
    python_files = sorted(HERE.glob("*.py"))
    assert python_files
    dataset_calls: set[str] = set()
    mutating_methods = {
        "write_text",
        "write_bytes",
        "mkdir",
        "to_csv",
        "savefig",
        "save",
        "copytree",
        "rename",
        "replace",
    }
    for path in python_files:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        direct_dataset_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "torchvision.datasets":
                direct_dataset_names.update(alias.asname or alias.name for alias in node.names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            tail = name.rsplit(".", 1)[-1]
            is_dataset = name.startswith("datasets.") or tail in direct_dataset_names
            if is_dataset:
                downloads = [keyword.value for keyword in node.keywords if keyword.arg == "download"]
                assert len(downloads) == 1, f"dataset call lacks explicit download=False: {path}:{node.lineno}"
                assert isinstance(downloads[0], ast.Constant) and downloads[0].value is False
                dataset_calls.add(tail)

            if name in ("tempfile.TemporaryDirectory", "tempfile.NamedTemporaryFile", "tempfile.mkdtemp"):
                assert any(keyword.arg == "dir" for keyword in node.keywords), (
                    f"temporary path is not rooted in expm1: {path}:{node.lineno}"
                )
            if tail in mutating_methods:
                segment = ast.get_source_segment(source, node) or ""
                assert "DATA_ROOT" not in segment
                assert "REPO_ROOT" not in segment
                if tail == "copytree":
                    assert "LOCAL_CHECKPOINT" in segment

    assert {"MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"} <= dataset_calls
    assert cfg.DATA_ROOT.resolve() == (REPO_ROOT / "data").resolve()
    for path in (ROOT, CACHE_ROOT, cfg.RESULTS_ROOT, cfg.LOGS_ROOT, cfg.TMP_ROOT):
        assert path.resolve().is_relative_to(ROOT.resolve())
    runtime_environment = (
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "TORCH_HOME",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "NUMBA_CACHE_DIR",
        "TMPDIR",
        "TMP",
        "TEMP",
    )
    for variable in runtime_environment:
        value = Path(os.environ[variable]).resolve()
        assert value.is_relative_to(ROOT.resolve()), f"{variable} escaped expm1: {value}"
    _passed("source scan: every dataset is download=False and runtime writes stay in expm1")


def check_incomplete_analysis_fails_atomically() -> None:
    from expm1 import analyze

    cfg.TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checks-incomplete-", dir=cfg.TMP_ROOT) as directory:
        isolated = Path(directory)
        old_results, old_runs = analyze.RESULTS, analyze.RUNS
        analyze.RESULTS, analyze.RUNS = isolated, isolated / "runs"
        failed = False
        try:
            analyze.analyze()
        except AssertionError as error:
            assert "missing formal run directory" in str(error)
            failed = True
        finally:
            analyze.RESULTS, analyze.RUNS = old_results, old_runs
        assert failed
        assert list(isolated.iterdir()) == [], "incomplete analysis wrote output"
    _passed("analyze rejects an incomplete grid before writing any output")


def check_data_preflight_last() -> None:

    required = {
        "MNIST": cfg.DATA_ROOT / "MNIST" / "raw" / "train-images-idx3-ubyte",
        "FashionMNIST": cfg.DATA_ROOT / "FashionMNIST" / "raw" / "train-images-idx3-ubyte",
        "CIFAR10": cfg.DATA_ROOT / "cifar-10-batches-py" / "data_batch_1",
        "CIFAR100": cfg.DATA_ROOT / "cifar-100-python" / "train",
    }
    missing = {name: path for name, path in required.items() if not path.is_file()}
    assert not missing, f"required fixed datasets are absent: {missing}"
    _passed("data preflight: MNIST, FashionMNIST, CIFAR-10, and CIFAR-100 are present")


def main() -> None:
    check_beta_zero()
    check_bk_against_explicit_gradients()
    check_bias_augmentation()
    check_trace_normalization()
    check_shaped_noise_covariance()
    check_kfm_a_never_reads_g()
    check_geometry_provenance()
    check_poisson_and_empty_step()
    check_one_backward_and_layernorm()
    check_formal_protocol()
    check_launcher_grid_assertion()
    check_launcher()
    check_source_boundaries()
    check_incomplete_analysis_fails_atomically()
    print("[PASS] all numerical/protocol/source checks completed; starting final data preflight", flush=True)
    check_data_preflight_last()


if __name__ == "__main__":
    main()

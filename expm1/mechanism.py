"""Metric shapes, standard KFC transforms, and matched Gaussian mechanisms."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from expm1.geometry import (
    DAMPING,
    affine_modules,
    augmented_width,
    matrix_power,
    output_width,
    spectrum,
)


METHODS = frozenset({"dp_sgd", "dp_kfc", "dp_kfm", "dp_kfm_a"})


def layer_group(name: str) -> str:
    if name.endswith(("q_proj", "k_proj", "v_proj")):
        return "attention_qkv"
    if name.endswith("attn.out_proj"):
        return "attention_out"
    if name.endswith("mlp.fc1"):
        return "mlp_fc1"
    if name.endswith("mlp.fc2"):
        return "mlp_fc2"
    if name in ("patch_embed", "head"):
        return "patch_head"
    return name if name in ("conv1", "conv2", "fc1", "fc2") else "identity"


@dataclass
class AffineShape:
    metric_a: torch.Tensor
    metric_g: torch.Tensor
    noise_a: torch.Tensor
    noise_g: torch.Tensor
    raw_trace: float

    @property
    def state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.metric_a, self.metric_g, self.noise_a, self.noise_g)
        )


def _identity(size: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.eye(size, device=reference.device, dtype=reference.dtype)


class Shape:
    """One fixed epoch geometry, including the single global trace scale tau."""

    def __init__(
        self,
        model: nn.Module,
        method: str,
        factors: dict[str, dict] | None,
        beta: float | None,
        damping: float = DAMPING,
    ) -> None:
        assert method in METHODS
        assert damping == 1e-3
        if method in ("dp_kfm", "dp_kfm_a"):
            assert beta in (0.0, 0.25, 0.5, 0.75, 1.0)
        else:
            assert beta is None
        self.model = model
        self.method = method
        self.beta = beta
        self.damping = damping
        self.modules = affine_modules(model)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.parameter_names = {
            parameter: name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.affine_parameter_ids = {
            id(parameter)
            for module in self.modules.values()
            for parameter in module.parameters(recurse=False)
            if parameter.requires_grad
        }
        self.identity_parameters = [
            parameter for parameter in self.parameters if id(parameter) not in self.affine_parameter_ids
        ]
        self.d_total = sum(parameter.numel() for parameter in self.parameters)
        assert self.d_total > 0
        self.data: dict[str, AffineShape] = {}

        if method == "dp_sgd":
            assert factors is None and beta is None
        else:
            assert factors is not None and set(factors) == set(self.modules)

        raw_trace = float(sum(parameter.numel() for parameter in self.identity_parameters))
        for name, module in self.modules.items():
            width, out = augmented_width(module), output_width(module)
            if method == "dp_sgd":
                reference = module.weight
                eye_a, eye_g = _identity(width, reference), _identity(out, reference)
                shape = AffineShape(eye_a, eye_g, eye_a, eye_g, float(width * out))
            else:
                factor = factors[name]
                a = factor["A"]
                assert tuple(a.shape) == (width, width)
                if method == "dp_kfc":
                    g = factor["G"]
                    assert tuple(g.shape) == (out, out)
                    eye_a, eye_g = _identity(width, a), _identity(out, g)
                    shape = AffineShape(
                        matrix_power(a, -0.5, damping),
                        matrix_power(g, -0.5, damping),
                        eye_a,
                        eye_g,
                        float(width * out),
                    )
                elif beta == 0:
                    # This exact branch (not an eigendecomposition of I) makes the
                    # beta-zero correctness check bit-for-bit equivalent to SGD.
                    eye_a = _identity(width, a)
                    eye_g = _identity(out, a)
                    shape = AffineShape(eye_a, eye_g, eye_a, eye_g, float(width * out))
                elif method == "dp_kfm_a":
                    # Do not even look up G: A-only has no G dependency.
                    assert beta is not None
                    metric_a = matrix_power(a, -beta / 2, damping)
                    noise_a = matrix_power(a, beta / 2, damping)
                    eye_g = _identity(out, a)
                    trace_a = float(torch.linalg.eigvalsh(((a + a.T) * 0.5).double())
                                    .clamp_min(0).add(damping).pow(beta).sum())
                    shape = AffineShape(
                        metric_a, eye_g, noise_a, eye_g, out * trace_a
                    )
                else:
                    assert method == "dp_kfm" and beta is not None
                    g = factor["G"]
                    assert tuple(g.shape) == (out, out)
                    metric_a = matrix_power(a, -beta / 2, damping)
                    metric_g = matrix_power(g, -beta / 2, damping)
                    noise_a = matrix_power(a, beta / 2, damping)
                    noise_g = matrix_power(g, beta / 2, damping)
                    trace_a = float(torch.linalg.eigvalsh(((a + a.T) * 0.5).double())
                                    .clamp_min(0).add(damping).pow(beta).sum())
                    trace_g = float(torch.linalg.eigvalsh(((g + g.T) * 0.5).double())
                                    .clamp_min(0).add(damping).pow(beta).sum())
                    shape = AffineShape(
                        metric_a, metric_g, noise_a, noise_g, trace_a * trace_g
                    )
            self.data[name] = shape
            raw_trace += shape.raw_trace

        self.tau = self.d_total / raw_trace if method in ("dp_kfm", "dp_kfm_a") else 1.0
        self.trace_s = self.tau * raw_trace
        tolerance = 2e-10 * max(1, self.d_total)
        assert abs(self.trace_s - self.d_total) <= tolerance
        if method in ("dp_kfm", "dp_kfm_a") and beta == 0:
            assert self.tau == 1.0
        if method == "dp_sgd":
            self.operator_state_bytes = 0
        else:
            storages: dict[tuple[torch.device, int], int] = {}
            for value in self.data.values():
                for tensor in (
                    value.metric_a,
                    value.metric_g,
                    value.noise_a,
                    value.noise_g,
                ):
                    storage = tensor.untyped_storage()
                    storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
            self.operator_state_bytes = sum(storages.values()) + 8

    @property
    def metric_scale(self) -> float:
        return 1.0 / self.tau if self.method in ("dp_kfm", "dp_kfm_a") else 1.0

    def metric_factors(
        self, name: str, raw_a: torch.Tensor, raw_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = self.data[name]
        metric_a = raw_a @ shape.metric_a.T
        metric_b = raw_b @ shape.metric_g.T
        return metric_a, metric_b

    def _transform_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        shape = self.data[name]
        return shape.metric_g @ matrix @ shape.metric_a

    def transform_aggregate(self, aggregate: dict[nn.Parameter, torch.Tensor]) -> dict[nn.Parameter, torch.Tensor]:
        """Standard DP-KFC signal transform; all other mechanisms stay raw."""
        result = {parameter: value.clone() for parameter, value in aggregate.items()}
        if self.method != "dp_kfc":
            return result
        for name, module in self.modules.items():
            weight = aggregate[module.weight].flatten(1)
            matrix = (
                torch.cat((weight, aggregate[module.bias][:, None]), dim=1)
                if module.bias is not None
                else weight
            )
            transformed = self._transform_matrix(name, matrix)
            result[module.weight] = transformed[:, : weight.shape[1]].reshape_as(module.weight)
            if module.bias is not None:
                result[module.bias] = transformed[:, -1]
        return result

    def trace_rows(self) -> list[dict[str, object]]:
        rows = []
        for name, shape in self.data.items():
            rows.append(
                {
                    "layer": name,
                    "group": layer_group(name),
                    "trace_S": self.tau * shape.raw_trace,
                    "d_total": self.d_total,
                    "tau": self.tau,
                    "operator_state_bytes": self.operator_state_bytes,
                }
            )
        identity_dimension = sum(parameter.numel() for parameter in self.identity_parameters)
        if identity_dimension:
            rows.append(
                {
                    "layer": "identity",
                    "group": "identity",
                    "trace_S": self.tau * identity_dimension,
                    "d_total": self.d_total,
                    "tau": self.tau,
                    "operator_state_bytes": self.operator_state_bytes,
                }
            )
        assert math.isclose(
            sum(float(row["trace_S"]) for row in rows),
            float(self.d_total),
            rel_tol=2e-10,
            abs_tol=2e-10 * max(1, self.d_total),
        )
        return rows

    def diagnostic_rows(self, factors: dict[str, dict] | None) -> list[dict[str, object]]:
        rows = self.trace_rows()
        if factors is None:
            return rows
        for row in rows:
            name = row["layer"]
            if name == "identity":
                continue
            a_stats = spectrum(factors[name]["A"], self.damping)
            row.update({f"A_{key}": value for key, value in a_stats.items()})
            if self.method in ("dp_kfc", "dp_kfm"):
                g_stats = spectrum(factors[name]["G"], self.damping)
                row.update({f"G_{key}": value for key, value in g_stats.items()})
                row["condition_number"] = (
                    a_stats["damped_condition"] * g_stats["damped_condition"]
                )
                row["log_eigenvalue_spread"] = math.sqrt(
                    a_stats["log_eigenvalue_spread"] ** 2
                    + g_stats["log_eigenvalue_spread"] ** 2
                )
                row["effective_rank"] = (
                    a_stats["effective_rank"] * g_stats["effective_rank"]
                )
            else:
                row["condition_number"] = a_stats["damped_condition"]
                row["log_eigenvalue_spread"] = a_stats["log_eigenvalue_spread"]
                row["effective_rank"] = a_stats["effective_rank"]
        return rows

    def _base_noise(self, generator: torch.Generator) -> dict[nn.Parameter, torch.Tensor]:
        # Draw in named-parameter order for exact beta=0/SGD RNG equivalence.
        return {
            parameter: torch.randn(
                parameter.shape,
                device=parameter.device,
                dtype=parameter.dtype,
                generator=generator,
            )
            for parameter in self.parameters
        }

    def sample_noise(
        self,
        sigma: float,
        bound: float,
        generator: torch.Generator,
    ) -> tuple[dict[nn.Parameter, torch.Tensor], dict[str, float]]:
        assert sigma > 0 and bound > 0
        base = self._base_noise(generator)
        scale = sigma * bound
        noise: dict[nn.Parameter, torch.Tensor] = {}
        expected: dict[str, float] = {}
        affine_ids: set[int] = set()
        for name, module in self.modules.items():
            shape = self.data[name]
            raw_weight = base[module.weight].flatten(1)
            raw_matrix = (
                torch.cat((raw_weight, base[module.bias][:, None]), dim=1)
                if module.bias is not None
                else raw_weight
            )
            if self.method in ("dp_kfm", "dp_kfm_a"):
                matrix = math.sqrt(self.tau) * (
                    shape.noise_g @ raw_matrix @ shape.noise_a
                )
                trace = self.tau * shape.raw_trace
            else:
                matrix = raw_matrix
                trace = float(matrix.numel())
            matrix = matrix * scale
            noise[module.weight] = matrix[:, : raw_weight.shape[1]].reshape_as(module.weight)
            affine_ids.add(id(module.weight))
            if module.bias is not None:
                noise[module.bias] = matrix[:, -1]
                affine_ids.add(id(module.bias))
            expected[name] = scale * scale * trace
        for parameter in self.identity_parameters:
            multiplier = math.sqrt(self.tau) if self.method in ("dp_kfm", "dp_kfm_a") else 1.0
            noise[parameter] = base[parameter] * (scale * multiplier)
            expected[self.parameter_names[parameter]] = (
                scale * scale * self.tau * parameter.numel()
                if self.method in ("dp_kfm", "dp_kfm_a")
                else scale * scale * parameter.numel()
            )
        assert set(noise) == set(self.parameters)
        target = scale * scale * self.d_total
        assert math.isclose(sum(expected.values()), target, rel_tol=2e-9, abs_tol=2e-9 * target)
        return noise, expected


def _dot(left: dict[nn.Parameter, torch.Tensor], right: dict[nn.Parameter, torch.Tensor]) -> torch.Tensor:
    return sum((left[p].double() * right[p].double()).sum() for p in left)


def _norm(values: dict[nn.Parameter, torch.Tensor]) -> torch.Tensor:
    return _dot(values, values).clamp_min(0).sqrt()


def distortion(
    reference: dict[nn.Parameter, torch.Tensor],
    candidate: dict[nn.Parameter, torch.Tensor],
) -> tuple[float, float]:
    reference_norm = _norm(reference)
    candidate_norm = _norm(candidate)
    assert reference_norm > 0 and candidate_norm > 0
    cosine = (_dot(reference, candidate) / (reference_norm * candidate_norm)).clamp(-1, 1)
    difference = {parameter: candidate[parameter] - value for parameter, value in reference.items()}
    relative = _norm(difference) / reference_norm
    return float(cosine), float(relative)


def add_noise_and_step(
    shape: Shape,
    optimizer: torch.optim.Optimizer,
    raw_sum: dict[nn.Parameter, torch.Tensor],
    clipped_sum: dict[nn.Parameter, torch.Tensor],
    *,
    sigma: float,
    bound: float,
    logical_batch_size: int,
    generator: torch.Generator,
) -> tuple[dict[str, float | bool], list[dict[str, object]]]:
    """Apply one mechanism event and return research-only distortion/noise rows."""
    assert logical_batch_size > 0
    actual_raw = shape.transform_aggregate(raw_sum)
    actual_clipped = shape.transform_aggregate(clipped_sum)
    clip_cos, clip_rel = distortion(actual_raw, actual_clipped)
    noise, expected = shape.sample_noise(sigma, bound, generator)
    private_sum = {
        parameter: actual_clipped[parameter] + noise[parameter]
        for parameter in shape.parameters
    }
    update_cos, update_rel = distortion(actual_raw, private_sum)
    for parameter in shape.parameters:
        parameter.grad = private_sum[parameter].div(logical_batch_size)
    optimizer.step()

    total_noise_sq = sum(value.double().square().sum() for value in noise.values())
    total_noise_rms = float((total_noise_sq / shape.d_total).sqrt())
    rows: list[dict[str, object]] = []
    parameter_layer: dict[nn.Parameter, str] = {}
    for name, module in shape.modules.items():
        parameter_layer[module.weight] = name
        if module.bias is not None:
            parameter_layer[module.bias] = name
    for parameter in shape.identity_parameters:
        parameter_layer[parameter] = shape.parameter_names[parameter]

    layer_names = list(shape.modules) + [shape.parameter_names[p] for p in shape.identity_parameters]
    total_expected = sum(expected.values())
    for name in layer_names:
        parameters = [p for p, layer in parameter_layer.items() if layer == name]
        actual_energy = sum(float(noise[p].double().square().sum()) for p in parameters)
        signal_norm = math.sqrt(
            sum(float(actual_clipped[p].double().square().sum()) for p in parameters)
        )
        expected_energy = expected[name]
        rows.append(
            {
                "level": "layer",
                "layer": name,
                "group": layer_group(name) if name in shape.modules else "identity",
                "noise_rms": math.sqrt(actual_energy / sum(p.numel() for p in parameters)),
                "noise_energy_share": expected_energy / total_expected,
                "expected_noise_energy": expected_energy,
                "signal_norm": signal_norm,
                "snr": signal_norm / math.sqrt(expected_energy),
                "research_only": True,
            }
        )

    for group in sorted({str(row["group"]) for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        names = {str(row["layer"]) for row in selected}
        parameters = [p for p, name in parameter_layer.items() if name in names]
        dimension = sum(p.numel() for p in parameters)
        actual_energy = sum(float(noise[p].double().square().sum()) for p in parameters)
        expected_energy = sum(float(row["expected_noise_energy"]) for row in selected)
        signal_sq = sum(float(actual_clipped[p].double().square().sum()) for p in parameters)
        rows.append(
            {
                "level": "group",
                "layer": group,
                "group": group,
                "noise_rms": math.sqrt(actual_energy / dimension),
                "noise_energy_share": expected_energy / total_expected,
                "expected_noise_energy": expected_energy,
                "signal_norm": math.sqrt(signal_sq),
                "snr": math.sqrt(signal_sq / expected_energy),
                "research_only": True,
            }
        )
    return {
        "clip_cos": clip_cos,
        "clip_rel_error": clip_rel,
        "update_cos": update_cos,
        "update_rel_error": update_rel,
        "total_noise_rms": total_noise_rms,
        "research_only": True,
    }, rows

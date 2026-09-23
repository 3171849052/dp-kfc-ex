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
    output_width,
    _sym_eig,
)


METHODS = frozenset({"dp_sgd", "dp_kfc", "dp_kfm", "dp_kfm_a"})


def _spectral_stats(values: torch.Tensor) -> dict[str, float]:
    probabilities = values / values.sum()
    return {
        "condition": float(values.max() / values.min()),
        "log_std": float(values.log().std(unbiased=False)),
        "effective_rank": float((-(probabilities * probabilities.log()).sum()).exp()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _effective_rank(values: torch.Tensor) -> float:
    probabilities = values / values.sum()
    return float((-(probabilities * probabilities.log()).sum()).exp())


def _shape_diagnostics(
    eigenvalues_a: torch.Tensor,
    eigenvalues_g: torch.Tensor | None,
    a: dict[str, float],
    g: dict[str, float] | None,
    beta: float,
    out: int,
):
    rank_a = _effective_rank(eigenvalues_a.pow(beta))
    if g is None:
        condition = a["condition"] ** beta
        spread = beta * a["log_std"]
        rank = rank_a * out
        g_raw = {}
    else:
        condition = (a["condition"] * g["condition"]) ** beta
        spread = beta * math.sqrt(a["log_std"] ** 2 + g["log_std"] ** 2)
        assert eigenvalues_g is not None
        rank = rank_a * _effective_rank(eigenvalues_g.pow(beta))
        g_raw = {
            "G_condition_raw": g["condition"],
            "G_eigenvalue_min": g["minimum"],
            "G_eigenvalue_max": g["maximum"],
        }
    return {
        "condition_S": condition,
        "log_eigenvalue_spread_S": spread,
        "effective_rank_S": rank,
        "A_condition_raw": a["condition"],
        "A_eigenvalue_min": a["minimum"],
        "A_eigenvalue_max": a["maximum"],
        **g_raw,
    }


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
    metric_a: torch.Tensor | None
    metric_g: torch.Tensor | None
    noise_a: torch.Tensor | None
    noise_g: torch.Tensor | None
    raw_trace: float
    diagnostics: dict[str, float]

    @property
    def state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.metric_a, self.metric_g, self.noise_a, self.noise_g)
            if tensor is not None
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
                shape = AffineShape(None, None, None, None, float(width * out), {})
            else:
                factor = factors[name]
                a = factor["A"]
                assert tuple(a.shape) == (width, width)
                values_a, vectors_a = _sym_eig(a)
                damped_a = values_a + damping
                def power_a(power: float) -> torch.Tensor:
                    return ((vectors_a * damped_a.pow(power)) @ vectors_a.T).to(a.dtype)
                a_stats = _spectral_stats(damped_a)
                if method == "dp_kfc":
                    g = factor["G"]
                    assert tuple(g.shape) == (out, out)
                    values_g, vectors_g = _sym_eig(g)
                    damped_g = values_g + damping
                    def power_g(power: float) -> torch.Tensor:
                        return ((vectors_g * damped_g.pow(power)) @ vectors_g.T).to(g.dtype)
                    shape = AffineShape(power_a(-0.5), power_g(-0.5), None, None,
                                        float(width * out), {
                                            "A_condition_raw": a_stats["condition"],
                                            "A_eigenvalue_min": a_stats["minimum"],
                                            "A_eigenvalue_max": a_stats["maximum"],
                                            "G_condition_raw": _spectral_stats(damped_g)["condition"],
                                            "G_eigenvalue_min": float(values_g.min()),
                                            "G_eigenvalue_max": float(values_g.max()),
                                        })
                elif method == "dp_kfm_a":
                    assert beta is not None
                    raw_trace_block = out * float(damped_a.pow(beta).sum())
                    diagnostics = _shape_diagnostics(damped_a, None, a_stats, None, beta, out)
                    shape = AffineShape(power_a(-beta / 2), None, power_a(beta / 2), None,
                                        raw_trace_block, diagnostics)
                else:
                    assert method == "dp_kfm" and beta is not None
                    g = factor["G"]
                    values_g, vectors_g = _sym_eig(g)
                    damped_g = values_g + damping
                    def power_g(power: float) -> torch.Tensor:
                        return ((vectors_g * damped_g.pow(power)) @ vectors_g.T).to(g.dtype)
                    g_stats = _spectral_stats(damped_g)
                    raw_trace_block = float(damped_a.pow(beta).sum() * damped_g.pow(beta).sum())
                    diagnostics = _shape_diagnostics(damped_a, damped_g, a_stats, g_stats, beta, out)
                    shape = AffineShape(power_a(-beta / 2), power_g(-beta / 2),
                                        power_a(beta / 2), power_g(beta / 2),
                                        raw_trace_block, diagnostics)
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
            self.factor_state_bytes = 0
            self.geometry_build_seconds = 0.0
        else:
            storages: dict[tuple[torch.device, int], int] = {}
            for value in self.data.values():
                for tensor in (
                    value.metric_a,
                    value.metric_g,
                    value.noise_a,
                    value.noise_g,
                ):
                    if tensor is None:
                        continue
                    storage = tensor.untyped_storage()
                    storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
            self.operator_state_bytes = sum(storages.values()) + 8
            self.factor_state_bytes = sum(
                tensor.numel() * tensor.element_size()
                for factor in factors.values()
                for tensor in factor.values()
                if isinstance(tensor, torch.Tensor)
            )
            self.geometry_build_seconds = 0.0

    @property
    def metric_scale(self) -> float:
        return 1.0 / self.tau if self.method in ("dp_kfm", "dp_kfm_a") else 1.0

    def metric_factors(
        self, name: str, raw_a: torch.Tensor, raw_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "dp_sgd":
            return raw_a, raw_b
        shape = self.data[name]
        assert shape.metric_a is not None
        metric_a = raw_a @ shape.metric_a.T
        metric_b = raw_b if shape.metric_g is None else raw_b @ shape.metric_g.T
        if self.method == "dp_kfm_a":
            return metric_a, raw_b
        return metric_a, metric_b

    def _transform_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        shape = self.data[name]
        assert shape.metric_a is not None and shape.metric_g is not None
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
            row.update(self.data[name].diagnostics)
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
                    (raw_matrix @ shape.noise_a) if shape.noise_g is None
                    else shape.noise_g @ raw_matrix @ shape.noise_a
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
    if reference_norm == 0:
        return 0.0, 0.0
    cosine = (
        (_dot(reference, candidate) / (reference_norm * candidate_norm)).clamp(-1, 1)
        if candidate_norm > 0
        else reference_norm.new_zeros(())
    )
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
    expected_batch_size: int,
    generator: torch.Generator,
) -> tuple[dict[str, float | bool], list[dict[str, object]]]:
    """Apply one mechanism event and return research-only distortion/noise rows."""
    assert expected_batch_size == 256
    clip_cos, clip_rel = distortion(raw_sum, clipped_sum)
    signal = shape.transform_aggregate(clipped_sum)
    signal_cos, signal_rel = distortion(raw_sum, signal)
    noise, expected = shape.sample_noise(sigma, bound, generator)
    private_sum = {
        parameter: signal[parameter] + noise[parameter]
        for parameter in shape.parameters
    }
    update_cos, update_rel = distortion(raw_sum, private_sum)
    for parameter in shape.parameters:
        parameter.grad = private_sum[parameter].div(expected_batch_size)
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
            sum(float(signal[p].double().square().sum()) for p in parameters)
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
        signal_sq = sum(float(signal[p].double().square().sum()) for p in parameters)
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
        "signal_cos": signal_cos,
        "signal_rel_error": signal_rel,
        "update_cos": update_cos,
        "update_rel_error": update_rel,
        "total_noise_rms": total_noise_rms,
        "research_only": True,
    }, rows

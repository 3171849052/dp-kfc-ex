"""Exp22 correctness tests.  They use small CPU tensors unless CUDA is needed."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
for key, relative in {
    "XDG_CACHE_HOME": ".cache",
    "MPLCONFIGDIR": ".cache/matplotlib",
    "CUDA_CACHE_PATH": ".cache/cuda",
    "TMPDIR": ".cache/tmp",
    "HF_HOME": ".cache/huggingface",
    "TORCH_HOME": ".cache/torch",
}.items():
    path = ROOT / "exp22" / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

from exp22.geometry import (
    AOnlyOperator,
    FullKFACOperator,
    build_from_batches,
    flatten_linear_input,
    matrix_function,
    synthetic_stream,
    synthetic_labels,
    synthetic_label_stream,
)
from exp22 import config as cfg
from exp22.analyze import accuracy_auc, analyze
from exp22.methods import Clipper
from exp22.model import convert_vit, initialize

torch.set_num_threads(4)


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def small_model(seed=11):
    from timm.models.vision_transformer import VisionTransformer
    torch.manual_seed(seed)
    return convert_vit(VisionTransformer(
        img_size=8, patch_size=4, in_chans=1, embed_dim=8,
        num_heads=2, depth=1, num_classes=3,
    )).to(device())


def test_explicit_attention_conversion_preserves_logits():
    import timm
    torch.manual_seed(17)
    old = timm.create_model(cfg.MODEL_NAME, pretrained=True).to(device()).eval()
    new = convert_vit(old).eval()
    x = torch.randn(2, 3, 224, 224, device=device())
    with torch.no_grad():
        torch.testing.assert_close(old(x), new(x), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(new.patch_embed.weight, old.patch_embed.proj.weight.flatten(1), rtol=0, atol=0)
    for original, converted in zip(old.blocks, new.blocks):
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            torch.testing.assert_close(getattr(converted.attn, name).weight,
                                       original.attn.qkv.weight.chunk(3)[index], rtol=0, atol=0)
        torch.testing.assert_close(original.mlp.fc1.weight, converted.mlp.fc1.weight, rtol=0, atol=0)
        torch.testing.assert_close(original.mlp.fc2.weight, converted.mlp.fc2.weight, rtol=0, atol=0)
    torch.testing.assert_close(old.cls_token, new.cls_token, rtol=0, atol=0)
    torch.testing.assert_close(old.pos_embed, new.pos_embed, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["dp_kfc_a_bk", "dp_kfc"])
def test_exact_linear_coverage_and_identity_layers(method):
    model = initialize(42, device())
    names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    assert len(names) == 74
    expected = {"patch_embed", "head"}
    expected |= {f"blocks.{i}.attn.{part}_proj" for i in range(12) for part in ("q", "k", "v")}
    expected |= {f"blocks.{i}.attn.out_proj" for i in range(12)}
    expected |= {f"blocks.{i}.mlp.{part}" for i in range(12) for part in ("fc1", "fc2")}
    assert names == expected
    assert sum(isinstance(module, nn.LayerNorm) for module in model.modules()) == 25
    assert model.head.out_features == 10
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224, device=device())).shape == (2, 10)
    assert all(p.requires_grad for p in model.parameters())
    cache = synthetic_stream(42, 1, device(), batches=1, batch_size=2)
    operator, stats = build_from_batches(model, method, cache, 42, 1)
    assert set(operator.data) == names
    assert set(stats["preconditioned_layers"]) == names
    assert {"pos_embed", "cls_token", "norm", "blocks.0.norm1", "blocks.0.norm2"} <= set(
        Clipper(model, operator).identity_geometry_layers
    )


def test_a_covariance_matrix_function_and_rms_matching():
    a = torch.tensor([[1., 0.], [0., 2.], [1., 2.]])
    model = nn.Linear(2, 3, bias=True)
    factors = {"x": {"A": torch.cat((a, torch.ones(3, 1)), dim=1).T @
                       torch.cat((a, torch.ones(3, 1)), dim=1) / 3,
               "output_dimension": 3}}
    operator = AOnlyOperator(factors, power=.4, damping=1e-3)
    eig, q = torch.linalg.eigh(factors["x"]["A"].double())
    expected = (q * (eig.clamp_min(0) + 1e-3).pow(-.4)) @ q.T
    torch.testing.assert_close(operator.data["x"].double(), expected, rtol=1e-6, atol=1e-6)
    raw = 3 * (eig.clamp_min(0) * (eig.clamp_min(0) + 1e-3).pow(-.8)).sum().item()
    ref = 3 * (eig.clamp_min(0) * (eig.clamp_min(0) + 1e-3).pow(-1)).sum().item()
    assert operator.scale ** 2 * raw == pytest.approx(ref, rel=1e-12)
    c = factors["x"]["A"]
    torch.testing.assert_close(matrix_function(c, .4), expected.to(c.dtype), rtol=1e-6, atol=1e-6)


def test_full_kfc_transform_matches_explicit_formula():
    torch.manual_seed(4)
    a = torch.randn(5, 4)
    g = torch.randn(5, 3)
    factors = {"x": {"A": a.T @ a / 5, "G": g.T @ g / 5, "output_dimension": 3}}
    operator = FullKFACOperator(factors, damping=1e-3)
    grad = torch.randn(7, 3, 4)
    ug, ua = operator.data["x"]
    expected = ug @ grad @ ua
    torch.testing.assert_close(operator.transform_gradient("x", grad), expected, rtol=1e-6, atol=1e-6)
    assert not hasattr(operator, "scale")


def test_full_builder_records_bt_covariances():
    model = small_model()
    cache = [torch.randn(2, 1, 8, 8, device=device())]
    op, _ = build_from_batches(model, "dp_kfc", cache, 42, 1)
    module = model.patch_embed
    a = flatten_linear_input(torch.nn.functional.unfold(cache[0], 4, stride=4).transpose(1, 2), module)
    torch.testing.assert_close(op.factors["patch_embed"]["A"], a.T @ a / a.shape[0], rtol=1e-5, atol=1e-6)
    assert op.factors["patch_embed"]["G"].shape == (8, 8)


def test_full_builder_g_and_bias_augmented_covariances_match_independent_hooks():
    model = small_model(33)
    x = torch.randn(2, 1, 8, 8, device=device())
    op, _ = build_from_batches(model, "dp_kfc", [x], 44, 2)
    names = ("patch_embed", "blocks.0.attn.q_proj", "blocks.0.mlp.fc1", "head")
    activations, backprops = {}, {}
    handles = []
    modules = dict(model.named_modules())
    for name in names:
        def capture(module, args, output, name=name):
            activations[name] = args[0].detach()
            output.register_hook(lambda grad, name=name: backprops.__setitem__(name, grad.detach()))
        handles.append(modules[name].register_forward_hook(capture))
    labels = synthetic_labels(44, 2, x.device, len(x), 3)
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x), labels, reduction="sum").backward()
    for handle in handles:
        handle.remove()
    for name in names:
        module = modules[name]
        a = flatten_linear_input(activations[name], module)
        b = backprops[name].reshape(-1, module.out_features)
        torch.testing.assert_close(op.factors[name]["A"], a.T @ a / a.shape[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(op.factors[name]["G"], b.T @ b / b.shape[0], rtol=1e-5, atol=1e-6)
    model.zero_grad(set_to_none=True)


def test_full_builder_uses_one_continuous_label_generator(monkeypatch):
    import torch.nn.functional as functional

    model = small_model(35)
    cache = [torch.randn(64, 1, 8, 8, device=device()), torch.randn(64, 1, 8, 8, device=device())]
    seen = []
    original = functional.cross_entropy

    def capture(input, target, *args, **kwargs):
        seen.append(target.detach().cpu())
        return original(input, target, *args, **kwargs)

    monkeypatch.setattr(functional, "cross_entropy", capture)
    build_from_batches(model, "dp_kfc", cache, 73, 4)
    expected = torch.cat(synthetic_label_stream(73, 4, device(), [64, 64], 3)).cpu()
    actual = torch.cat(seen)
    assert torch.equal(actual, expected)
    assert len(seen) == 2


@pytest.mark.parametrize("method", ["dp_sgd", "dp_kfc_a_bk", "dp_kfc"])
def test_physical_batch_equivalence_including_adamw_update(method):
    base = small_model(31)
    cache = [torch.randn(2, 1, 8, 8, device=device())]
    op = None if method == "dp_sgd" else build_from_batches(base, method, cache, 31, 1)[0]
    x = torch.randn(4, 1, 8, 8, device=device())
    y = torch.tensor([0, 1, 2, 1], device=device())
    results = []
    for physical in (4, 2, 1):
        model = copy.deepcopy(base)
        clipper = Clipper(model, copy.deepcopy(op), max_grad_norm=1.)
        loss, norms, factors, layer_sq, stats = clipper.aggregate_logical(x, y, physical)
        clipped_gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        noise = torch.Generator(device=device()).manual_seed(40031)
        # Compare the private update with paired DP noise. With zero noise,
        # AdamW amplifies FP32 reduction differences on near-zero gradients.
        # The clipped sums above are still compared before adding noise.
        clipper.step(optimizer, 0.25, len(x), noise)
        results.append((loss, norms, factors, layer_sq, clipped_gradients,
                        {n: p.detach().clone() for n, p in model.named_parameters()}, stats))
    for left, right in zip(results, results[1:]):
        torch.testing.assert_close(left[0], right[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(left[1], right[1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(left[2], right[2], rtol=2e-5, atol=2e-6)
        for name in left[4]:
            torch.testing.assert_close(left[4][name], right[4][name], rtol=2e-4, atol=2e-5)
        for name in left[5]:
            torch.testing.assert_close(left[5][name], right[5][name], rtol=1e-3, atol=1e-4)
    assert results[0][6]["backward_calls"] == 1
    assert results[1][6]["backward_calls"] == 2
    assert results[2][6]["backward_calls"] == 4


def test_one_logical_batch_has_one_step_noise_and_accountant():
    from opacus.accountants import RDPAccountant

    model = nn.Linear(4, 3).to(device())
    x = torch.randn(256, 4, device=device())
    y = torch.randint(3, (256,), device=device())
    clipper = Clipper(model)
    clipper.aggregate_logical(x, y, 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sigma = 0.25
    generator = torch.Generator(device=device()).manual_seed(40008)
    expected_generator = torch.Generator(device=device()).manual_seed(40008)
    for parameter in clipper.parameters:
        torch.randn(
            parameter.numel(), device=parameter.device, dtype=parameter.dtype,
            generator=expected_generator,
        )
    clipper.step(optimizer, sigma, 256, generator)
    accountant = RDPAccountant()
    accountant.step(noise_multiplier=sigma, sample_rate=256 / 50_000)
    assert clipper.optimizer_steps == 1
    assert clipper.noise_events == 1
    assert clipper.backward_calls == 1
    assert len(accountant.history) == 1
    assert sum(value[2] for value in accountant.history) == 1
    assert accountant.history[0][1] == pytest.approx(256 / 50_000)
    assert sigma > 0
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_synthetic_label_stream_is_continuous_and_rng_isolated():
    stream = synthetic_label_stream(42, 1, "cpu", [7, 7], 10)
    reference = torch.Generator().manual_seed(42 + 20000 + 1)
    expected = torch.randint(10, (14,), generator=reference).split(7)
    assert all(torch.equal(left, right) for left, right in zip(stream, expected))
    assert not torch.equal(stream[0], stream[1])
    cpu_before = torch.random.get_rng_state().clone()
    synthetic_label_stream(42, 1, "cpu", [256, 256], 10)
    assert torch.equal(cpu_before, torch.random.get_rng_state())


def test_accuracy_auc_matches_exp20_trapezoid_definition():
    epochs = [1, 2, 3, 5]
    accuracies = [0.1, 0.3, 0.2, 0.6]
    assert accuracy_auc(epochs, accuracies) == pytest.approx(
        1.25
    )
    assert accuracy_auc([1], [0.4]) == 0.0


def _independent_oracle(model, operator, x, y):
    """Independent single-example backward oracle; no Clipper helpers."""
    values = []
    layer_values = {}
    group_values = []
    for index in range(len(x)):
        reference = copy.deepcopy(model)
        reference.zero_grad(set_to_none=True)
        logits = reference(x[index:index + 1])
        torch.nn.functional.cross_entropy(logits, y[index:index + 1]).backward()
        transformed = {}
        sample_layers = {}
        for name, module in reference.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            gradient = module.weight.grad
            if module.bias is not None:
                gradient = torch.cat((gradient, module.bias.grad[:, None]), dim=1)
            if operator is not None and name in operator.data:
                gradient = operator.transform_gradient(name, gradient)
            transformed[f"{name}.weight"] = gradient[:, :-1] if module.bias is not None else gradient
            if module.bias is not None:
                transformed[f"{name}.bias"] = gradient[:, -1]
            sample_layers[name] = gradient.square().sum()
        for name, parameter in reference.named_parameters():
            if name not in transformed:
                transformed[name] = parameter.grad
                sample_layers[name] = parameter.grad.square().sum()
        norm = torch.stack([value.square().sum() for value in transformed.values()]).sum().sqrt()
        factor = (1.0 / (norm + 1e-6)).clamp(max=1.0)
        groups = {}
        for name, value in sample_layers.items():
            if name.endswith(("q_proj", "k_proj", "v_proj")):
                group = "attention_qkv"
            elif name.endswith("attn.out_proj"):
                group = "attention_out"
            elif name.endswith(("mlp.fc1", "mlp.fc2")):
                group = "mlp"
            elif name in ("patch_embed", "head"):
                group = "patch_head"
            else:
                group = "identity"
            groups[group] = groups.get(group, 0) + value
        group_values.append(groups)
        values.append((norm, factor, {name: factor * gradient for name, gradient in transformed.items()}, sample_layers))
    aggregate = {}
    aggregate_layers = {}
    for _, _, gradients, sample_layers in values:
        for name, gradient in gradients.items():
            aggregate[name] = aggregate.get(name, 0) + gradient
        for name, value in sample_layers.items():
            aggregate_layers.setdefault(name, []).append(value)
    aggregate_groups = {}
    for groups in group_values:
        for name, value in groups.items():
            aggregate_groups.setdefault(name, []).append(value)
    return (
        torch.stack([value[0] for value in values]),
        torch.stack([value[1] for value in values]),
        aggregate,
        {name: torch.stack(values) for name, values in aggregate_layers.items()},
        {name: torch.stack(values) for name, values in aggregate_groups.items()},
    )


@pytest.mark.parametrize("method", ["dp_sgd", "dp_kfc_a_bk", "dp_kfc"])
def test_bk_matches_independent_per_sample_oracle(method):
    model = small_model(51)
    x = torch.randn(3, 1, 8, 8, device=device())
    y = torch.tensor([0, 1, 2], device=device())
    cache = [torch.randn(2, 1, 8, 8, device=device())]
    operator = None if method == "dp_sgd" else build_from_batches(model, method, cache, 51, 1)[0]
    expected_norms, expected_factors, expected_aggregate, expected_layers, expected_groups = _independent_oracle(model, operator, x, y)
    clipper = Clipper(model, operator, method=method)
    _, norms, factors, _, stats = clipper.aggregate(x, y)
    torch.testing.assert_close(norms, expected_norms, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(factors, expected_factors, rtol=2e-4, atol=2e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_aggregate[name], rtol=2e-4, atol=2e-5)
    _, _, _, actual_layers, actual_stats = clipper.aggregate(x, y)
    for name, value in expected_layers.items():
        if name in actual_layers:
            torch.testing.assert_close(actual_layers[name], value, rtol=2e-4, atol=2e-5)
    for name, value in expected_groups.items():
        torch.testing.assert_close(actual_stats["layer_group_sq"][name], value, rtol=2e-4, atol=2e-5)
    assert stats["bk_cache_bytes"] > 0
    assert stats["temporary_per_sample_grad_bytes"] < len(x) * sum(p.numel() for p in model.parameters()) * 4
    assert stats["cache_empty_after_step"]


def test_rng_pairing_for_initialization_synthetic_labels_and_noise():
    models = [initialize(42, "cpu") for _ in cfg.METHODS]
    for model in models[1:]:
        for name, value in models[0].state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
    different_seed = initialize(7, "cpu")
    for name, value in models[0].state_dict().items():
        if name.startswith("head."):
            assert not torch.equal(value, different_seed.state_dict()[name])
        else:
            assert torch.equal(value, different_seed.state_dict()[name])
    images = [next(iter(synthetic_stream(42, 1, "cpu", batches=1, batch_size=3))) for _ in cfg.METHODS]
    assert all(torch.equal(images[0], image) for image in images[1:])
    labels = [synthetic_labels(42, 1, "cpu", 8) for _ in cfg.METHODS]
    assert all(torch.equal(labels[0], value) for value in labels[1:])
    draws = []
    for model in models:
        generator = torch.Generator().manual_seed(40042)
        draws.append([torch.randn(p.numel(), generator=generator) for p in model.parameters()])
    for other in draws[1:]:
        for left, right in zip(draws[0], other):
            assert torch.equal(left, right)


def test_private_bk_path_keeps_model_dtype_and_no_per_example_linear_helper():
    import exp22.handlers as handlers

    assert not hasattr(handlers, "per_example_linear_gradient")
    model = nn.Linear(7, 6).float()
    clipper = Clipper(model)
    aggregate = clipper._empty_aggregate()
    assert all(value.dtype == parameter.dtype == torch.float32
               for parameter, value in aggregate.items())

    torch.manual_seed(2201)
    z = torch.randn(4, 5, 7)
    b = torch.randn(4, 5, 6)
    factors = torch.rand(4)
    result, temporary_bytes = Clipper._linear_aggregate(z, b, factors)
    expected = torch.einsum("b,bto,bti->oi", factors, b, z)
    assert result.dtype == z.dtype == b.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    assert temporary_bytes >= result.numel() * result.element_size()


@pytest.mark.parametrize("tokens", [1, 5])
def test_linear_bias_reuses_augmented_aggregate_column(monkeypatch, tokens):
    torch.manual_seed(2208)
    model = nn.Sequential(nn.Linear(4, 3))
    clipper = Clipper(model)
    original = clipper._linear_aggregate
    reconstructed = []

    def capture(z, b, factors):
        clipped, workspace = original(z, b, factors)
        # A sentinel distinguishes reuse from a second bias reconstruction.
        clipped[:, -1].add_(0.125)
        reconstructed.append(clipped.clone())
        return clipped, workspace

    monkeypatch.setattr(clipper, "_linear_aggregate", capture)
    x = torch.randn(3, tokens, 4)
    y = torch.zeros(3, dtype=torch.long)
    clipper.aggregate(x, y, loss_fn=lambda logits, target: logits.square().sum((1, 2)))
    assert len(reconstructed) == 1
    torch.testing.assert_close(model[0].weight.grad, reconstructed[0][:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(model[0].bias.grad, reconstructed[0][:, -1], rtol=0, atol=0)


def test_bk_workspace_bytes_include_gram_and_reconstruction_payload():
    torch.manual_seed(2202)
    z = torch.randn(4, 5, 7)
    b = torch.randn(4, 5, 6)
    _, gram_temporary_bytes = Clipper._linear_norm_squared(z, b)
    rows = min(8, max(1, b.shape[1] - 1))
    b_gram = b[:, :rows] @ b[:, :rows].transpose(1, 2)
    z_gram = z[:, :rows] @ z[:, :rows].transpose(1, 2)
    gram_payload = (
        b_gram.numel() * b_gram.element_size()
        + z_gram.numel() * z_gram.element_size()
    )
    assert gram_temporary_bytes >= gram_payload

    factors = torch.rand(4)
    aggregate, aggregate_temporary_bytes = Clipper._linear_aggregate(z, b, factors)
    weighted_b = b * factors[:, None, None]
    reconstruction_payload = (
        weighted_b.numel() * weighted_b.element_size()
        + aggregate.numel() * aggregate.element_size()
    )
    assert aggregate_temporary_bytes >= reconstruction_payload


def test_logical_aggregate_uses_one_shared_buffer_and_returns_no_aggregate(monkeypatch):
    torch.manual_seed(2203)
    model = nn.Linear(4, 3)
    clipper = Clipper(model)
    original = clipper._one_batch
    seen_aggregate_ids = []

    def wrapped(x, y, aggregate, loss_fn=None):
        seen_aggregate_ids.append(id(aggregate))
        part = original(x, y, aggregate, loss_fn)
        assert "aggregate" not in part
        return part

    monkeypatch.setattr(clipper, "_one_batch", wrapped)
    x = torch.randn(8, 4)
    y = torch.randint(3, (8,))
    clipper.aggregate_logical(x, y, physical_batch_size=2)
    assert len(seen_aggregate_ids) == 4
    assert len(set(seen_aggregate_ids)) == 1


def test_linear_bk_primitives_match_explicit_oracles_without_materialization():
    torch.manual_seed(2204)
    z = torch.randn(3, 4, 5)
    b = torch.randn(3, 4, 6)
    factors = torch.rand(3)
    expected_gradient = torch.einsum("bto,bti->boi", b, z)
    expected_sq = expected_gradient.square().sum((1, 2))
    actual_sq, _ = Clipper._linear_norm_squared(z, b)
    torch.testing.assert_close(actual_sq, expected_sq, rtol=1e-5, atol=1e-6)

    expected_aggregate = torch.einsum("b,bto,bti->oi", factors, b, z)
    actual_aggregate, _ = Clipper._linear_aggregate(z, b, factors)
    torch.testing.assert_close(actual_aggregate, expected_aggregate, rtol=1e-5, atol=1e-6)


def test_full_builder_reverse_vectors_count_samples_not_vjp_calls():
    model = small_model(2205)
    cache = [torch.randn(2, 1, 8, 8, device=device()), torch.randn(2, 1, 8, 8, device=device())]
    _, stats = build_from_batches(model, "dp_kfc", cache, 2205, 1)
    assert stats["builder_vjp_calls"] == 2
    assert stats["builder_reverse_vectors"] == 4
    assert stats["builder_samples"] == 4
    assert stats["builder_reverse_vectors"] == stats["builder_samples"]
    assert stats["builder_vjp_calls"] < stats["builder_reverse_vectors"]
    assert cfg.SYNTHETIC_BATCHES * cfg.SYNTHETIC_BATCH_SIZE == 2560
    assert cfg.SYNTHETIC_BATCHES * cfg.SYNTHETIC_BATCH_SIZE // cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE == 10

    _, dp_stats = build_from_batches(model, "dp_sgd", cache, 2205, 1)
    assert {name: dp_stats[name] for name in (
        "builder_forward_calls", "builder_logical_batches", "builder_vjp_calls",
        "builder_reverse_vectors", "builder_samples", "operator_state_bytes",
    )} == {
        "builder_forward_calls": 0, "builder_logical_batches": 0,
        "builder_vjp_calls": 0, "builder_reverse_vectors": 0,
        "builder_samples": 0, "operator_state_bytes": 0,
    }


def test_layer_strategy_metadata_matches_production_paths():
    model = small_model(2206)
    clipper = Clipper(model)
    x = torch.randn(2, 1, 8, 8, device=device())
    y = torch.tensor([0, 1], device=device())
    _, _, _, _, stats = clipper.aggregate(x, y)
    for name in clipper.linear_modules:
        assert stats["layer_strategies"][name] == "bk_ghost"
    for name in clipper.norm_modules:
        assert stats["layer_strategies"][name] == "identity_analytic"
    assert stats["layer_strategies"]["pos_embed"] == "identity_direct"
    assert stats["layer_strategies"]["cls_token"] == "identity_direct"


def _write_analysis_run(root, method, seed, value):
    run = root / "runs" / f"{method}_{seed}"
    run.mkdir(parents=True)
    pd.DataFrame([{
        "method": method, "seed": seed, "epoch": 1, "train_loss": value,
    }]).to_csv(run / "metrics.csv", index=False)


def test_single_seed_analysis_uses_nan_for_std_and_ci(tmp_path):
    for index, method in enumerate(cfg.METHODS):
        _write_analysis_run(tmp_path, method, 42, float(index + 1))
    analyze(tmp_path)
    method_summary = pd.read_csv(tmp_path / "method_summary.csv")
    paired_summary = pd.read_csv(tmp_path / "paired_summary.csv")
    assert set(method_summary["n_seeds"]) == {1}
    assert method_summary["train_loss_sample_std"].isna().all()
    assert paired_summary["n_seeds"].eq(1).all()
    assert paired_summary["sample_std"].isna().all()
    assert paired_summary["bootstrap_ci95_low"].isna().all()
    assert paired_summary["bootstrap_ci95_high"].isna().all()
    assert paired_summary["bootstrap_seed"].isna().all()


def test_multi_seed_analysis_bootstrap_is_finite_and_paired(tmp_path):
    for seed in (42, 7, 123):
        for index, method in enumerate(cfg.METHODS):
            _write_analysis_run(tmp_path, method, seed, float(seed + index))
    analyze(tmp_path)
    method_summary = pd.read_csv(tmp_path / "method_summary.csv")
    paired_summary = pd.read_csv(tmp_path / "paired_summary.csv")
    assert set(method_summary["n_seeds"]) == {3}
    assert np.isfinite(method_summary["train_loss_sample_std"]).all()
    assert paired_summary["n_seeds"].eq(3).all()
    assert np.isfinite(paired_summary["sample_std"]).all()
    assert np.isfinite(paired_summary["bootstrap_ci95_low"]).all()
    assert np.isfinite(paired_summary["bootstrap_ci95_high"]).all()
    assert (paired_summary["bootstrap_ci95_low"] <= paired_summary["bootstrap_ci95_high"]).all()
    assert paired_summary["bootstrap_seed"].eq(2209).all()


def test_pretrained_full_model_bk_matches_single_example_backward():
    model = initialize(19, device())
    x = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device())
    y = torch.tensor([2, 8], device=device())
    expected_norms, expected_factors, expected_gradients, _, _ = _independent_oracle(model, None, x, y)
    clipper = Clipper(model)
    _, norms, factors, _, stats = clipper.aggregate_logical(x, y, 2)
    torch.testing.assert_close(norms, expected_norms, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(factors, expected_factors, rtol=3e-4, atol=3e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=3e-4, atol=3e-5)
    assert stats["layer_strategies"]["cls_token"] == "identity_direct"
    assert stats["fallback_temporary_grad_bytes"] == 0


def test_private_shuffle_pairing_and_formal_privacy_budget():
    from exp22.run_one import private_loader, _sigma
    from opacus.accountants import RDPAccountant

    dataset = torch.arange(3 * cfg.LOGICAL_BATCH_SIZE)
    loaders = [private_loader(dataset, 42) for _ in cfg.METHODS]
    for _ in range(2):
        orders = [torch.cat(list(loader)) for loader in loaders]
        assert all(torch.equal(orders[0], order) for order in orders[1:])
    assert cfg.EPOCHS == 10 and cfg.LEARNING_RATE == 1e-4
    assert cfg.LOGICAL_BATCH_SIZE == cfg.PHYSICAL_BATCH_SIZE == 256
    assert cfg.ACCUMULATION_STEPS == 1
    steps = cfg.EPOCHS * (cfg.TRAIN_SAMPLES // cfg.LOGICAL_BATCH_SIZE)
    assert steps == 1950
    sigma = _sigma(steps, cfg.TRAIN_SAMPLES, False)
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=cfg.LOGICAL_BATCH_SIZE / cfg.TRAIN_SAMPLES)
    assert sigma > 0
    assert accountant.get_epsilon(cfg.DELTA) <= cfg.EPSILON


def test_patch_qkv_conversion_and_cls_pooling():
    import timm
    source = timm.create_model(cfg.MODEL_NAME, pretrained=True).to(device()).double().eval()
    model = convert_vit(source).eval()
    x = torch.randn(2, 3, 224, 224, device=device(), dtype=torch.float64)
    with torch.no_grad():
        patches = model.patch_embed(model.patchify(x))
        torch.testing.assert_close(patches, source.patch_embed(x), rtol=1e-10, atol=1e-12)
        tokens = torch.randn(2, 197, 192, device=device(), dtype=torch.float64)
        for old, new in zip(source.blocks, model.blocks):
            projected = torch.cat([getattr(new.attn, name)(tokens)
                                   for name in ("q_proj", "k_proj", "v_proj")], dim=-1)
            torch.testing.assert_close(projected, old.attn.qkv(tokens), rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(new.attn.out_proj.weight, old.attn.proj.weight, rtol=0, atol=0)
            torch.testing.assert_close(new.attn.out_proj.bias, old.attn.proj.bias, rtol=0, atol=0)
        seen = {}
        h1 = model.norm.register_forward_hook(lambda m, a, out: seen.__setitem__("norm", out))
        h2 = model.head.register_forward_pre_hook(lambda m, a: seen.__setitem__("head", a[0]))
        assert model(x).shape == (2, 1000)
        h1.remove()
        h2.remove()
        torch.testing.assert_close(seen["head"], seen["norm"][:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("method", cfg.METHODS)
def test_default_builder_counters_and_release_before_next_batch(method):
    import weakref
    model = nn.Linear(4, 10).to(device())
    previous = []

    def batches():
        for _ in range(cfg.SYNTHETIC_BATCHES):
            assert all(reference() is None for reference in previous)
            x = torch.randn(cfg.SYNTHETIC_BATCH_SIZE, 4, device=device())
            previous.append(weakref.ref(x))
            yield x
            del x

    _, stats = build_from_batches(model, method, batches(), 42, 1)
    assert cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE == 256
    expected = {
        "builder_forward_calls": 0 if method == "dp_sgd" else 10,
        "builder_logical_batches": 0 if method == "dp_sgd" else 10,
        "builder_vjp_calls": 10 if method == "dp_kfc" else 0,
        "builder_reverse_vectors": 2560 if method == "dp_kfc" else 0,
        "builder_samples": 0 if method == "dp_sgd" else 2560,
    }
    for name, value in expected.items():
        assert stats[name] == value
    assert all(reference() is None for reference in previous)


def test_synthetic_image_stream_matches_baseline_continuous_rng():
    from dp_kfac.optimizer import generate_pink_noise
    target_device = device()
    devices = [target_device.index] if target_device.type == "cuda" else []
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(target_device).clone() if devices else None
    images = list(synthetic_stream(42, 3, target_device, batches=2, batch_size=2))
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    if devices:
        assert torch.equal(cuda_before, torch.cuda.get_rng_state(target_device))
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(42 + 10000 + 3)
        if devices:
            torch.cuda.default_generators[target_device.index].manual_seed(42 + 10000 + 3)
        expected = [generate_pink_noise(2, (3, 224, 224), target_device, alpha=1.0) for _ in range(2)]
    for actual, reference in zip(images, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    assert not torch.equal(images[0], images[1])


def test_synthetic_stream_is_lazy_and_retains_no_previous_batch():
    import weakref
    generator = iter(synthetic_stream(42, 1, "cpu", batches=2, batch_size=1))
    first = next(generator)
    reference = weakref.ref(first)
    del first
    assert reference() is None
    second = next(generator)
    assert second.shape == (1, 3, 224, 224)


def test_token_parameters_enter_global_clipping_and_reconstruction():
    model = small_model(29)
    x = torch.randn(2, 1, 8, 8, device=device())
    y = torch.tensor([0, 2], device=device())
    expected = _independent_oracle(model, None, x, y)
    clipper = Clipper(model)
    _, norms, factors, layers, _ = clipper.aggregate(x, y)
    for name in ("cls_token", "pos_embed"):
        assert layers[name].gt(0).all()
        torch.testing.assert_close(layers[name], expected[3][name], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(getattr(model, name).grad, expected[2][name], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(norms.square(), sum(layers.values()), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(factors, expected[1], rtol=2e-4, atol=2e-5)


def test_cifar_preprocessing_has_exact_resize_and_no_augmentation():
    from exp22.run_one import data_transform
    from torchvision import transforms
    from PIL import Image
    transform = data_transform()
    resize, tensor, normalize = transform.transforms
    assert isinstance(resize, transforms.Resize)
    assert resize.size == (224, 224)
    assert resize.interpolation == transforms.InterpolationMode.BICUBIC
    assert isinstance(tensor, transforms.ToTensor)
    assert normalize.mean == normalize.std == (0.5, 0.5, 0.5)
    x = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    assert transform(x).shape == (3, 224, 224)
    assert torch.equal(transform(x), transform(x))


def test_production_linear_path_never_allocates_sample_weight_tensor():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class NoSampleWeights(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            for value in tree_leaves(output):
                if isinstance(value, torch.Tensor):
                    assert tuple(value.shape) not in {(3, 7, 5), (3, 7, 6)}
            return output

    model = nn.Linear(5, 7)
    clipper = Clipper(model)
    x, y = torch.randn(3, 2, 5), torch.zeros(3, dtype=torch.long)
    with NoSampleWeights():
        clipper.aggregate(x, y, loss_fn=lambda output, target: output.square().sum((1, 2)))

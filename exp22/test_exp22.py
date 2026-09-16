"""Exp22 correctness tests.  They use small CPU tensors unless CUDA is needed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from exp22.geometry import (
    AOnlyOperator,
    FullKFACOperator,
    build_from_cache,
    flatten_linear_input,
    matrix_function,
    synthetic_cache,
    synthetic_labels,
)
from exp22.methods import Clipper
from exp22.model import TinyViT, convert_tinyvit, initialize


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def small_model(seed=11):
    torch.manual_seed(seed)
    from dp_kfac.models import TinyViT as PackedTinyViT

    return convert_tinyvit(PackedTinyViT(
        img_size=8, patch_size=4, in_channels=1, embed_dim=8,
        num_heads=2, num_blocks=1, num_classes=3,
    )).to(device())


def test_explicit_attention_conversion_preserves_logits():
    torch.manual_seed(17)
    from dp_kfac.models import TinyViT as PackedTinyViT

    old = PackedTinyViT(img_size=32, patch_size=8, in_channels=3, embed_dim=64,
                        num_heads=4, num_blocks=2, num_classes=10)
    new = convert_tinyvit(old)
    x = torch.randn(3, 3, 32, 32)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=1e-6)


def test_exact_linear_coverage_and_identity_layers():
    model = initialize(42, "cpu")
    names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    assert len(names) == 12
    expected_fragments = {"patch_embed", "head"}
    expected_fragments |= {f"blocks.{i}.attn.{part}_proj" for i in range(2) for part in ("q", "k", "v", "out")}
    expected_fragments |= {f"blocks.{i}.ffn" for i in range(2)}
    assert names == expected_fragments
    cache = synthetic_cache(42, 1, "cpu", batches=1, batch_size=2)
    operator, stats = build_from_cache(model, "dp_kfc_a_bk", cache, 42, 1)
    assert set(operator.data) == names
    assert set(stats["preconditioned_layers"]) == names
    assert {"pos_embed", "norm", "blocks.0.norm1", "blocks.0.norm2"} <= set(
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
    op, _ = build_from_cache(model, "dp_kfc", cache, 42, 1)
    module = model.patch_embed
    a = flatten_linear_input(cache[0].unfold(2, 4, 4).unfold(3, 4, 4).contiguous().view(2, -1, 16), module)
    torch.testing.assert_close(op.factors["patch_embed"]["A"], a.T @ a / a.shape[0], rtol=1e-5, atol=1e-6)
    assert op.factors["patch_embed"]["G"].shape == (8, 8)


def test_physical_batch_equivalence_including_adamw_update():
    base = small_model(31)
    cache = [torch.randn(2, 1, 8, 8, device=device())]
    op, _ = build_from_cache(base, "dp_kfc_a_bk", cache, 31, 1)
    x = torch.randn(4, 1, 8, 8, device=device())
    y = torch.tensor([0, 1, 2, 1], device=device())
    results = []
    for physical in (4, 2, 1):
        model = copy.deepcopy(base)
        clipper = Clipper(model, copy.deepcopy(op), max_grad_norm=1.)
        loss, norms, factors, layer_sq, stats = clipper.aggregate_logical(x, y, physical)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        noise = torch.Generator(device=device()).manual_seed(40031)
        clipper.step(optimizer, 0., len(x), noise)
        results.append((loss, norms, factors, layer_sq, {n: p.detach().clone() for n, p in model.named_parameters()}, stats))
    for left, right in zip(results, results[1:]):
        torch.testing.assert_close(left[0], right[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(left[1], right[1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(left[2], right[2], rtol=2e-5, atol=2e-6)
        for name in left[4]:
            torch.testing.assert_close(left[4][name], right[4][name], rtol=2e-5, atol=2e-6)
    assert results[0][5]["backward_calls"] == 1
    assert results[1][5]["backward_calls"] == 2
    assert results[2][5]["backward_calls"] == 4


def test_one_logical_batch_has_one_step_noise_and_accountant():
    model = small_model(8)
    x = torch.randn(4, 1, 8, 8, device=device())
    y = torch.tensor([0, 1, 2, 0], device=device())
    clipper = Clipper(model)
    clipper.aggregate_logical(x, y, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device()).manual_seed(40008)
    before = generator.get_state().clone()
    clipper.step(optimizer, 0.0, 4, generator)
    reference = torch.Generator(device=device()).manual_seed(40008)
    for p in model.parameters():
        torch.randn(p.numel(), device=p.device, dtype=p.dtype, generator=reference)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert clipper.optimizer_steps == 1
    assert clipper.backward_calls == 2


def test_rng_pairing_for_initialization_synthetic_labels_and_noise():
    models = [initialize(42, "cpu") for _ in range(3)]
    for first, second in zip(models[0].parameters(), models[1].parameters()):
        assert torch.equal(first, second)
    caches = [synthetic_cache(42, 1, "cpu", batches=1, batch_size=3) for _ in range(3)]
    assert torch.equal(caches[0][0], caches[1][0])
    labels = [synthetic_labels(42, 1, "cpu", 8) for _ in range(3)]
    assert torch.equal(labels[0], labels[1])
    draws = []
    for model in models:
        generator = torch.Generator().manual_seed(40042)
        draws.append([torch.randn(p.numel(), generator=generator) for p in model.parameters()])
    for left, right in zip(draws[0], draws[1]):
        assert torch.equal(left, right)


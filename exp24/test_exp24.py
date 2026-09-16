"""Exp24 protocol and numerical correctness tests."""

from __future__ import annotations

import copy
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
    path = ROOT / "exp24" / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

from exp24 import config as cfg
from exp24.analyze import accuracy_auc, analyze
from exp24.geometry import (
    FullKFACOperator,
    AOnlyOperator,
    build_from_batches,
    flatten_linear_input,
    synthetic_label_stream,
    synthetic_labels,
    synthetic_stream,
)
from exp24.methods import Clipper, make_optimizer
from exp24.model import convert_vit, initialize

torch.set_num_threads(4)


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FrozenHeadModel(nn.Module):
    """Small model with the same trainable/frozen parameter topology as Exp24."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 192)
        self.norm = nn.LayerNorm(192)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, 192))
        self.head = nn.Linear(192, 10)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self.num_classes = 10
        self.eval()

    def forward(self, x):
        feature = self.norm(self.patch_embed(x))
        feature = feature + self.cls_token[:, 0] + self.pos_embed[:, 0]
        return self.head(feature)


def small_model():
    return FrozenHeadModel().to(device())


def _operator(model, method):
    calibration = [torch.randn(8, 4, device=device())]
    return build_from_batches(model, method, calibration, 42, 1)[0]


def _oracle(model, operator, x, y):
    gradients = []
    norms = []
    for index in range(len(x)):
        reference = copy.deepcopy(model)
        reference.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(reference(x[index:index + 1]), y[index:index + 1]).backward()
        gradient = reference.head.weight.grad
        gradient = torch.cat((gradient, reference.head.bias.grad[:, None]), dim=1)
        if operator is not None:
            gradient = operator.transform_gradient("head", gradient)
        norm = gradient.square().sum().sqrt()
        factor = (1.0 / (norm + 1e-6)).clamp(max=1.0)
        gradients.append(factor * gradient)
        norms.append(norm)
    aggregate = torch.stack(gradients).sum(0)
    return torch.stack(norms), aggregate


def test_explicit_pretrained_conversion_preserves_logits_and_qkv():
    import timm

    source = timm.create_model(cfg.MODEL_NAME, pretrained=True).to(device()).eval()
    converted = convert_vit(source).eval()
    x = torch.randn(2, 3, 224, 224, device=device())
    with torch.no_grad():
        torch.testing.assert_close(source(x), converted(x), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        converted.patch_embed.weight,
        source.patch_embed.proj.weight.flatten(1),
        rtol=0,
        atol=0,
    )
    for old, new in zip(source.blocks, converted.blocks):
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            torch.testing.assert_close(
                getattr(new.attn, name).weight,
                old.attn.qkv.weight.chunk(3, dim=0)[index],
                rtol=0,
                atol=0,
            )


def test_exact_head_only_trainability_and_optimizer_ids():
    model = initialize(42, device())
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert names == {"head.weight", "head.bias"}
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 1930
    optimizer = make_optimizer(model)
    ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert ids == {id(parameter) for parameter in model.head.parameters()}


def test_clipper_and_geometry_exclude_frozen_backbone():
    model = small_model()
    clipper = Clipper(model)
    assert set(clipper.linear_modules) == {"head"}
    assert clipper.norm_modules == {}
    assert "norm" not in clipper.identity_geometry_layers
    assert "cls_token" not in clipper.identity_geometry_layers
    assert "pos_embed" not in clipper.identity_geometry_layers
    a_operator = _operator(model, "dp_kfc_a_bk")
    full_operator = _operator(model, "dp_kfc")
    assert a_operator.data.keys() == {"head"}
    assert full_operator.data.keys() == {"head"}
    assert a_operator.data["head"].shape == (193, 193)
    assert full_operator.data["head"][0].shape == (10, 10)
    assert full_operator.data["head"][1].shape == (193, 193)


@pytest.mark.parametrize("method", cfg.METHODS)
def test_bk_norm_and_aggregate_match_independent_per_example_oracle(method):
    model = small_model()
    operator = None if method == "dp_sgd" else _operator(model, method)
    x = torch.randn(3, 4, device=device())
    y = torch.tensor([0, 1, 2], device=device())
    expected_norms, expected_aggregate = _oracle(model, operator, x, y)
    clipper = Clipper(model, operator, method=method)
    _, actual_norms, _, _, stats = clipper.aggregate(x, y)
    torch.testing.assert_close(actual_norms, expected_norms, rtol=2e-4, atol=2e-5)
    # The two mathematically identical KFC factorizations have different
    # FP32 matmul reduction order on the transformed aggregate.
    torch.testing.assert_close(model.head.weight.grad, expected_aggregate[:, :-1], rtol=3e-4, atol=6e-5)
    torch.testing.assert_close(model.head.bias.grad, expected_aggregate[:, -1], rtol=3e-4, atol=6e-5)
    assert stats["preconditioned_layers"] == ([] if method == "dp_sgd" else ["head"])


def test_production_bk_does_not_materialize_per_example_weight_gradient():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class NoPerExampleGradient(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            for value in tree_leaves(output):
                if isinstance(value, torch.Tensor):
                    assert tuple(value.shape) != (3, 10, 193)
            return output

    model = small_model()
    with NoPerExampleGradient():
        Clipper(model).aggregate(torch.randn(3, 4, device=device()), torch.tensor([0, 1, 2], device=device()))


def test_one_private_step_updates_only_head_and_has_one_event():
    from opacus.accountants import RDPAccountant

    model = small_model()
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    head_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    clipper = Clipper(model)
    x = torch.randn(256, 4, device=device())
    y = torch.randint(10, (256,), device=device())
    clipper.aggregate_logical(x, y, 256)
    optimizer = make_optimizer(model)
    sigma = 0.25
    generator = torch.Generator(device=device()).manual_seed(40042)
    clipper.step(optimizer, sigma, 256, generator)
    accountant = RDPAccountant()
    accountant.step(noise_multiplier=sigma, sample_rate=256 / 50_000)
    assert clipper.backward_calls == 1
    assert clipper.optimizer_steps == 1
    assert clipper.noise_events == 1
    assert sum(value[2] for value in accountant.history) == 1
    assert any(
        not torch.equal(head_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    assert all(torch.equal(frozen[name], parameter.detach()) for name, parameter in model.named_parameters() if name in frozen)


def test_logical_physical_batch_one_backward_and_one_aggregate():
    model = small_model()
    clipper = Clipper(model)
    _, _, _, _, stats = clipper.aggregate_logical(
        torch.randn(256, 4, device=device()),
        torch.randint(10, (256,), device=device()),
        256,
    )
    assert stats["backward_calls"] == 1
    assert stats["accumulation_steps"] == 1


def test_builder_counters_and_streaming_semantics():
    model = small_model()
    def batches():
        for _ in range(cfg.SYNTHETIC_BATCHES):
            yield torch.randn(cfg.SYNTHETIC_BATCH_SIZE, 4, device=device())

    for method in cfg.METHODS:
        _, stats = build_from_batches(model, method, batches(), 42, 1)
        assert stats["builder_forward_calls"] == (0 if method == "dp_sgd" else 10)
        assert stats["builder_logical_batches"] == (0 if method == "dp_sgd" else 10)
        assert stats["builder_samples"] == (0 if method == "dp_sgd" else 2560)
        assert stats["builder_vjp_calls"] == (10 if method == "dp_kfc" else 0)
        assert stats["builder_reverse_vectors"] == (2560 if method == "dp_kfc" else 0)
    assert cfg.SYNTHETIC_BATCHES * cfg.SYNTHETIC_BATCH_SIZE == 2560


def test_same_seed_pairing_and_different_seed_head_only_change():
    models = [initialize(42, "cpu") for _ in cfg.METHODS]
    for model in models[1:]:
        for name, value in models[0].state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
    different = initialize(7, "cpu")
    for name, value in models[0].state_dict().items():
        if name.startswith("head."):
            assert not torch.equal(value, different.state_dict()[name])
        else:
            assert torch.equal(value, different.state_dict()[name])


def test_synthetic_rngs_are_continuous_and_isolated():
    labels = synthetic_label_stream(42, 1, "cpu", [7, 7], 10)
    reference = torch.Generator().manual_seed(42 + 20000 + 1)
    expected = torch.randint(10, (14,), generator=reference).split(7)
    assert all(torch.equal(left, right) for left, right in zip(labels, expected))
    cpu_before = torch.random.get_rng_state().clone()
    images = list(synthetic_stream(42, 1, "cpu", batches=2, batch_size=2))
    synthetic_labels(42, 1, "cpu", 256)
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    assert images[0].shape == (2, 3, 224, 224)
    assert not torch.equal(images[0], images[1])


def test_physical_batch_bk_primitives_have_no_sample_weight_tensor():
    torch.manual_seed(2204)
    z = torch.randn(3, 1, 5)
    b = torch.randn(3, 1, 6)
    factors = torch.rand(3)
    expected = torch.einsum("bto,bti->boi", b, z)
    actual, _ = Clipper._linear_aggregate(z, b, factors)
    torch.testing.assert_close(actual, torch.einsum("b,bto,bti->oi", factors, b, z))
    norm, _ = Clipper._linear_norm_squared(z, b)
    torch.testing.assert_close(norm, expected.square().sum((1, 2)))


def test_formal_rdp_calibration_is_within_epsilon():
    from opacus.accountants import RDPAccountant
    from exp24.run_one import _sigma

    assert cfg.EPOCHS * (cfg.TRAIN_SAMPLES // cfg.LOGICAL_BATCH_SIZE) == 975
    sigma = _sigma(cfg.FORMAL_ACCOUNTANT_STEPS, cfg.TRAIN_SAMPLES)
    accountant = RDPAccountant()
    for _ in range(cfg.FORMAL_ACCOUNTANT_STEPS):
        accountant.step(noise_multiplier=sigma, sample_rate=cfg.LOGICAL_BATCH_SIZE / cfg.TRAIN_SAMPLES)
    assert sigma > 0
    assert accountant.get_epsilon(cfg.DELTA) <= cfg.EPSILON


def test_accuracy_auc_and_single_multi_seed_analysis(tmp_path):
    assert accuracy_auc([1, 2, 3, 5], [0.1, 0.3, 0.2, 0.6]) == pytest.approx(1.25)
    assert accuracy_auc([1], [0.4]) == 0.0
    for method_index, method in enumerate(cfg.METHODS):
        run = tmp_path / "single" / "runs" / f"{method}_42"
        run.mkdir(parents=True)
        pd.DataFrame([{"method": method, "seed": 42, "epoch": 1, "train_loss": float(method_index)}]).to_csv(run / "metrics.csv", index=False)
    analyze(tmp_path / "single")
    single = pd.read_csv(tmp_path / "single" / "method_summary.csv")
    paired = pd.read_csv(tmp_path / "single" / "paired_summary.csv")
    assert single["train_loss_sample_std"].isna().all()
    assert paired["sample_std"].isna().all()
    assert paired["bootstrap_ci95_low"].isna().all()
    multi_root = tmp_path / "multi"
    for seed in (42, 7, 123):
        for method_index, method in enumerate(cfg.METHODS):
            run = multi_root / "runs" / f"{method}_{seed}"
            run.mkdir(parents=True)
            pd.DataFrame([{"method": method, "seed": seed, "epoch": 1, "train_loss": float(seed + method_index)}]).to_csv(run / "metrics.csv", index=False)
    analyze(multi_root)
    paired = pd.read_csv(multi_root / "paired_summary.csv")
    assert np.isfinite(paired["sample_std"]).all()
    assert np.isfinite(paired["bootstrap_ci95_low"]).all()
    assert np.isfinite(paired["bootstrap_ci95_high"]).all()

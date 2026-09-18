"""Offline checks using a real, small DistilBERT MLM with tied weights."""

import copy
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from transformers import BertTokenizer, DistilBertConfig, DistilBertForMaskedLM

from scripts.paper import exp_distilbert_sst2 as exp


@pytest.fixture
def tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "it", "was", ".", "terrible", "great", "a", "film"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab) + "\n")
    key = "vocab" if "vocab" in inspect.signature(BertTokenizer).parameters else "vocab_file"
    return BertTokenizer(**{key: str(path)}, do_lower_case=True)


def tiny_model(tokenizer, device=torch.device("cpu")):
    config = DistilBertConfig(vocab_size=len(tokenizer), max_position_embeddings=128,
                             n_layers=1, n_heads=2, dim=8, hidden_dim=12,
                             dropout=0.0, attention_dropout=0.0, pad_token_id=0)
    config._attn_implementation = "eager"
    return exp.PromptMLM(DistilBertForMaskedLM(config), exp.prompt_parts(tokenizer)[1]).to(device)


def test_prompt_and_mlm_objective(tokenizer):
    packed = exp.pack_prompts(["a " * 300, "a film", "[MASK] film"], tokenizer)
    suffix, verbalizer = exp.prompt_parts(tokenizer)
    assert packed.shape == (3, 3, 128)
    for item in packed:
        length = item[1].sum().item()
        ids = item[0, :length].tolist()
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-len(suffix) - 1:] == suffix + [tokenizer.sep_token_id]
        assert item[0, item[2, 0]] == tokenizer.mask_token_id
    assert packed[0, 1].sum() == 128
    assert packed[2, 2, 0] > 1  # choose the prompt's mask, not a sentence mask
    model = tiny_model(tokenizer).eval()
    actual = model(packed)
    full = model.mlm(input_ids=packed[:, 0], attention_mask=packed[:, 1]).logits
    expected = full[torch.arange(len(packed)), packed[:, 2, 0]][:, verbalizer]
    torch.testing.assert_close(actual, expected)
    labels = torch.tensor([0, 1, 0])
    assert F.cross_entropy(actual, labels, reduction="none").shape == (3,)
    assert all(p.requires_grad for p in model.parameters())
    assert model.mlm.vocab_projector.weight is model.mlm.distilbert.embeddings.word_embeddings.weight


@pytest.mark.parametrize("geometry", ["base", "full", "a_only"])
@pytest.mark.parametrize("device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_bk_explicit_and_autograd_agree(tokenizer, geometry, device_name, monkeypatch):
    # Keep sequence length distinct from parameter axes for allocation guards.
    monkeypatch.setattr(exp, "MAX_LENGTH", 11)
    monkeypatch.setattr(exp, "GEOMETRY_BATCH_SIZE", 5)
    monkeypatch.setattr(exp, "GEOMETRY_PHYSICAL_BATCH_SIZE", 2)
    torch.manual_seed(42)
    device = torch.device(device_name)
    model = tiny_model(tokenizer, device).train()
    explicit = copy.deepcopy(model)
    reference = copy.deepcopy(model)
    x = exp.pack_prompts(["a a film", "film", "great a"], tokenizer).to(device)
    y = torch.tensor([0, 1, 0], device=device)
    layers = exp.geometry_layers(model)
    assert "mlm.vocab_projector" not in layers
    covariances = []
    original_power = exp.matrix_power

    def record_covariance(covariance, power):
        covariances.append(covariance.clone())
        return original_power(covariance, power)

    monkeypatch.setattr(exp, "matrix_power", record_covariance)
    state = torch.random.get_rng_state()
    ua, ug = exp.build_geometry(model, layers, geometry, tokenizer, device, 42, 1)
    assert torch.equal(state, torch.random.get_rng_state())
    # Accumulate covariance sufficient statistics across geometry microbatches,
    # including the remainder; never average already-inverted operators.
    monkeypatch.setattr(exp, "GEOMETRY_PHYSICAL_BATCH_SIZE", 5)
    exp.build_geometry(model, layers, geometry, tokenizer, device, 42, 1)
    count = len(covariances) // 2
    for chunked, whole in zip(covariances[:count], covariances[count:]):
        torch.testing.assert_close(chunked, whole, rtol=2e-5, atol=1e-6)
    # Independent single-example autograd oracle, merging the shared parameter
    # through actual autograd before applying preconditioning and global clipping.
    sample_grads = {name: [] for name, _ in reference.named_parameters()}
    for i in range(len(x)):
        reference.zero_grad(set_to_none=True)
        F.cross_entropy(reference(x[i:i + 1]), y[i:i + 1]).backward()
        for name, module in exp.geometry_layers(reference).items():
            if name in ua:
                g = torch.cat((module.weight.grad, module.bias.grad[:, None]), -1)
                if name in ug:
                    g = ug[name] @ g
                g = g @ ua[name]
                module.weight.grad = g[:, :-1]
                module.bias.grad = g[:, -1]
        for name, parameter in reference.named_parameters():
            sample_grads[name].append(parameter.grad.clone())
    sample_grads = {name: torch.stack(values) for name, values in sample_grads.items()}
    expected_norms = sum(g.flatten(1).square().sum(1) for g in sample_grads.values()).sqrt()
    expected_factors = (exp.MAX_GRAD_NORM / (expected_norms + 1e-6)).clamp(max=1)
    expected = {name: torch.einsum("b,b...->...", expected_factors, g) for name, g in sample_grads.items()}

    stats = exp.new_stats(True)
    loss, records = exp.bk_differentiate(model, x, y, stats, device, True)
    assert all(p.grad is None and p.requires_grad and not hasattr(p, "grad_sample") for p in model.parameters())
    raw = [(r, r.z.clone() if r.z is not None else None, r.b.clone()) for r in records]
    forbidden = {(len(x), *r.module.weight.shape) for r in records if r.kind in ("linear", "embedding")}
    forbidden.update((len(x), r.module.out_features, r.module.in_features + 1)
                     for r in records if r.kind == "linear")

    class NoSampleMatrices(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            for tensor in tree_leaves(output):
                if isinstance(tensor, torch.Tensor) and tuple(tensor.shape) in forbidden:
                    # matmul may broadcast a single G operator as a view.
                    assert tensor.stride(0) == 0
            return output

    with NoSampleMatrices():
        norms, factors = exp.bk_clip(model, records, ua, ug, stats, device, True)
    assert not records
    for record, z, b in raw:
        if z is not None:
            assert torch.equal(record.z, z)
        assert torch.equal(record.b, b)
    torch.testing.assert_close(norms, expected_norms, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(factors, expected_factors, rtol=2e-4, atol=2e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=5e-4, atol=3e-5)
    assert stats["sample_gradient_matrices_preconditioned"] == 0
    assert stats["grad_sample_peak_bytes"] == 0

    wrapper = exp.GradSampleModule(explicit, loss_reduction="sum")
    try:
        explicit_loss = F.cross_entropy(wrapper(x), y, reduction="sum")
        explicit_loss.backward()
        explicit_stats = exp.new_stats(True)
        explicit_norms, _ = exp.explicit_clip(wrapper, exp.geometry_layers(explicit), ua, ug, explicit_stats, device, len(x), True)
        torch.testing.assert_close(loss, explicit_loss.detach())
        torch.testing.assert_close(explicit_norms, expected_norms, rtol=2e-4, atol=2e-5)
        for name, parameter in explicit.named_parameters():
            torch.testing.assert_close(parameter.grad, expected[name], rtol=5e-4, atol=3e-5)
        assert explicit_stats["grad_sample_peak_bytes"] > 0
        assert explicit_stats["aggregate_matrices_preconditioned"] == 0
    finally:
        wrapper.remove_hooks()


@pytest.mark.parametrize("engine,profile_mode,batch_constant,physical_steps", [
    ("bk", "practical", "BK_PHYSICAL_BATCH_SIZE", 6),
    ("bk", "controlled", "PROFILE_PHYSICAL_BATCH_SIZE", 12),
    ("explicit", "controlled", "PROFILE_PHYSICAL_BATCH_SIZE", 12),
])
@pytest.mark.parametrize("geometry,profile", [("base", True), ("base", False), ("full", True), ("a_only", True)])
def test_logical_batches_csv_and_final_evaluation(tokenizer, monkeypatch, tmp_path, engine, geometry, profile,
                                                  profile_mode, batch_constant, physical_steps):
    monkeypatch.setattr(exp, "MAX_LENGTH", 12)
    monkeypatch.setattr(exp, "LOGICAL_BATCH_SIZE", 8)
    monkeypatch.setattr(exp, "BK_PHYSICAL_BATCH_SIZE", 4)
    monkeypatch.setattr(exp, "EXPLICIT_PHYSICAL_BATCH_SIZE", 2)
    monkeypatch.setattr(exp, "PROFILE_PHYSICAL_BATCH_SIZE", 2)
    monkeypatch.setattr(exp, "GEOMETRY_BATCH_SIZE", 3)
    monkeypatch.setattr(exp, "GEOMETRY_PHYSICAL_BATCH_SIZE", 2)
    monkeypatch.setattr(exp, "make_model", tiny_model)
    data = TensorDataset(exp.pack_prompts(["a film"] * 11, tokenizer), torch.arange(11) % 2)
    calls = []
    original = exp.evaluate

    def evaluate(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(exp, "evaluate", evaluate)
    learning_rates = []
    original_adam = torch.optim.Adam

    def adam(*args, **kwargs):
        learning_rates.append(kwargs["lr"])
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", adam)
    result = exp.run_one(data, data, tokenizer, geometry, "none" if geometry == "base" else "synthetic", engine, 3., 42, 2,
                         torch.device("cpu"), tmp_path, getattr(exp, batch_constant), profile, profile_mode, lr=1e-4)
    assert learning_rates == [1e-4]
    assert result["lr"] == 1e-4
    assert result["device"] == "cpu"
    assert len(calls) == 1
    assert result["train_examples"] == 22
    assert result["logical_steps"] == result["optimizer_steps"] == result["noise_events"] == result["accountant_steps"] == 4
    assert result["physical_steps"] == physical_steps
    assert result["sample_rate"] == 8 / 11
    assert result["physical_batch_size"] == getattr(exp, batch_constant)
    assert result["profile_mode"] == profile_mode
    assert result["profiled"] == profile
    assert 0 < result["epsilon_spent"] <= 3.01
    rows = pd.read_csv(tmp_path / f"{geometry}_{result['source']}_{engine}_{profile_mode}_eps3_seed42.csv")
    assert (rows["physical_batch_size"] == getattr(exp, batch_constant)).all()
    assert (rows["profile_mode"] == profile_mode).all()
    assert len(rows) == 2
    assert pd.isna(rows.iloc[0]["accuracy"])
    assert 0 <= rows.iloc[-1]["accuracy"] <= 1
    if profile:
        assert all(result[key] >= 0 for key in exp.TIMINGS)
        assert result["algorithm_seconds"] == result["geometry_build_seconds"] + result["private_train_seconds"]
    else:
        assert all(pd.isna(result[key]) for key in (*exp.TIMINGS, *exp.COUNTERS, *exp.MEMORY))


def test_synthetic_prompt_payloads(tokenizer, monkeypatch):
    monkeypatch.setattr(exp, "MAX_LENGTH", 12)
    packed, labels = exp.synthetic_prompts(tokenizer, 42, 1)
    suffix, _ = exp.prompt_parts(tokenizer)
    allowed = set(range(len(tokenizer))) - set(tokenizer.all_special_ids)
    lengths = []
    for ids, attention, positions in packed:
        valid = int(attention.sum())
        payload_len = valid - len(suffix) - 2
        lengths.append(payload_len)
        assert 1 <= payload_len <= exp.MAX_LENGTH - len(suffix) - 2
        assert ids[0] == tokenizer.cls_token_id
        assert set(ids[1:1 + payload_len].tolist()) <= allowed
        assert ids[1 + payload_len:valid - 1].tolist() == suffix
        assert ids[valid - 1] == tokenizer.sep_token_id
        assert (ids[valid:] == tokenizer.pad_token_id).all()
        assert (attention[:valid] == 1).all()
        assert (attention[valid:] == 0).all()
        assert (positions == 1 + payload_len + suffix.index(tokenizer.mask_token_id)).all()
        assert ids[positions[0]] == tokenizer.mask_token_id
    assert len(set(lengths)) > 1
    assert set(labels.tolist()) == {0, 1}
    repeated, repeated_labels = exp.synthetic_prompts(tokenizer, 42, 1)
    assert torch.equal(packed, repeated)
    assert torch.equal(labels, repeated_labels)


@pytest.mark.parametrize("engine,epsilon,fast,expected", [
    ("auto", 8., False, [("bk", 128, False, "practical")]),
    ("auto", 3., False, [("bk", 128, True, "practical"),
                         ("bk", 16, True, "controlled"), ("explicit", 16, True, "controlled")]),
    ("bk", 3., False, [("bk", 128, True, "practical")]),
    ("explicit", 3., False, [("explicit", 16, True, "practical")]),
    ("all", 8., False, [("bk", 128, False, "practical"),
                        ("bk", 16, True, "controlled"), ("explicit", 16, True, "controlled")]),
    ("bk", 8., True, [("bk", 128, True, "practical"),
                      ("bk", 16, True, "controlled"), ("explicit", 16, True, "controlled")]),
])
def test_main_run_selection(monkeypatch, tmp_path, engine, epsilon, fast, expected):
    argv = ["experiment", "--engine", engine, "--epsilon", str(epsilon),
            "--seed", "7", "--output_dir", str(tmp_path), "--lr", "1e-4", "--gpu", "1"]
    if fast:
        argv.append("--fast")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    selected_devices = []
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    monkeypatch.setattr(exp.AutoTokenizer, "from_pretrained", lambda _: None)
    monkeypatch.setattr(exp, "load_data", lambda _: (None, None))
    calls = []

    def run_one(train, validation, tokenizer, geometry, source, engine, epsilon, seed,
                epochs, device, output_dir, physical_batch_size, profile, profile_mode, lr=exp.LR):
        assert lr == 1e-4
        assert device == torch.device("cuda:1")
        calls.append((geometry, source, engine, physical_batch_size, profile, profile_mode,
                      epsilon, seed, epochs))
        return {}

    monkeypatch.setattr(exp, "run_one", run_one)
    exp.main()
    assert selected_devices == [1]
    assert calls == [(geometry, source, *run, 3. if fast else epsilon,
                      42 if fast else 7, 1 if fast else exp.EPOCHS)
                     for geometry, source in exp.CONDITIONS for run in expected]


def test_evaluation_batch_size(tokenizer, monkeypatch):
    monkeypatch.setattr(exp, "MAX_LENGTH", 12)
    monkeypatch.setattr(exp, "EVAL_BATCH_SIZE", 2)
    data = TensorDataset(exp.pack_prompts(["a film"] * 5, tokenizer), torch.arange(5) % 2)
    model = tiny_model(tokenizer)
    sizes = []
    model.register_forward_pre_hook(lambda module, inputs: sizes.append(len(inputs[0])))
    exp.evaluate(model, data, torch.device("cpu"))
    assert sizes == [2, 2, 1]

"""Offline checks for the StackOverflow A-only BK experiment."""
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

SPEC = importlib.util.spec_from_file_location(
    "paper_stackoverflow_a",
    Path(__file__).resolve().parents[1] / "scripts/paper/exp_stackoverflow_a.py",
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1100, 4)
        self.requires_grad_(False)
        self.config = SimpleNamespace(vocab_size=1100)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(pooler_output=self.embedding(input_ids).mean(1))


class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = Backbone()
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.backbone(input_ids, attention_mask).pooler_output)


def loaders():
    private = DataLoader([
        dict(input_ids=torch.tensor([i + 1, 2, 3]),
             attention_mask=torch.ones(3, dtype=torch.long), labels=torch.tensor(i % 2))
        for i in range(8)
    ], batch_size=4)
    # Deliberately omit public labels: A-only geometry does not need them.
    public = DataLoader([
        dict(input_ids=torch.tensor([5, 6, 7]), attention_mask=torch.ones(3, dtype=torch.long))
        for _ in range(4)
    ], batch_size=4)
    return private, public


def test_bk_matches_explicit_sample_gradients():
    torch.manual_seed(42)
    model = Model().double()
    private, _ = loaders()
    batch = next(iter(private))
    ids, mask, labels = (batch[k] for k in ("input_ids", "attention_mask", "labels"))
    features = model.backbone(ids, mask).pooler_output
    operator = experiment.compute_a_operator(features)
    # Force clipping so the comparison exercises both norm and aggregation paths.
    operator = operator * 10
    summed, norms, factors = experiment.bk_aggregate(model, ids, mask, labels, operator)
    samples = []
    for i in range(len(labels)):
        loss = nn.functional.cross_entropy(model(ids[i:i+1], mask[i:i+1]), labels[i:i+1])
        weight, bias = torch.autograd.grad(loss, tuple(model.classifier.parameters()))
        samples.append(torch.cat([weight, bias[:, None]], dim=1) @ operator)
    samples = torch.stack(samples)
    expected_norms = samples.flatten(1).norm(dim=1)
    expected_factors = (experiment.MAX_GRAD_NORM / (expected_norms + 1e-6)).clamp(max=1)
    torch.testing.assert_close(norms, expected_norms)
    torch.testing.assert_close(factors, expected_factors)
    torch.testing.assert_close(summed, (samples * expected_factors[:, None, None]).sum(0))
    assert (factors < 1).any()
    assert all(p.grad is None and not hasattr(p, "grad_sample") for p in model.parameters())


@pytest.mark.parametrize("public_mode", [False, True])
def test_training_both_sources(public_mode):
    private, public = loaders()
    model = Model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = experiment.train_dp_kfc_a(
        model, private, private, public, epsilon=3., seed=42, epochs=2,
        use_public_data=public_mode, device=torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(result)).all()
    assert 0 <= result[0] <= 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_complete_sweep_csv(monkeypatch, tmp_path):
    private, public = loaders()
    monkeypatch.setattr(experiment, "get_text_loaders", lambda *a, **kw: (private, private, public, 8))
    monkeypatch.setattr(experiment, "BERTClassifier", Model)
    calls = []

    def train(*args, **kwargs):
        calls.append((kwargs["epsilon"], kwargs["seed"], kwargs["use_public_data"]))
        assert kwargs["epochs"] == 5
        return .75, .5, kwargs["epsilon"]

    monkeypatch.setattr(experiment, "train_dp_kfc_a", train)
    experiment.run_experiment(output_dir=tmp_path)
    expected = {(epsilon, seed, public) for epsilon in experiment.EPSILONS
                for seed in experiment.SEEDS for public in (False, True)}
    assert len(calls) == len(expected) == 140
    assert set(calls) == expected
    with (tmp_path / "stackoverflow_a_results.csv").open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["Method", "Epsilon", "Seed", "Accuracy", "Loss"]
        assert len(list(reader)) == 140

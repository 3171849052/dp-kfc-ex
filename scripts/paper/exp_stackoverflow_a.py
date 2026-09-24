"""StackOverflow duplicate detection: public/synthetic DP-KFC-A + BK sweep.

Uses frozen BERT and the original StackOverflow training settings. A-only
preconditioning uses (A + damping I)^(-0.4), before clipping and noise, as in
exp_imdb_logreg_a.py. The single Linear head uses the existing Exp25 BK
factor-norm/aggregate and noise-update primitives: no per-example weight gradients
are materialized. Calibration needs only activations, never public labels.
Accounting follows the original fixed-size shuffled/drop-last convention.
"""

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from rich.console import Console

from dp_kfac.data import get_text_loaders
from dp_kfac.models import BERTClassifier
from exp25.methods import reconstruct, update
from dp_kfac.results import save_results_csv
from dp_kfac.trainer import evaluate, generate_synthetic_text_batch, set_seed

EPSILONS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 64
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
MAX_LENGTH = 128
MAX_SAMPLES = 5000
A_POWER = 0.4
DAMPING = 1e-3
MODEL_NAME = "bert-base-uncased"
METHODS = [("DP-KFC-A+BK (Synthetic)", False), ("DP-KFC-A+BK (Public)", True)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLUMNS = ["Method", "Epsilon", "Seed", "Accuracy", "Loss"]
console = Console()


def compute_a_operator(activation: torch.Tensor) -> torch.Tensor:
    """Bias-augmented covariance and unnormalized fractional inverse."""
    augmented = torch.cat([activation, torch.ones_like(activation[:, :1])], dim=1)
    identity = torch.eye(augmented.size(1), device=activation.device,
                         dtype=activation.dtype)
    covariance = augmented.T @ augmented / augmented.size(0) + 1e-5 * identity
    values, vectors = torch.linalg.eigh(covariance + DAMPING * identity)
    return (vectors * values.clamp(min=1e-6).pow(-A_POWER)) @ vectors.T


def bk_aggregate(model, input_ids, attention_mask, target, operator):
    """Output-anchor backward, then exact factor norms and clipped aggregation."""
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        features = model.backbone(input_ids=input_ids,
                                  attention_mask=attention_mask).pooler_output
        logits = model.classifier(features)
        augmented = torch.cat([features, torch.ones_like(features[:, :1])], dim=1)
        transformed = augmented @ operator
    anchor = logits.requires_grad_(True)
    loss = nn.functional.cross_entropy(anchor, target, reduction="sum")
    backprop, = torch.autograd.grad(loss, anchor)
    # ||b_i (a_i U_A)^T||_F = ||b_i|| * ||a_i U_A||.
    # reconstruct forms only the batch-summed O x (I+1) gradient.
    return reconstruct(transformed, backprop.detach(), clip=MAX_GRAD_NORM)


def train_dp_kfc_a(base_model, train_loader, test_loader, public_loader,
                   *, epsilon, seed, epochs, use_public_data, device=DEVICE):
    set_seed(seed)
    model = copy.deepcopy(base_model).to(device)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if trainable != {"classifier.weight", "classifier.bias"}:
        raise ValueError("BK requires a frozen backbone and a single trainable Linear head")
    # Only the classifier is trainable; the pretrained backbone stays frozen.
    classifier = model.classifier
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    sample_rate = train_loader.batch_size / len(train_loader.dataset)
    if not len(train_loader):
        raise ValueError("Training data must contain at least one full batch")
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon, target_delta=DELTA, sample_rate=sample_rate,
        steps=epochs * len(train_loader), accountant="rdp",
    )
    accountant = RDPAccountant()
    noise_rng = torch.Generator(device=device).manual_seed(seed)
    public_iter = iter(public_loader) if use_public_data else None

    for epoch in range(epochs):
        model.train()
        # Refresh A once per epoch, matching the existing paper experiments.
        if use_public_data:
            try:
                calibration = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                calibration = next(public_iter)
            input_ids = calibration["input_ids"].to(device)
            mask = calibration["attention_mask"].to(device)
        else:
            input_ids, mask, _ = generate_synthetic_text_batch(
                batch_size=train_loader.batch_size, max_len=MAX_LENGTH,
                vocab_size=model.backbone.config.vocab_size,
                num_classes=classifier.out_features, device=device,
            )
        activations = []
        handle = classifier.register_forward_pre_hook(
            lambda module, inputs: activations.append(inputs[0].detach())
        )
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=mask)
        finally:
            handle.remove()
        operator = compute_a_operator(activations[0])

        for batch in train_loader:
            target = batch["labels"].to(device)
            summed, _, _ = bk_aggregate(
                model, batch["input_ids"].to(device),
                batch["attention_mask"].to(device), target, operator,
            )
            update(model, optimizer, summed, noise_multiplier, target.size(0),
                   noise_rng, accountant, clip=MAX_GRAD_NORM, sample_rate=sample_rate)
        console.print(f"    Epoch {epoch + 1}/{epochs}")

    accuracy, loss = evaluate(model, test_loader, device, is_text=True)
    return accuracy, loss, accountant.get_epsilon(delta=DELTA)


def run_experiment(fast=False, output_dir="results",
                   override_seed=None, override_epsilon=None, dry_run=False):
    epsilons = EPSILONS[:2] if fast else EPSILONS
    seeds = SEEDS[:2] if fast else SEEDS
    epochs = 1 if fast else EPOCHS
    if override_epsilon is not None:
        if override_epsilon <= 0:
            raise ValueError("epsilon must be positive")
        epsilons = [override_epsilon]
    if override_seed is not None:
        seeds = [override_seed]
    output_path = Path(output_dir) / "stackoverflow_a_results.csv"
    console.print(f"DP-KFC-A+BK StackOverflow | device={DEVICE} | epochs={epochs}")
    console.print(f"Epsilons: {epsilons}\nSeeds: {seeds}")
    console.print(f"Runs: {len(epsilons) * len(seeds) * len(METHODS)} | {output_path}")
    if dry_run:
        for epsilon in epsilons:
            for seed in seeds:
                for method, _ in METHODS:
                    console.print(f"  {method} | epsilon={epsilon} | seed={seed}")
        return

    set_seed(SEEDS[0])
    train_loader, test_loader, public_loader, train_size = get_text_loaders(
        "stackoverflow", "agnews", BATCH_SIZE, max_length=MAX_LENGTH,
        max_samples=MAX_SAMPLES, tokenizer_name=MODEL_NAME,
    )
    console.print(f"Train size: {train_size}")
    base_model = BERTClassifier(num_classes=2, model_name=MODEL_NAME,
                                freeze_backbone=True)
    results = []
    for epsilon in epsilons:
        for seed in seeds:
            for method, use_public_data in METHODS:
                console.print(f"{method} | epsilon={epsilon} | seed={seed}")
                accuracy, loss, spent = train_dp_kfc_a(
                    base_model, train_loader, test_loader, public_loader,
                    epsilon=epsilon, seed=seed, epochs=epochs,
                    use_public_data=use_public_data,
                )
                results.append(dict(Method=method, Epsilon=epsilon, Seed=seed,
                                    Accuracy=accuracy, Loss=loss))
                save_results_csv(results, output_path, columns=COLUMNS)
                console.print(f"  Accuracy={accuracy:.4f} Loss={loss:.4f} ε={spent:.4f}")
    console.print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="Run the first two epsilons/seeds for one epoch")
    parser.add_argument("--seed", type=int, help="Run only this seed")
    parser.add_argument("--epsilon", type=float, help="Run only this epsilon")
    parser.add_argument("--output_dir", "--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sweep without loading data or models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(fast=args.fast, output_dir=args.output_dir,
                   override_seed=args.seed, override_epsilon=args.epsilon,
                   dry_run=args.dry_run)

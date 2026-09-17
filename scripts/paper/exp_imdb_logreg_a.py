"""IMDB logistic regression with public or synthetic DP-KFC-A."""

import argparse
import copy
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from rich.console import Console

from dp_kfac.data import get_imdb_data, get_agnews_data, get_tfidf_features
from dp_kfac.models import LogisticRegression
from dp_kfac.trainer import evaluate, generate_white_noise, set_seed
from dp_kfac.privacy import clip_and_noise_gradients
from dp_kfac.results import save_results_csv
from exp_imdb_logreg import (
    EPSILONS, SEEDS, EPOCHS, LR, BATCH_SIZE, MAX_GRAD_NORM_KFAC,
    MAX_FEATURES, NUM_CLASSES,
)

A_POWER = 0.4
DAMPING = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console = Console()


def compute_a_covariance(activation: torch.Tensor) -> torch.Tensor:
    """Dense, bias-augmented Linear covariance used by IMDB DP-KFAC."""
    A = activation
    A_bias = torch.cat([A, torch.ones_like(A[:, :1])], dim=1)
    return A_bias.T @ A_bias / A.size(0) + 1e-5 * torch.eye(
        A_bias.size(1), device=A.device, dtype=A.dtype
    )


def compute_a_operator(
    A: torch.Tensor,
    power: float = A_POWER,
    damping: float = DAMPING,
) -> torch.Tensor:
    """Compute the dense (A + damping I)^(-power), without rescaling."""
    eigvals, eigvecs = torch.linalg.eigh(
        A + damping * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    )
    return eigvecs @ torch.diag(eigvals.clamp(min=1e-6).pow(-power)) @ eigvecs.T


def precondition_per_sample_gradients_a(
    model: GradSampleModule,
    operators: dict[str, torch.Tensor],
) -> None:
    """Right-multiply each Linear layer's bias-augmented sample gradients."""
    for name, module in model._module.named_modules():
        if isinstance(module, nn.Linear):
            g_sample_w = module.weight.grad_sample
            g_sample_b = module.bias.grad_sample
            g_aug = torch.cat([g_sample_w, g_sample_b.unsqueeze(2)], dim=2)
            preconditioned = torch.einsum("bok,ki->boi", g_aug, operators[name])
            module.weight.grad_sample = preconditioned[:, :, :-1]
            module.bias.grad_sample = preconditioned[:, :, -1]


def train_dp_kfc_a(
    model: nn.Module,
    train_loader: DataLoader,
    public_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    seed: int,
    use_public_data: bool,
) -> tuple:
    set_seed(seed)
    # Match Trainer._fresh_model(): every run starts from the same base model.
    model = copy.deepcopy(model).to(device)
    model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9
    )
    train_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size or BATCH_SIZE
    sample_rate = batch_size / train_size
    total_steps = epochs * (train_size // batch_size)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp",
    )
    accountant = RDPAccountant()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    public_iter = iter(itertools.cycle(public_loader))
    operators = {}

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad(set_to_none=True)

            if batch_idx % len(train_loader) == 0:
                if use_public_data:
                    calibration_data = next(public_iter)[0].to(device)
                else:
                    calibration_data = generate_white_noise(
                        data.size(0), (data.size(1),), device
                    )

                activations = {}

                def capture_activation(name):
                    def hook(module, inputs, output):
                        activations[name] = inputs[0].detach()
                    return hook

                handles = [
                    module.register_forward_hook(capture_activation(name))
                    for name, module in model._module.named_modules()
                    if isinstance(module, nn.Linear)
                ]
                # Opacus does not record calibration activations under no_grad.
                with torch.no_grad():
                    model(calibration_data)
                for handle in handles:
                    handle.remove()
                operators = {
                    name: compute_a_operator(compute_a_covariance(activation))
                    for name, activation in activations.items()
                }

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            precondition_per_sample_gradients_a(model, operators)
            clip_and_noise_gradients(
                model, noise_multiplier, max_grad_norm, data.size(0)
            )
            optimizer.step()
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        console.print(f"      Epoch {epoch + 1}/{epochs}")

    return evaluate(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser(
        description="IMDB logistic regression: DP-KFC-A"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke-test with fewer epsilons and seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run with a single specific seed (e.g. --seed 42)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Run with a single specific epsilon (e.g. --epsilon 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save result CSV (default: results)",
    )
    args = parser.parse_args()

    epsilons = EPSILONS if not args.fast else [1.0, 4.0]
    seeds = SEEDS if not args.fast else [42, 43]
    epochs = EPOCHS if not args.fast else 2

    if args.epsilon is not None:
        epsilons = [args.epsilon]
    if args.seed is not None:
        seeds = [args.seed]

    console.print("[bold cyan]Loading IMDB data...[/bold cyan]")
    train_texts, train_labels, test_texts, test_labels = get_imdb_data()

    console.print("[bold cyan]Loading AG News (public proxy)...[/bold cyan]")
    public_texts, _ = get_agnews_data()

    console.print("[bold cyan]Extracting TF-IDF features...[/bold cyan]")
    train_feats, test_feats, public_feats = get_tfidf_features(
        train_texts, test_texts, public_texts=public_texts, max_features=MAX_FEATURES
    )

    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    input_dim = train_feats.shape[1]

    train_ds = TensorDataset(train_feats, train_labels_t)
    test_ds = TensorDataset(test_feats, test_labels_t)
    public_ds = TensorDataset(public_feats)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    public_loader = DataLoader(
        public_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    train_size = len(train_ds)
    delta = 1.0 / train_size

    console.print(
        f"[green]Train: {train_size}  Test: {len(test_ds)}  "
        f"Public: {len(public_ds)}  delta={delta:.2e}[/green]"
    )

    results = []
    base_model = LogisticRegression(input_dim=input_dim, num_classes=NUM_CLASSES)

    for eps in epsilons:
        console.print(f"\n[bold yellow]===  Epsilon = {eps}  ===[/bold yellow]")
        for seed in seeds:
            console.print(f"[dim]  Seed {seed}[/dim]")
            for method, use_public_data in [
                ("DP-KFC-A (Public)", True),
                ("DP-KFC-A (Synthetic)", False),
            ]:
                console.print(f"    [cyan]{method}[/cyan]")
                acc, loss = train_dp_kfc_a(
                    model=base_model,
                    train_loader=train_loader,
                    public_loader=public_loader,
                    test_loader=test_loader,
                    device=DEVICE,
                    epochs=epochs,
                    epsilon=eps,
                    delta=delta,
                    max_grad_norm=MAX_GRAD_NORM_KFAC,
                    seed=seed,
                    use_public_data=use_public_data,
                )
                results.append({
                    "Method": method,
                    "Epsilon": eps,
                    "Seed": seed,
                    "Accuracy": acc,
                    "Loss": loss,
                })
                console.print(f"      acc={acc:.4f}  loss={loss:.4f}")

    output_dir = Path(args.output_dir)
    output_path = output_dir / "imdb_logreg_a_results.csv"
    save_results_csv(
        results,
        output_path,
        columns=["Method", "Epsilon", "Seed", "Accuracy", "Loss"],
    )
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  {'Method':<20} {'Epsilon':>8} {'Seed':>6} {'Accuracy':>10} {'Loss':>10}")
    console.print("  " + "-" * 60)
    for r in results:
        console.print(
            f"  {r['Method']:<20} {r['Epsilon']:>8.1f} {r['Seed']:>6} "
            f"{r['Accuracy']:>10.4f} {r['Loss']:>10.4f}"
        )


if __name__ == "__main__":
    main()

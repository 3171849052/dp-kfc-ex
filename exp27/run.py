"""Run all 50 paired grid entries using the reference training implementation."""
from inspect import signature

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from exp27 import config as cfg
from scripts.paper import exp_cnn_mnist_a as reference


def load_data():
    # Same normalization and loader settings as reference.get_mnist_loaders;
    # download=False makes the existing repository dataset strictly read-only.
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(cfg.ROOT / "data", train=True, download=False, transform=transform)
    test = datasets.MNIST(cfg.ROOT / "data", train=False, download=False, transform=transform)
    return (
        DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4,
                   pin_memory=True, drop_last=True),
        DataLoader(test, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4,
                   pin_memory=True),
    )


def run_grid(train, test, device):
    assert (reference.EPOCHS, reference.BATCH_SIZE, reference.DELTA,
            reference.DAMPING, reference.A_POWER) == (
                cfg.EPOCHS, cfg.BATCH_SIZE, cfg.DELTA, cfg.DAMPING, cfg.A_POWER)
    assert signature(reference.generate_pink_noise).parameters["alpha"].default == cfg.PINK_ALPHA
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    # A new invocation always starts the entire grid from scratch.
    pd.DataFrame(columns=["method", "C", "learning_rate"]).to_csv(cfg.RESULTS / "summary.csv", index=False)
    for c, lr, slug, method, geometry in cfg.grid():
        output_dir = cfg.RESULTS / "runs" / f"{slug}_C{c:g}_LR{lr:g}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{len(summaries) + 1}/50] {method}-Pink C={c:g} lr={lr:g}", flush=True)
        original_lr, original_c = reference.LR, reference.MAX_GRAD_NORM
        try:
            reference.LR, reference.MAX_GRAD_NORM = lr, c
            row = reference.run_one(
                train=train, test=test, public={}, geometry=geometry, source=cfg.SOURCE,
                engine=cfg.ENGINE, epsilon=cfg.EPSILON, seed=cfg.SEED,
                epochs=cfg.EPOCHS, device=device, output_dir=output_dir, profile=False,
            )
        finally:
            reference.LR, reference.MAX_GRAD_NORM = original_lr, original_c
        assert row["method"] == method and row["epoch"] == cfg.EPOCHS
        assert all(row[key] == cfg.EPOCHS * len(train) for key in cfg.STEP_FIELDS)
        row.update(C=c, learning_rate=lr, noise_std=row["noise_multiplier"] * c,
                   dataset="MNIST", model="SimpleCNN", optimizer="Adam",
                   batch_size=cfg.BATCH_SIZE, damping=cfg.DAMPING, a_power=cfg.A_POWER,
                   pink_alpha=cfg.PINK_ALPHA, geometry_rebuild_every_epochs=1)
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(cfg.RESULTS / "summary.csv", index=False)


def main():
    assert torch.cuda.is_available(), "Exp27 requires CUDA; no CPU fallback"
    train, test = load_data()
    assert len(train.dataset) == 60000 and len(test.dataset) == 10000
    run_grid(train, test, torch.device("cuda"))


if __name__ == "__main__":
    main()

"""Exp10 fixed paired comparison of unbounded Equil parameterizations."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import math
import numpy as np
from exp10 import run_exp10 as base
from exp10b import config as cfg
from exp10b.operators import build
from exp10b.reporting import report

torch, F, pd = base.torch, base.F, base.pd
OUTPUT = ROOT / "exp10b/results"


def run(method, seed, train, test, sigma, device, epochs):
    torch.manual_seed(seed)
    model = base.GradSampleModule(base.SimpleCNN().to(device), loss_reduction="sum")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)
    loader = base.DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
    test_loader = base.DataLoader(test, batch_size=cfg.BATCH_SIZE,
                                 generator=torch.Generator().manual_seed(seed))
    accountant = base.RDPAccountant()
    rows, records = [], []
    operator = None
    for epoch in range(1, epochs+1):
        model.train()
        model.zero_grad(set_to_none=True)
        operator = None
        start = base.timestamp(device)
        operator, gains, diagnostics = build(model, method, seed, epoch, device)
        build_seconds = base.timestamp(device)-start
        records.extend(diagnostics)
        start = base.timestamp(device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            loss.backward()
            if operator is not None:
                operator.apply(model)
            norms_sq = base._compute_per_sample_norms_squared(params, len(x), device)
            factors = base._compute_clip_factors(norms_sq, cfg.MAX_GRAD_NORM)
            clipped += (norms_sq.sqrt() > cfg.MAX_GRAD_NORM).sum().item()
            factor_sum += factors.sum().item()
            base.clip_and_noise_gradients(model, sigma, cfg.MAX_GRAD_NORM, len(x))
            optimizer.step()
            if sigma > 0:
                accountant.step(noise_multiplier=sigma, sample_rate=cfg.BATCH_SIZE/len(train))
            loss_sum += loss.item()
            count += len(x)
        train_seconds = base.timestamp(device)-start
        test_loss, accuracy = base.evaluate(model, test_loader, device)
        assert all(torch.isfinite(p).all() for p in params), "Nonfinite model parameters"
        assert math.isfinite(loss_sum) and math.isfinite(test_loss), "Nonfinite loss"
        rows.append(dict(method=method, seed=seed, epoch=epoch, tau=cfg.TAU,
            preconditioner_batches=cfg.PRECONDITIONER_BATCHES,
            synthetic_samples=cfg.SYNTHETIC_BATCH_SIZE*cfg.PRECONDITIONER_BATCHES,
            noise_multiplier=sigma, test_accuracy=accuracy, test_loss=test_loss,
            train_loss=loss_sum/count,
            epsilon_spent=accountant.get_epsilon(delta=cfg.DELTA) if sigma > 0 else float("inf"),
            clip_fraction=clipped/count, mean_clip_factor=factor_sum/count,
            precond_build_seconds=build_seconds, private_train_seconds=train_seconds,
            total_epoch_seconds=build_seconds+train_seconds, **gains))
        print(f"{method} seed={seed} epoch={epoch} accuracy={accuracy:.4f} "
              f"build={build_seconds:.2f}s train={train_seconds:.2f}s", flush=True)
    model.remove_hooks()
    return rows, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="512 samples, seed 42, one epoch, sigma=0.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    output = OUTPUT / "smoke" if args.smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    transform = base.transforms.Compose([base.transforms.ToTensor(),
                                          base.transforms.Normalize((.1307,), (.3081,))])
    train = base.datasets.MNIST(ROOT/"exp1/data", train=True, download=False, transform=transform)
    test = base.datasets.MNIST(ROOT/"exp1/data", train=False, download=False, transform=transform)
    epochs, seeds = cfg.EPOCHS, cfg.SEEDS
    if args.smoke:
        train, test = base.Subset(train, range(512)), base.Subset(test, range(512))
        epochs, seeds = 1, (42,)
    sigma = 0. if args.smoke else base.get_noise_multiplier(
        target_epsilon=cfg.EPSILON, target_delta=cfg.DELTA,
        sample_rate=cfg.BATCH_SIZE/len(train),
        steps=epochs*math.ceil(len(train)/cfg.BATCH_SIZE), accountant="rdp")
    resolved = {k.lower(): v for k, v in vars(cfg).items() if k.isupper()}
    resolved.update(epochs=epochs, seeds=seeds, noise_multiplier=sigma,
                    train_samples=len(train), test_samples=len(test),
                    model="SimpleCNN", optimizer="SGD")
    (output/"config.json").write_text(json.dumps(resolved, indent=2)+"\n")
    print(f"adamex python={sys.executable}; torch={torch.__version__}; "
          f"GPU={torch.cuda.get_device_name(device)}; sigma={sigma}; epochs={epochs}", flush=True)
    rows, records = [], []
    for seed in seeds:
        for method in cfg.METHODS:
            new_rows, new_records = run(method, seed, train, test, sigma, device, epochs)
            rows.extend(new_rows)
            records.extend(new_records)
            frame, final = base.save_results(rows, output)
            pd.DataFrame(records).to_csv(output/"operator_approximation.csv", index=False)
    report(frame, final, pd.DataFrame(records), output, epochs, seeds, smoke=args.smoke)
    print((output/"verification.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()

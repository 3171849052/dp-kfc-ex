"""Exp14b training with a scoped, diagnostic-only builder observer."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import argparse
import json
from unittest.mock import patch
import torch
from exp14b import run_exp14b as training
from exp14b.builders import build_from_cache
from exp14d import config as cfg
from exp14d.whitening import synthetic_probe, diagnose
from exp14d.analyze import save

OUTPUT = Path(__file__).parent / 'results'


def run(beta, seed, train, test, sigma, epochs, device, smoke=False):
    whitening, profiles = [], []

    def observed_builder(model, beta, cache, seed, epoch):
        operator, stats, factors = build_from_cache(model, beta, cache, seed, epoch)
        x, y = synthetic_probe(seed, epoch, device)
        aggregate, layers, profile = diagnose(model, factors, operator, x, y)
        identity = dict(beta=beta, seed=seed, epoch=epoch)
        whitening.extend(dict(**identity, **row) for row in layers)
        profiles.extend(dict(**identity, direction_percentile=i/10,
                             normalized_energy=value) for i, value in enumerate(profile))
        stats.update(aggregate)
        return operator, stats, factors

    # The original training loop looks up this module global at each epoch.
    # Scope restoration is automatic; no source files or DP functions are changed.
    with patch.object(training, 'build_from_cache', observed_builder):
        rows, _ = training.run(beta, seed, train, test, sigma, epochs, device, smoke)
    return rows, whitening, profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    transform = training.transforms.Compose([training.transforms.ToTensor(),
        training.transforms.Normalize((.1307,), (.3081,))])
    train = training.datasets.MNIST(ROOT/'exp1/data', train=True, download=False, transform=transform)
    test = training.datasets.MNIST(ROOT/'exp1/data', train=False, download=False, transform=transform)
    epochs, seeds, betas = cfg.EPOCHS, cfg.SEEDS, cfg.BETAS
    if args.smoke:
        train = training.Subset(train, range(cfg.BATCH_SIZE))
        test = training.Subset(test, range(cfg.BATCH_SIZE))
        epochs, seeds, betas = 1, (42,), cfg.SMOKE_BETAS
    sigma = 0. if args.smoke else training.get_noise_multiplier(target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA, sample_rate=cfg.BATCH_SIZE/len(train),
        steps=epochs*(len(train)//cfg.BATCH_SIZE), accountant='rdp')
    output = OUTPUT / ('smoke' if args.smoke else 'formal')
    rows, whitening, profiles = [], [], []
    with training.runtime(device):
        training.disposable_warmup(train, device)
        training.save_config(output, sigma, epochs, seeds, len(train), args.smoke)
        config_path = output/'config.json'
        metadata = json.loads(config_path.read_text())
        metadata.update(betas=betas, probe_samples=cfg.PROBE_SAMPLES,
            probe_batch_size=cfg.PROBE_BATCH_SIZE, probe_input_seed='seed + 50000 + epoch',
            probe_label_seed='seed + 60000 + epoch',
            whitening_timing='before private epoch updates; accuracy after updates',
            preconditioner_build_seconds='includes held-out whitening diagnostic',
            profile='1001 quantiles of concatenated within-layer normalized energy')
        config_path.write_text(json.dumps(metadata, indent=2)+'\n')
        for seed in seeds:
            for beta in betas:
                r, w, p = run(beta, seed, train, test, sigma, epochs, device, args.smoke)
                rows.extend(r)
                whitening.extend(w)
                profiles.extend(p)
                save(rows, whitening, profiles, output)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

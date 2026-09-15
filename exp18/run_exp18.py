"""Scoped adapters reuse Exp14b's complete private training loop unchanged."""
from pathlib import Path
import os
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(HERE / '.cache')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from opacus.accountants.utils import get_noise_multiplier
from exp12.runtime import runtime
from exp14b import run_exp14b as training
from exp18 import config as cfg
from exp18.builders import build, synthetic_cache
from exp18.analyze import save


class Adapter:
    def __init__(self, method):
        self.method = method
        self.operator = None
        self.diagnostics = []
        self.rebuilds = 0

    def cache(self, seed, epoch, device, batches, batch_size):
        return synthetic_cache(seed, epoch, device, batches, batch_size) if cfg.rebuild(self.method, epoch-1) else []

    def builder(self, model, method, cache, seed, epoch):
        if cfg.rebuild(method, epoch-1):
            self.operator = None
            self.operator, self.stats, diag = build(model, method, cache, seed, epoch)
            self.rebuilds += 1
            self.diagnostics.extend(dict(method=method, seed=seed, epoch=epoch-1, **d) for d in diag)
            stats = dict(self.stats)
        else:
            stats = {k: 0 if k.startswith('builder_') else v for k, v in self.stats.items()}
        return self.operator, dict(stats, rebuild_count=self.rebuilds,
                                   rebuilt=int(cfg.rebuild(method, epoch-1))), {}


def run(method, train, test, sigma, epochs, device, smoke):
    adapter = Adapter(method)
    settings = SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    if smoke:
        settings.BATCH_SIZE = 4
        settings.SYNTHETIC_BATCH_SIZE = 4
    with patch.multiple(training, cfg=settings, synthetic_cache=adapter.cache,
                        build_from_cache=adapter.builder, diagnose=lambda *a: [],
                        synthetic_state_metrics=lambda *a: {}):
        rows, _ = training.run(method, 42, train, test, sigma, epochs, device, smoke)
    for row in rows:
        row['method'] = row.pop('beta')
        row['beta'] = cfg.BETA
        row['epoch'] -= 1
        row['state_metric_forward_calls'] = 0
    return rows, adapter.diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train = datasets.MNIST(ROOT / 'exp1/data', train=True, download=False, transform=transform)
    test = datasets.MNIST(ROOT / 'exp1/data', train=False, download=False, transform=transform)
    epochs = cfg.EPOCHS
    if args.smoke:
        train, test = Subset(train, range(4)), Subset(test, range(4))
        epochs = 1
    steps = epochs*(len(train)//cfg.BATCH_SIZE)
    assert args.smoke or steps == 1170
    sigma = 0. if args.smoke else get_noise_multiplier(target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA, sample_rate=cfg.BATCH_SIZE/len(train), steps=steps, accountant='rdp')
    output = HERE / 'results' / 'smoke' if args.smoke else HERE / 'results'
    output.mkdir(parents=True, exist_ok=True)
    config = {k.lower(): getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    config.update(smoke=args.smoke, epochs=epochs, noise_multiplier=sigma,
        accountant_steps=0 if args.smoke else steps, batch_size=4 if args.smoke else cfg.BATCH_SIZE,
        synthetic_batches=1 if args.smoke else cfg.SYNTHETIC_BATCHES,
        synthetic_batch_size=4 if args.smoke else cfg.SYNTHETIC_BATCH_SIZE,
        dataset='MNIST', model='dp_kfac.models.SimpleCNN', shuffle=True, drop_last=True,
        optimizer='SGD', accounting='Exp14b shuffled fixed-batch RDP convention',
        synthetic_seed='42 + 10000 + (zero_based_epoch + 1)',
        label_seed='42 + 20000 + (zero_based_epoch + 1)', sketch_seed='42 + 30000 + (zero_based_epoch + 1)',
        noise_seed=40042, shuffle_seed=42, scale_reference_beta=.5,
        scale_matching='sqrt(m_ref/m_raw), using only each compressed structure; identity side moment = dimension',
        single_side_scale_caveat='Self-contained lightweight operators: identity moment is dimension for both powers; '
            'their reference RMS differs from full KFC. A-only versus C-only includes scale-weighting effects, '
            'not just factor geometry. No private-norm calibration or full-factor oracle.',
        builder_unique_samples='Samples in the synthetic cache; zero on reuse epochs',
        builder_processed_samples='Sample instances processed: unique for one pass, 2*unique for rank replay; zero on reuse',
        algorithm_time='build + private training; evaluation recorded separately',
        state_bytes='action tensor payload + float64 global scale/tail powers; excludes Python metadata',
        refresh2_epochs=[0,2,4], frozen_epochs=[0], deterministic=True, tf32=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    rows, diagnostics = [], []
    with runtime(device):
        # Small disposable build and private step warm all ten action paths.
        tiny_train, tiny_test = Subset(train, range(4)), Subset(test, range(4))
        if not args.smoke:
            with torch.random.fork_rng():
                for method in cfg.METHODS:
                    run(method, tiny_train, tiny_test, 0., 1, device, True)
        for method in cfg.METHODS:
            run_rows, diag = run(method, train, test, sigma, epochs, device, args.smoke)
            rows.extend(run_rows)
            diagnostics.extend(diag)
            save(rows, diagnostics, output, args.smoke)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

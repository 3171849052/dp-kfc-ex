from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import json
import random
import numpy as np
import pandas as pd
import torch
from dp_kfac.models import SimpleCNN
from exp12.config import parse
from exp12.probes import synthetic_cache
from exp12.curvature import estimate
from exp12 import oracle
from exp12.metrics import compare, relative
from exp12.benchmark import measure


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model = SimpleCNN().to(args.device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    cache = synthetic_cache(args.synthetic_batches, args.batch_size, args.device, args.seed+1)
    private = oracle.private_cache(args.private_samples, args.batch_size, args.device, args.seed+2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    budgets, raw = [], []
    def benchmark(name, k, seed, fn, private_input=False):
        factors, stats, rows = measure(fn, args.device, args.warmup, args.repeats)
        identity = dict(estimator=name, k=k, label_seed=seed)
        stats['synthetic_samples'] = 0 if private_input else stats['samples']
        budgets.append(dict(**identity, **stats))
        raw.extend(dict(**identity, **r) for r in rows)
        print(f'{name} seed={seed}: median={stats["build_median_seconds"]:.4f}s, '
              f'range=[{stats["min_build_seconds"]:.4f}, {stats["max_build_seconds"]:.4f}]s, '
              f'increment={stats["peak_allocated_increment"]/2**20:.2f} MiB', flush=True)
        return factors
    ref = benchmark('Private-KFLR', 9, -1, lambda: oracle.build(model, private), True)
    exact = benchmark('KFLR', 9, -1, lambda: estimate(model, cache, 'KFLR'))
    errors, whites, mc = [], [], []
    def record(name, seed, factors):
        e, w = compare(factors, ref, args.damping)
        for rows, target in [(e, errors), (w, whites)]:
            target.extend(dict(estimator=name, label_seed=seed, **r) for r in rows)
        for n in exact:
            torch.testing.assert_close(factors[n]['A'], exact[n]['A'])
    record('KFLR', -1, exact)
    jobs = [('KFRA-block', 0, -1)] + [(m, k, s) for m in ['KFAC-U', 'KFAC-M'] for k in [1, 3, 9] for s in args.label_seeds]
    random.Random(args.seed).shuffle(jobs)
    for method, k, seed in jobs:
        name = method if method == 'KFRA-block' else f'{method}-{k}'
        factors = benchmark(name, k, seed, lambda: estimate(model, cache, method, k, seed))
        record(name, seed, factors)
        if method != 'KFRA-block':
            mc.extend(dict(estimator=name, k=k, label_seed=seed, layer=n,
                           C_relative_error=relative(factors[n]['C'], exact[n]['C'])) for n in exact)
        del factors
    tables = {'curvature_error': pd.DataFrame(errors), 'whitening': pd.DataFrame(whites),
              'mc_convergence': pd.DataFrame(mc), 'compute_budget': pd.DataFrame(budgets),
              'benchmark_raw': pd.DataFrame(raw)}
    per_layer = tables['curvature_error'].drop(columns='label_seed').groupby(['estimator', 'layer']).mean(numeric_only=True).reset_index()
    per_layer['scope'] = 'layer_seed_mean'
    global_mean = per_layer.drop(columns=['layer', 'scope']).groupby('estimator').mean(numeric_only=True).reset_index()
    global_mean['layer'], global_mean['scope'] = 'ALL', 'global_mean'
    tables['summary'] = pd.concat([per_layer, global_mean], ignore_index=True)
    for name, table in tables.items():
        assert np.isfinite(table.select_dtypes(include='number').to_numpy()).all(), name
        table.to_csv(args.output/f'{name}.csv', index=False)
    print('MC C error vs synthetic KFLR (layer/seed mean; smoke is not conclusive):')
    print(tables['mc_convergence'].groupby('estimator')['C_relative_error'].mean().to_string())


if __name__ == '__main__':
    main(sys.argv[1:])

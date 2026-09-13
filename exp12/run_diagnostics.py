from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import json
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


def main():
    args = parse()
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
    ref, budget = measure(lambda: oracle.build(model, private), args.device)
    budgets = [dict(estimator='Private-KFLR', k=9, label_seed=-1, **budget)]
    exact, budget = measure(lambda: estimate(model, cache, 'KFLR'), args.device)
    budgets.append(dict(estimator='KFLR', k=9, label_seed=-1, **budget))
    errors, whites, mc = [], [], []
    def record(name, seed, factors):
        e, w = compare(factors, ref, args.damping)
        for rows, target in [(e, errors), (w, whites)]:
            target.extend(dict(estimator=name, label_seed=seed, **r) for r in rows)
        for n in exact:
            torch.testing.assert_close(factors[n]['A'], exact[n]['A'])
    record('KFLR', -1, exact)
    for method, k, seed in [('KFRA', 0, -1)] + [(m, k, s) for m in ['KFAC-U', 'KFAC-M'] for k in [1, 3, 9] for s in args.label_seeds]:
        factors, budget = measure(lambda: estimate(model, cache, method, k, seed), args.device)
        name = method if method == 'KFRA' else f'{method}-{k}'
        budgets.append(dict(estimator=name, k=k, label_seed=seed, **budget))
        record(name, seed, factors)
        if method != 'KFRA':
            mc.extend(dict(estimator=name, k=k, label_seed=seed, layer=n,
                           C_relative_error=relative(factors[n]['C'], exact[n]['C'])) for n in exact)
        print(f'{name} seed={seed}: {budget["build_seconds"]:.3f}s', flush=True)
    tables = {'curvature_error': pd.DataFrame(errors), 'whitening': pd.DataFrame(whites),
              'mc_convergence': pd.DataFrame(mc), 'compute_budget': pd.DataFrame(budgets)}
    tables['summary'] = tables['curvature_error'].groupby('estimator').mean(numeric_only=True).drop(columns='label_seed').reset_index()
    for name, table in tables.items():
        assert np.isfinite(table.select_dtypes(include='number').to_numpy()).all(), name
        table.to_csv(args.output/f'{name}.csv', index=False)
    print(tables['summary'].to_string(index=False))


if __name__ == '__main__':
    main()

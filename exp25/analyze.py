import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from exp25 import config as c


def analyze(smoke=False):
    folder = c.ROOT / 'results' / ('smoke' if smoke else 'formal')
    seeds = (42,) if smoke else c.SEEDS
    rows = {}
    for method in c.CONDITIONS:
        for seed in seeds:
            path = folder / f'{method}_{seed}.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(records) == (1 if smoke else c.EPOCHS), f'incomplete {path}'
            row = records[-1]
            assert row['accountant_steps'] == (1 if smoke else c.TOTAL_STEPS)
            assert (row['method'], row['seed'], row['smoke']) == (method, seed, smoke)
            rows[method, seed] = row
    summary = []
    for method in c.CONDITIONS:
        acc = np.array([rows[method, s]['test_accuracy'] for s in seeds])
        best = np.array([rows[method, s]['best_accuracy'] for s in seeds])
        delta = acc - np.array([rows[c.CONDITIONS[0], s]['test_accuracy'] for s in seeds])
        for seed in seeds:
            for field in ('classifier_initialization_sha256', 'private_order_sha256'):
                assert rows[method, seed][field] == rows[c.CONDITIONS[0], seed][field]
        ci = None
        if len(seeds) > 1:
            bootstrap = np.random.default_rng(25).choice(delta, (10000, len(seeds))).mean(1)
            ci = np.quantile(bootstrap, [.025, .975]).tolist()
        result = dict(method=method, n=len(seeds), accuracy_mean=acc.mean().item(),
            accuracy_std=acc.std(ddof=1).item() if len(seeds)>1 else None,
            best_accuracy_mean=best.mean().item(),
            best_accuracy_std=best.std(ddof=1).item() if len(seeds)>1 else None,
            paired_accuracy_delta=dict(zip(map(str, seeds), delta.tolist())),
            paired_delta_mean=delta.mean().item(), paired_bootstrap_95_ci=ci,
            oracle_non_deployable_upper_bound=c.condition(method)[1] == 'oracle')
        summary.append(result)
        std = f"{result['accuracy_std']*100:.3f}" if len(seeds)>1 else 'NA (smoke n=1)'
        print(f'{method}: {acc.mean()*100:.3f} ± {std}%; paired delta={delta.mean()*100:+.3f} pp')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    analyze(parser.parse_args().smoke)

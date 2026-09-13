"""Final epochs and descriptive paired comparisons (two seeds, no significance tests)."""
from pathlib import Path
import pandas as pd


def save(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'metrics.csv', index=False)
    keys = ['method', 'seed']
    totals = frame.groupby(keys).agg(
        total_training_seconds=('private_train_seconds', 'sum'),
        total_build_seconds=('preconditioner_build_seconds', 'sum'),
        total_runtime_seconds=('total_epoch_seconds', 'sum'),
        peak_allocated_bytes=('total_peak_cuda_allocated_bytes', 'max'))
    final = frame.groupby(keys, sort=False).tail(1).set_index(keys).join(totals).reset_index()
    final.to_csv(output / 'summary.csv', index=False)
    pairs = []
    for seed, group in final.groupby('seed'):
        group = group.set_index('method')
        for a, b in [('DP-KFLR', 'DP-KFC'), ('DP-KFRA-block', 'DP-KFC'), ('DP-KFLR', 'DP-KFRA-block')]:
            if a in group.index and b in group.index:
                pairs.append(dict(seed=seed, comparison=f'{a} minus {b}', **{
                    f'{k}_difference': group.loc[a, k] - group.loc[b, k]
                    for k in ('test_accuracy', 'clip_fraction', 'mean_clip_factor',
                              'total_training_seconds', 'total_runtime_seconds')}))
    pd.DataFrame(pairs).to_csv(output / 'paired_summary.csv', index=False)
    summary = final.groupby('method').agg(
        seeds=('seed', 'count'),
        final_accuracy_mean=('test_accuracy', 'mean'), final_accuracy_std=('test_accuracy', 'std'),
        clip_fraction_mean=('clip_fraction', 'mean'), clip_fraction_std=('clip_fraction', 'std'),
        mean_clip_factor_mean=('mean_clip_factor', 'mean'), mean_clip_factor_std=('mean_clip_factor', 'std'),
        preconditioner_build_time_mean=('total_build_seconds', 'mean'),
        total_runtime_mean=('total_runtime_seconds', 'mean'),
        peak_allocated_memory_mean=('peak_allocated_bytes', 'mean'),
        peak_allocated_memory_max=('peak_allocated_bytes', 'max'))
    summary.to_csv(output / 'method_summary.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    save(pd.read_csv(args.output / 'metrics.csv').to_dict('records'), args.output)

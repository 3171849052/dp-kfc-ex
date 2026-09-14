"""Final-epoch summaries and within-seed differences against beta=0.5."""
from pathlib import Path
import pandas as pd

MEASURES = ('test_accuracy', 'clip_fraction', 'transformed_norm_p90',
            'transformed_norm_p99', 'transformed_block_floored_condition_number')


def save(rows, geometry, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, geo = pd.DataFrame(rows), pd.DataFrame(geometry)
    frame.to_csv(output / 'metrics.csv', index=False)
    geo.to_csv(output / 'geometry.csv', index=False)
    keys = ['estimator', 'beta', 'seed']
    # Equal-weight layer mean; an all-zero factor keeps the result undefined.
    block = geo.groupby(keys + ['epoch']).floored_kappa_block.agg(lambda x: sum(x) / len(x))
    frame = frame.merge(block.rename('transformed_block_floored_condition_number').reset_index(),
                        on=keys + ['epoch'])
    totals = frame.groupby(keys).agg(
        total_build_seconds=('preconditioner_build_seconds', 'sum'),
        total_private_train_seconds=('private_train_seconds', 'sum'),
        total_algorithm_seconds=('algorithm_epoch_seconds', 'sum'),
        total_geometry_seconds=('geometry_seconds', 'sum'),
        total_state_metric_seconds=('state_metric_seconds', 'sum'),
        total_wall_seconds=('wall_epoch_seconds', 'sum'),
        peak_allocated_bytes=('total_peak_cuda_allocated_bytes', 'max'))
    final = frame.groupby(keys, sort=False).tail(1).set_index(keys).join(totals).reset_index()
    final.to_csv(output / 'summary.csv', index=False)
    pairs = []
    for (estimator, seed), group in final.groupby(['estimator', 'seed']):
        group = group.set_index('beta')
        if .5 in group.index:
            for beta in group.index:
                if beta != .5:
                    pairs.append(dict(estimator=estimator, seed=seed, beta=beta, reference_beta=.5, **{
                        f'{"accuracy" if k == "test_accuracy" else k}_difference': float(group.loc[beta, k]) - float(group.loc[.5, k])
                        for k in MEASURES}))
    pd.DataFrame(pairs, columns=['estimator', 'seed', 'beta', 'reference_beta'] +
                 [f'{"accuracy" if k == "test_accuracy" else k}_difference' for k in MEASURES]).to_csv(output / 'exponent_effects.csv', index=False)
    summary = []
    for (estimator, beta), group in final.groupby(['estimator', 'beta']):
        row = dict(estimator=estimator, beta=beta, seeds=len(group))
        for key in MEASURES:
            row[key + '_mean'] = sum(group[key]) / len(group)
            row[key + '_std'] = group[key].std()
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / 'estimator_beta_summary.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    save(pd.read_csv(args.output / 'metrics.csv').to_dict('records'),
         pd.read_csv(args.output / 'geometry.csv').to_dict('records'), args.output)

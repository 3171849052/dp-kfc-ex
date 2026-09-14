"""Aggregate final-epoch diagnostics and plot the damping sweep."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def analyze(smoke=False):
    directory = ROOT / 'results' / ('smoke' if smoke else 'formal')
    records = []
    for path in sorted(directory.glob('lambda*_seed*/summary.json')):
        record = json.loads(path.read_text())
        diagnostics = record.pop('diagnostics')
        for factor, values in diagnostics.items():
            record.update({f'{factor}/{key}': value for key, value in values.items()})
        records.append(record)
    frame = pd.DataFrame(records).sort_values(['lambda_damping', 'seed'])
    frame.to_csv(directory / 'summary.csv', index=False)
    columns = ['test_accuracy', 'test_loss', 'epsilon_spent', 'total_steps',
               'clip_fraction', 'preclip_norm_mean', 'preclip_norm_p90', 'preclip_norm_p99',
               'powell_rate', 'modified_rate', 'damping_trigger_rate']
    columns += [c for c in frame.columns if '/' in c]
    grouped = frame.groupby('lambda_damping')[columns].agg(['mean', 'std'])
    grouped.columns = ['_'.join(c) for c in grouped.columns]
    grouped.insert(0, 'seed_count', frame.groupby('lambda_damping').size())
    grouped.to_csv(directory / 'lambda_summary.csv')
    for filename, metrics, ylabel in [
        ('test_accuracy', ['test_accuracy'], 'Final test accuracy'),
        ('clip_fraction', ['clip_fraction'], 'Final epoch clip fraction'),
        ('preclip_norms', ['preclip_norm_p90', 'preclip_norm_p99'], 'Final epoch mean batch quantile'),
        ('damping_rates', ['powell_rate', 'modified_rate'], 'Cumulative trigger rate per eligible pair'),
    ]:
        fig, ax = plt.subplots()
        for metric in metrics:
            ax.errorbar(np.log10(grouped.index), grouped[metric + '_mean'],
                        yerr=grouped[metric + '_std'].fillna(0), marker='o', capsize=4, label=metric)
        ax.set(xlabel='log10(lambda_damping)', ylabel=ylabel,
               title='Smoke only; not a performance result' if smoke else '2-seed mean ± sample std')
        ax.legend()
        fig.savefig(directory / f'{filename}_vs_log10_lambda.png', bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    analyze(parser.parse_args().smoke)

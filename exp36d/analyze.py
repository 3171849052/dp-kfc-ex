"""Analyze all six completed runs; comparisons use final-epoch metrics."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36d.config import ROOT, RUNS, cfg
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_complete():
    rows = []
    for method, (family, power) in RUNS.items():
        directory = cfg.RESULTS / 'runs' / method
        config = json.loads((directory / 'config.json').read_text())
        assert config['method'] == method and config['epochs'] == 5
        assert config['p'] == power and config['max_grad_norm'] == 2
        assert config['damping'] == .001 and config['seed'] == 42
        frame = pd.read_csv(directory / 'metrics.csv')
        assert frame.epoch.tolist() == [1, 2, 3, 4, 5]
        assert frame.method.eq(method).all()
        assert frame.parameters_finite.all() and frame.parameters_updated.all()
        for key in ('logical_steps', 'optimizer_steps', 'noise_steps', 'accountant_steps'):
            assert frame[key].eq(frame.epoch * (50000 // 256)).all()
        rows.append(dict(frame.iloc[-1], family=family, p=power))
    return pd.DataFrame(rows)


def family_norms(summary):
    result = summary[['method', 'family', 'p']].copy()
    for family in ('attention_qkv', 'attention_out', 'patch_head', 'identity'):
        result[family] = summary['group_norm_' + family]
    for family in ('mlp_fc1', 'mlp_fc2'):
        columns = [c for c in summary if c.startswith('layer_norm_') and c.endswith('_' + family)]
        assert columns, family
        # Each layer metric is sqrt(mean squared norm); sum squares across layers.
        result[family] = summary[columns].pow(2).sum(axis=1).pow(.5)
    return result


def plot(summary, fields, destination, filename):
    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4), squeeze=False)
    baseline = summary.loc[summary.family == 'baseline'].iloc[0]
    for ax, field in zip(axes[0], fields):
        frame = summary.loc[summary.family == 'a_only'].sort_values('p')
        ax.plot(frame.p, frame[field], 'o-', label='DP-KFC-A')
        ax.axhline(baseline[field], color='black', linestyle='--', label='DP-AdamW, C=2')
        ax.set(xlabel='p', ylabel=field, xticks=[.1, .2, .3, .4, .5], title='Epoch 5; C=2')
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / filename, dpi=180)
    plt.close(fig)


def main():
    summary = load_complete()  # Never publish incomplete or fabricated results.
    destination = ROOT / 'results'
    summary.to_csv(destination / 'summary.csv', index=False)
    summary.loc[summary.family == 'a_only'].sort_values('p').to_csv(
        destination / 'a_only_summary.csv', index=False)
    fields = ('test_accuracy', 'test_loss', 'clip_fraction', 'mean_clip_factor',
              'transformed_norm_p90', 'transformed_norm_p99')
    for field in fields:
        plot(summary, [field], destination, f'{field}_vs_p.png')
    groups = ['group_norm_' + name for name in
              ('attention_qkv', 'attention_out', 'mlp', 'patch_head', 'identity')]
    plot(summary, groups, destination, 'group_norm_vs_p.png')
    columns = ['p', 'test_accuracy', 'clip_fraction', 'transformed_norm_p90']
    print(summary.loc[summary.family == 'a_only', columns].sort_values('p').to_string(index=False))
    norms = family_norms(summary)
    norms.to_csv(destination / 'layer_family_norms.csv', index=False)
    families = ['attention_qkv', 'attention_out', 'mlp_fc1', 'mlp_fc2', 'patch_head', 'identity']
    baseline = norms.loc[norms.family == 'baseline'].iloc[0]
    ratios = norms.copy()
    for family in families:
        ratios[family] /= baseline[family]
    ratios.to_csv(destination / 'layer_family_norm_ratios.csv', index=False)
    plot(ratios, families, destination, 'layer_family_norm_ratios_vs_p.png')


if __name__ == '__main__':
    main()

"""Validate the complete grid and compare epoch-five results."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33d import config as cfg
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAIRED_FIELDS = ['beta', 'test_accuracy', 'accuracy_auc', 'test_loss', 'clip_fraction',
    'mean_clip_factor', 'wiener_norm_ratio', 'cos_filtered_clean', 'nsr_filtered',
    'wiener_gain_median', 'wiener_gain_p90', 'wiener_gain_p99',
    'gain_frac_gt_001', 'gain_frac_gt_01', 'gain_frac_gt_05', 'gain_frac_gt_09',
    'signal_to_noise_median', 'signal_to_noise_p90', 'signal_to_noise_p99',
    'eig_zero_fraction', 'eig_near_zero_fraction']


def analyze(results=cfg.RESULTS):
    results = Path(results)
    assert {p.name for p in (results / 'runs').iterdir() if p.is_dir()} == set(cfg.RUNS)
    rows = []
    for name, spec in cfg.RUNS.items():
        directory = results / 'runs' / name
        configuration = json.loads((directory / 'config.json').read_text())
        assert configuration['method'] == name and configuration['beta'] == spec['beta']
        frame = pd.read_csv(directory / 'metrics.csv')
        assert frame.epoch.tolist() == list(range(1, 6)), name
        assert (frame.method == name).all() and (frame.logical_steps == 195).all()
        for key in ('accountant_steps', 'optimizer_steps', 'noise_events'):
            assert frame[key].tolist() == [195 * e for e in range(1, 6)]
        if spec['beta'] is None:
            assert frame.beta.isna().all()
        else:
            assert (frame.beta == spec['beta']).all()
            assert (frame.wiener_gain_min >= 0).all() and (frame.wiener_gain_max <= 1).all()
        rows.append(frame.iloc[-1].to_dict())
    summary = pd.DataFrame(rows)
    baseline = summary[summary.method == 'dp_adamw'].iloc[0]
    filtered = summary[summary.beta.notna()].sort_values('beta')
    paired = filtered[PAIRED_FIELDS].copy()
    paired['delta_accuracy_vs_adamw'] = paired.test_accuracy - baseline.test_accuracy
    summary.to_csv(results / 'summary.csv', index=False)
    paired.to_csv(results / 'paired.csv', index=False)

    def panel(ax, columns, identity=None):
        for column in columns:
            ax.plot(filtered.beta, filtered[column], 'o-', label=column)
            if pd.notna(baseline[column]):
                ax.axhline(baseline[column], linestyle='--', label=f'DP-AdamW {column}')
        if identity is not None:
            ax.axhline(identity, color='black', linestyle='--', label='DP-AdamW identity')
        ax.set_xscale('log')
        ax.set_xlabel('beta')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)

    plots = {
        'accuracy': (['test_accuracy'], None), 'accuracy_auc': (['accuracy_auc'], None),
        'test_loss': (['test_loss'], None), 'clip_fraction': (['clip_fraction'], None),
        'wiener_norm_ratio': (['wiener_norm_ratio'], 1),
        'gain_quantiles': ([f'wiener_gain_{s}' for s in ('median', 'p10', 'p90', 'p99')], 1),
        'gain_fraction': ([f'gain_frac_gt_{s}' for s in ('001', '01', '05', '09')], 1),
        'signal_to_noise': ([f'signal_to_noise_{s}' for s in ('median', 'p90', 'p99')], None),
        'cosine': (['cos_raw_clean', 'cos_filtered_clean'], None),
        'nsr': (['nsr_raw', 'nsr_filtered'], None),
    }
    for suffix, (columns, identity) in plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        panel(ax, columns, identity)
        if suffix == 'signal_to_noise':
            ax.set_title('Synthetic spectrum / actual DP variance; identity has no spectrum')
        fig.tight_layout()
        fig.savefig(results / f'beta_vs_{suffix}.png', dpi=160)
        plt.close(fig)
    for metric in ('update_norm', 'relative_update_norm'):
        fig, axes = plt.subplots(1, 5, figsize=(24, 4))
        for ax, group in zip(axes, cfg.UPDATE_GROUPS):
            panel(ax, [f'{metric}_{group}'])
            ax.set_title(group)
        fig.tight_layout()
        fig.savefig(results / f'beta_vs_{metric}.png', dpi=160)
        plt.close(fig)
    return summary, paired


if __name__ == '__main__':
    analyze()

"""Require the complete six-run grid before producing epoch-five comparisons."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33c import config as cfg
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(results=cfg.RESULTS):
    results = Path(results)
    assert {p.name for p in (results / 'runs').iterdir() if p.is_dir()} == set(cfg.RUNS)
    rows = []
    for name, spec in cfg.RUNS.items():
        run_dir = results / 'runs' / name
        configuration = json.loads((run_dir / 'config.json').read_text())
        assert configuration['method'] == name and configuration['gamma'] == spec['gamma']
        frame = pd.read_csv(run_dir / 'metrics.csv')
        assert frame.epoch.tolist() == list(range(1, 6)), name
        assert (frame.method == name).all() and (frame.gamma == spec['gamma']).all()
        assert (frame.logical_steps == 195).all()
        for key in ('accountant_steps', 'optimizer_steps', 'noise_events'):
            assert frame[key].tolist() == [195 * e for e in range(1, 6)]
        assert (frame.effective_gain_min >= 1 - spec['gamma'] - 1e-7).all()
        assert (frame.effective_gain_max <= 1 + 1e-7).all()
        row = frame.iloc[-1].to_dict()
        row.update(run_name=name, exp33_dp_adamw_accuracy=cfg.EXP33_DP_ADAMW_ACCURACY,
                   exp33_wiener_a_accuracy=cfg.EXP33_WIENER_A_ACCURACY)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values('gamma')
    fields = ['gamma', 'test_accuracy', 'accuracy_auc', 'test_loss', 'clip_fraction',
              'mean_clip_factor', 'wiener_norm_ratio', 'cos_raw_clean', 'cos_filtered_clean',
              'nsr_raw', 'nsr_filtered'] + [f'effective_gain_{s}' for s in ('mean', 'median', 'p10', 'p90', 'p99')]
    paired = summary[fields].copy()
    paired['delta_accuracy_vs_gamma0'] = paired.test_accuracy - paired.loc[paired.gamma == 0, 'test_accuracy'].item()
    summary.to_csv(results / 'summary.csv', index=False)
    paired.to_csv(results / 'paired.csv', index=False)

    def finish(fig, ax, suffix):
        ax.set_xlabel('gamma')
        ax.set_xscale('linear')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results / f'gamma_vs_{suffix}.png', dpi=160)
        plt.close(fig)

    for suffix, column in (('accuracy', 'test_accuracy'), ('accuracy_auc', 'accuracy_auc'),
                           ('test_loss', 'test_loss'), ('clip_fraction', 'clip_fraction'),
                           ('wiener_norm_ratio', 'wiener_norm_ratio')):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(summary.gamma, summary[column], 'o-', label=column)
        ax.set_ylabel(column)
        if suffix == 'accuracy':
            for gamma in (0, 1):
                point = summary[summary.gamma == gamma].iloc[0]
                ax.scatter([gamma], [point.test_accuracy], s=85, label=f'gamma={gamma}')
            ax.axhline(cfg.EXP33_DP_ADAMW_ACCURACY, color='black', linestyle='--', label='Exp33 DP-AdamW 0.9276')
            best = summary.loc[summary.test_accuracy.idxmax()]
            interior = summary[(summary.gamma > 0) & (summary.gamma < 1)].test_accuracy.max()
            endpoints = summary[summary.gamma.isin([0, 1])].test_accuracy.max()
            ax.set_title(f'Best gamma={best.gamma:g}; interior beats endpoints: {interior > endpoints}')
        finish(fig, ax, suffix)
    fig, ax = plt.subplots(figsize=(7, 4))
    for stat in ('mean', 'median', 'p99'):
        ax.plot(summary.gamma, summary[f'effective_gain_{stat}'], 'o-', label=stat)
    ax.fill_between(summary.gamma, summary.effective_gain_p10, summary.effective_gain_p90, alpha=.2, label='p10–p90')
    ax.set_ylabel('effective gain')
    finish(fig, ax, 'effective_gain')
    for suffix, prefix in (('cosine', 'cos'), ('nsr', 'nsr')):
        fig, ax = plt.subplots(figsize=(7, 4))
        for stage in ('raw', 'filtered'):
            key = f'{prefix}_{stage}' + ('_clean' if prefix == 'cos' else '')
            ax.plot(summary.gamma, summary[key], 'o-', label=stage)
        ax.set_ylabel(suffix)
        finish(fig, ax, suffix)
    for metric in ('update_norm', 'relative_update_norm'):
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        for ax, group in zip(axes, cfg.UPDATE_GROUPS):
            ax.plot(summary.gamma, summary[f'{metric}_{group}'], 'o-', label=group)
            ax.set_xlabel('gamma')
            ax.set_xscale('linear')
            ax.set_title(group)
            ax.set_ylabel(metric)
            ax.grid(alpha=.25)
        finish(fig, axes[-1], metric)
    return summary, paired


if __name__ == '__main__':
    analyze()

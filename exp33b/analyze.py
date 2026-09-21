"""Analyze only the completed eight-run grid; no partial-run summaries."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33b import config as cfg
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(results=cfg.RESULTS):
    results = Path(results)
    actual = {p.name for p in (results / 'runs').iterdir() if p.is_dir()}
    assert actual == set(cfg.RUNS), 'Expected exactly the eight formal run directories'
    rows = []
    for name, spec in cfg.RUNS.items():
        run_dir = results / 'runs' / name
        configuration = json.loads((run_dir / 'config.json').read_text())
        assert configuration['method'] == name
        assert configuration['linear_lr'] == spec['linear_lr']
        assert configuration['scale_mode'] == spec['scale_mode']
        frame = pd.read_csv(run_dir / 'metrics.csv')
        assert frame.epoch.tolist() == list(range(1, cfg.EPOCHS + 1)), name
        assert (frame.logical_steps == 195).all(), name
        for key in ('accountant_steps', 'optimizer_steps', 'noise_events'):
            assert frame[key].tolist() == [195 * e for e in range(1, 6)], (name, key)
        row = frame.iloc[-1].to_dict()
        assert row['method'] == name and row['linear_lr'] == spec['linear_lr']
        assert row['scale_mode'] == spec['scale_mode']
        row.update(run_name=name, exp33_dp_adamw_accuracy=cfg.EXP33_DP_ADAMW_ACCURACY)
        rows.append(row)
    summary = pd.DataFrame(rows)
    paired = []
    for lr in cfg.LINEAR_LRS:
        pair = summary[summary.linear_lr == lr].set_index('scale_mode')
        none, rms = pair.loc['none'], pair.loc['rms_match']
        row = {'linear_lr': lr}
        for field, column in (('accuracy', 'test_accuracy'), ('accuracy_auc', 'accuracy_auc'),
                              ('test_loss', 'test_loss'), ('clip_fraction', 'clip_fraction')):
            row[f'none_{field}'] = none[column]
            row[f'rms_match_{field}'] = rms[column]
        row.update(delta_rms_vs_none=rms.test_accuracy - none.test_accuracy,
                   exp33_dp_adamw_accuracy=cfg.EXP33_DP_ADAMW_ACCURACY,
                   delta_none_vs_exp33_adamw=none.test_accuracy - cfg.EXP33_DP_ADAMW_ACCURACY,
                   delta_rms_vs_exp33_adamw=rms.test_accuracy - cfg.EXP33_DP_ADAMW_ACCURACY)
        paired.append(row)
    summary.to_csv(results / 'summary.csv', index=False)
    pd.DataFrame(paired).to_csv(results / 'paired.csv', index=False)

    def curves(ax, column):
        for mode in cfg.SCALE_MODES:
            data = summary[summary.scale_mode == mode].sort_values('linear_lr')
            ax.plot(data.linear_lr, data[column], 'o-', label=mode)
        ax.set_xscale('log')
        ax.set_xlabel('Linear LR')
        ax.set_ylabel(column)
        ax.grid(alpha=.25)
        ax.legend()

    for suffix, column in (
        ('accuracy', 'test_accuracy'), ('accuracy_auc', 'accuracy_auc'),
        ('test_loss', 'test_loss'), ('clip_fraction', 'clip_fraction'),
        ('wiener_norm_ratio', 'wiener_norm_ratio'), ('nsr_filtered', 'nsr_filtered'),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        curves(ax, column)
        if column == 'test_accuracy':
            ax.axhline(cfg.EXP33_DP_ADAMW_ACCURACY, color='black', linestyle='--', label='Exp33 DP-AdamW (0.9276)')
            ax.legend()
        fig.tight_layout()
        fig.savefig(results / f'lr_vs_{suffix}.png', dpi=160)
        plt.close(fig)
    for metric in ('update_norm', 'relative_update_norm'):
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        for ax, group in zip(axes, cfg.UPDATE_GROUPS):
            curves(ax, f'{metric}_{group}')
            ax.set_title(group)
        fig.tight_layout()
        fig.savefig(results / f'lr_vs_{metric}.png', dpi=160)
        plt.close(fig)
    for prefix, filename in (('wiener_scale', 'scale_distribution_by_lr'),
                             ('effective_gain', 'effective_gain_by_lr')):
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode in cfg.SCALE_MODES:
            data = summary[summary.scale_mode == mode].sort_values('linear_lr')
            line, = ax.plot(data.linear_lr, data[f'{prefix}_median'], 'o-', label=f'{mode}: median')
            ax.fill_between(data.linear_lr, data[f'{prefix}_p10'], data[f'{prefix}_p90'],
                            alpha=.2, color=line.get_color(), label=f'{mode}: p10–p90')
            ax.plot(data.linear_lr, data[f'{prefix}_p99'], ':', color=line.get_color(), label=f'{mode}: p99')
        ax.set_xscale('log')
        ax.set_xlabel('Linear LR')
        ax.set_ylabel(prefix)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results / f'{filename}.png', dpi=160)
        plt.close(fig)
    return summary, pd.DataFrame(paired)


if __name__ == '__main__':
    analyze()

"""Require all seven complete trajectories before producing final comparisons."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp34 import config as cfg
from exp34.geometry import GROUPS
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(output=cfg.RESULTS):
    frames = []
    for _, method, damping in cfg.grid():
        frame = pd.read_csv(output / 'runs' / cfg.run_name(method, damping) / 'metrics.csv')
        assert frame.epoch.tolist() == list(range(1, cfg.EPOCHS + 1))
        assert (frame.method == method).all()
        assert frame.damping.isna().all() if damping is None else (frame.damping == damping).all()
        assert all(frame.iloc[-1][k] == cfg.TOTAL_STEPS for k in
                   ('logical_steps', 'accountant_steps', 'optimizer_steps', 'noise_events'))
        frames.append(frame)
    final = pd.concat([f.tail(1) for f in frames], ignore_index=True)
    final.to_csv(output / 'summary.csv', index=False)
    baseline = final[final.method == 'dp_adamw'].iloc[0]
    paired = []
    for damping in cfg.DAMPING_VALUES:
        row = {'damping': damping}
        for method in cfg.METHODS:
            record = baseline if method == 'dp_adamw' else final[(final.method == method) & (final.damping == damping)].iloc[0]
            row[f'{method}_accuracy'] = record.test_accuracy
            row[f'{method}_accuracy_auc'] = record.accuracy_auc
        for label, a, b in [('kfc_vs_adamw', 'dp_kfc', 'dp_adamw'),
                            ('kfc_a_vs_adamw', 'dp_kfc_a', 'dp_adamw'),
                            ('kfc_a_vs_kfc', 'dp_kfc_a', 'dp_kfc')]:
            row[f'delta_{label}'] = row[f'{a}_accuracy'] - row[f'{b}_accuracy']
        paired.append(row)
    pd.DataFrame(paired).to_csv(output / 'paired.csv', index=False)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=160)
        plt.close(fig)

    for metric, name in [('test_accuracy', 'accuracy_by_epoch'), ('test_loss', 'test_loss_by_epoch')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for frame in frames:
            r = frame.iloc[0]
            ax.plot(frame.epoch, frame[metric], label=cfg.run_name(r.method, r.damping))
        ax.set(xlabel='Epoch', ylabel=metric, xticks=range(1, 21))
        ax.legend(fontsize=7)
        save(fig, name)
    for metric, name in [('test_accuracy', 'final_accuracy'), ('accuracy_auc', 'accuracy_auc'),
                         ('clip_fraction', 'clip_fraction'), ('mean_clip_factor', 'mean_clip_factor'),
                         ('transformed_norm_p99', 'norm_p99')]:
        fig, ax = plt.subplots()
        for method in ('dp_kfc', 'dp_kfc_a'):
            subset = final[final.method == method].sort_values('damping')
            ax.plot(subset.damping, subset[metric], 'o-', label=cfg.LABELS[method])
        ax.axhline(baseline[metric], label='DP-AdamW', linestyle='--')
        ax.set(xscale='log', xlabel='Damping', ylabel=metric)
        ax.legend()
        save(fig, f'damping_vs_{name}')
    for metric in ('A_trace_per_dim', 'relative_damping_A', 'G_trace_per_dim', 'relative_damping_G'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, group in zip(axes.flat, GROUPS):
            for frame in frames:
                r = frame.iloc[0]
                if r.method == 'dp_adamw' or (metric.endswith('G') or metric.startswith('G_')) and r.method != 'dp_kfc':
                    continue
                ax.plot(frame.epoch, frame[f'{group}_median_{metric}'], label=cfg.run_name(r.method, r.damping))
            ax.set(title=group, xlabel='Epoch', ylabel=f'Median {metric}', yscale='log')
            ax.legend(fontsize=6)
        save(fig, f'{metric}_by_epoch')


if __name__ == '__main__':
    analyze()

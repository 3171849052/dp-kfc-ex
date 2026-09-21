"""Require all seven epoch-five results, then produce paired plots/tables."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp32c import config as cfg
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(results=cfg.RESULTS):
    rows = []
    for _, method, damping in cfg.grid():
        frame = pd.read_csv(results / 'runs' / cfg.run_name(method, damping) / 'metrics.csv')
        assert frame.epoch.tolist() == list(range(1, cfg.EPOCHS + 1))
        assert (frame.method == method).all()
        assert frame.damping.isna().all() if damping is None else (frame.damping == damping).all()
        rows.append(frame.iloc[-1].to_dict())
    summary = pd.DataFrame(rows)
    assert len(summary) == 7
    summary['p90'] = summary.transformed_norm_p90
    summary['p99'] = summary.transformed_norm_p99
    summary.to_csv(results / 'summary.csv', index=False)
    baseline = summary[summary.method == 'dp_adam'].iloc[0]
    paired = []
    for d in cfg.DAMPING_VALUES:
        subset = summary[summary.damping == d].set_index('method')
        k, a, b = subset.loc['dp_kfc', 'test_accuracy'], subset.loc['dp_kfc_a', 'test_accuracy'], baseline.test_accuracy
        paired.append(dict(damping=d, dp_adam_accuracy=b, dp_kfc_accuracy=k,
                           dp_kfc_a_accuracy=a, delta_kfc_vs_adam=k-b,
                           delta_kfc_a_vs_adam=a-b, delta_kfc_a_vs_kfc=a-k))
    pd.DataFrame(paired).to_csv(results / 'paired.csv', index=False)
    for metric, suffix in [('test_accuracy', 'accuracy'), ('test_loss', 'test_loss'), ('clip_fraction', 'clip_fraction'),
                           ('mean_clip_factor', 'mean_clip_factor'), ('transformed_norm_p99', 'norm_p99')]:
        fig, ax = plt.subplots()
        for method in ('dp_kfc', 'dp_kfc_a'):
            subset = summary[summary.method == method].sort_values('damping')
            ax.plot(subset.damping, subset[metric], marker='o', label=cfg.LABELS[method])
        ax.axhline(baseline[metric], linestyle='--', color='gray', label=cfg.LABELS['dp_adam'])
        ax.set(xscale='log', xlabel='Damping', ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results / f'damping_vs_{suffix}.png', dpi=160)
        plt.close(fig)
    return summary

if __name__ == '__main__':
    analyze()

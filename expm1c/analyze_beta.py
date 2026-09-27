"""Read completed runs, compare epoch five, and select each source independently."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1c import config as cfg
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(directory, source, beta, damping, origin):
    completion = json.loads((directory / 'complete.json').read_text())
    assert completion['run_name'] == directory.name and completion['epochs'] == 5
    configuration = json.loads((directory / 'config.json').read_text())
    assert configuration['task'] == 'vit' and configuration['seed'] == 42
    assert configuration['source'] == source and configuration['beta'] == beta
    assert configuration['damping'] == damping
    metrics = pd.read_csv(directory / 'metrics.csv').sort_values('epoch')
    assert metrics.epoch.tolist() == [1, 2, 3, 4, 5]
    final = metrics.iloc[-1]
    geometry = pd.read_csv(directory / 'geometry.csv')
    assert set(geometry.epoch) == {1, 2, 3, 4, 5}
    geometry = geometry[(geometry.epoch == 5) & (geometry.layer != 'identity')]
    row = dict(source=source, beta=beta, origin=origin, run_directory=str(directory),
               final_accuracy=final.test_accuracy, accuracy_auc=final.accuracy_auc,
               clip_fraction=final.clip_fraction, mean_clip_factor=final.mean_clip_factor,
               matched_raw_p90_ratio=final.matched_norm_p90 / final.raw_norm_p90)
    row['lambda'] = damping
    for key in ('condition_S', 'log_eigenvalue_spread_S', 'effective_rank_S'):
        row['median_' + key] = geometry[key].median() if key in geometry else np.nan
    assert np.isfinite([row[k] for k in ('final_accuracy', 'accuracy_auc', 'mean_clip_factor')]).all()
    return row


def reference(source):
    path = cfg.REPO_ROOT / 'expm1b/results/runs' / f'vit_dp_kfm_a_{source}_beta0.25_seed42'
    return summarize(path, source, .25, cfg.DAMPING, 'expm1b_reference')


def beta_table():
    rows = [summarize(s.directory, s.source, s.beta, s.damping, 'phase_a') for s in cfg.BETA_GRID]
    rows += [reference(source) for source in cfg.SOURCES]
    return pd.DataFrame(rows).sort_values(['source', 'beta']).reset_index(drop=True)


def select_betas(table):
    assert len(table) == 10
    selected = {}
    for source in cfg.SOURCES:
        part = table[table.source == source]
        assert len(part) == 5 and set(part.beta) == set(cfg.BETA_CANDIDATES)
        assert set(part['lambda']) == {cfg.DAMPING}
        winner = part.sort_values(['final_accuracy', 'accuracy_auc', 'mean_clip_factor', 'beta'],
                                 ascending=[False, False, False, True]).iloc[0]
        selected[source] = float(winner.beta)
    return selected


def plots(table, axis, output):
    output.mkdir(parents=True, exist_ok=True)
    for label, metric in (('accuracy', 'final_accuracy'), ('auc', 'accuracy_auc'),
                          ('clipping', 'clip_fraction'), ('mean_clip_factor', 'mean_clip_factor')):
        fig, ax = plt.subplots()
        for source, curve in table.groupby('source'):
            curve = curve.sort_values(axis)
            ax.plot(curve[axis], curve[metric], 'o-', label=source)
        if axis == 'lambda':
            ax.set_xscale('log')
        ax.set(xlabel=axis, ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{label}_vs_{axis}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots()
    for source, curve in table.groupby('source'):
        ax.plot(curve.median_condition_S, curve.final_accuracy, 'o-', label=source)
        for _, row in curve.iterrows():
            ax.annotate(f'{axis}={row[axis]:g}', (row.median_condition_S, row.final_accuracy), fontsize=7)
    ax.set(xlabel='median condition S (epoch 5 affine layers)', ylabel='final accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'accuracy_vs_median_condition.png', dpi=160)
    plt.close(fig)


def analyze():
    table = beta_table()  # Requires all eight completion markers and both references.
    selected = select_betas(table)
    cfg.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.RESULTS_ROOT / 'beta_summary.csv', index=False)
    plots(table, 'beta', cfg.RESULTS_ROOT / 'beta_plots')
    (cfg.RESULTS_ROOT / 'selected_beta.json').write_text(json.dumps(selected, indent=2) + '\n')
    print(json.dumps(selected))


if __name__ == '__main__':
    analyze()

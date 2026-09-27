"""Read 8 inverse runs and 9 existing references; write only below ExpM2."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm2 import config as cfg
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reference_path(source='none', beta=None):
    if beta is None:
        return cfg.REPO_ROOT / 'expm1/results/runs/vit_dp_sgd_none_seed42'
    return cfg.REPO_ROOT / 'expm1c/results/beta' / f'vit_dp_kfm_a_{source}_beta{beta:g}_lambda0.001_seed42'


def summarize(directory, source, beta, origin):
    completion = json.loads((directory / 'complete.json').read_text())
    assert completion['run_name'] == directory.name and completion['epochs'] == 5
    configuration = json.loads((directory / 'config.json').read_text())
    assert configuration['task'] == 'vit' and configuration['seed'] == 42
    assert configuration['source'] == source and configuration['beta'] == beta
    assert configuration['damping'] == cfg.DAMPING
    expected_method = {'inverse': 'inverse_dp_kfm_a', 'forward': 'dp_kfm_a', 'dp_sgd': 'dp_sgd'}[origin]
    assert configuration['method'] == expected_method
    if origin == 'inverse':
        assert configuration['geometry_power'] == 'inverse'
    metrics = pd.read_csv(directory / 'metrics.csv').sort_values('epoch')
    assert metrics.epoch.tolist() == [1, 2, 3, 4, 5]
    final = metrics.iloc[-1]
    row = final.to_dict()
    row.update(source=source, beta=beta, origin=origin, geometry_power=origin,
               run_directory=str(directory), final_accuracy=final.test_accuracy,
               matched_raw_p90_ratio=final.matched_norm_p90 / final.raw_norm_p90,
               signed_beta=0 if beta is None else (-beta if origin == 'inverse' else beta))
    row['lambda'] = cfg.DAMPING
    if origin == 'dp_sgd':
        row['median_condition_S'] = 1.
    else:
        geometry = pd.read_csv(directory / 'geometry.csv')
        assert set(geometry.epoch) == {1, 2, 3, 4, 5}
        geometry = geometry[(geometry.epoch == 5) & (geometry.layer != 'identity')]
        for key in ('condition_S', 'log_eigenvalue_spread_S', 'effective_rank_S'):
            row['median_' + key] = geometry[key].median()
    assert np.isfinite([row[k] for k in ('final_accuracy', 'accuracy_auc', 'mean_clip_factor')]).all()
    return row


def reference_rows():
    return [summarize(reference_path(s.source, s.beta), s.source, s.beta, 'forward') for s in cfg.GRID] + [
        summarize(reference_path(), 'none', None, 'dp_sgd')]


def paired_table(table):
    baseline = table[table.origin == 'dp_sgd'].iloc[0]
    rows = []
    for spec in cfg.GRID:
        pair = table[(table.source == spec.source) & (table.beta == spec.beta)].set_index('origin')
        inverse, forward = pair.loc['inverse'], pair.loc['forward']
        row = dict(source=spec.source, beta=spec.beta, dp_sgd_accuracy=baseline.final_accuracy,
                   delta_inverse_minus_forward=inverse.final_accuracy - forward.final_accuracy,
                   delta_inverse_minus_dp_sgd=inverse.final_accuracy - baseline.final_accuracy)
        for label, metric in (('accuracy', 'final_accuracy'), ('auc', 'accuracy_auc'),
                              ('clip_fraction', 'clip_fraction'), ('mean_clip_factor', 'mean_clip_factor'),
                              ('matched_raw_p90_ratio', 'matched_raw_p90_ratio'),
                              ('condition_S', 'median_condition_S'),
                              ('update_cos', 'update_cos'), ('update_rel_error', 'update_rel_error')):
            for origin, record in (('inverse', inverse), ('forward', forward)):
                row[f'{origin}_{label}'] = record[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def plots(table, output):
    output.mkdir(parents=True, exist_ok=True)
    baseline = table[table.origin == 'dp_sgd']
    for label, metric in (('accuracy', 'final_accuracy'), ('auc', 'accuracy_auc'),
                          ('clipping', 'clip_fraction'), ('mean_clip_factor', 'mean_clip_factor'),
                          ('matched_raw_p90_ratio', 'matched_raw_p90_ratio')):
        fig, ax = plt.subplots()
        for source in cfg.SOURCES:
            curve = pd.concat([table[table.source == source], baseline]).sort_values('signed_beta')
            ax.plot(curve.signed_beta, curve[metric], 'o-', label=source)
        ax.set(xlabel='signed beta (visualization only; DP-SGD at 0)', ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{label}_vs_signed_beta.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots()
    for (source, origin), curve in table[table.origin != 'dp_sgd'].groupby(['source', 'origin']):
        curve = curve.sort_values('beta')
        ax.plot(curve.median_condition_S, curve.final_accuracy, 'o-' if origin == 'inverse' else 's--', label=f'{source} {origin}')
        for _, row in curve.iterrows():
            ax.annotate(f'{row.beta:g}', (row.median_condition_S, row.final_accuracy), fontsize=7)
    ax.scatter(baseline.median_condition_S, baseline.final_accuracy, marker='*', label='DP-SGD')
    ax.set(xlabel='median condition S (epoch 5 affine layers)', ylabel='final accuracy', xscale='log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'accuracy_vs_condition.png', dpi=160)
    plt.close(fig)


def report(paired):
    lines = ['# Inverse versus forward (single seed 42)', '',
        'Research-only: unnoised private diagnostics; not a DP release.', '',
        'Beta magnitude, lambda, per-layer trace, total expected noise energy, privacy budget and training protocol are matched.',
        'Forward gives large-A directions larger clipping radii and noise standard deviations; inverse gives them smaller ones.',
        'Equal beta has equal condition magnitude for the SAME A, with reversed spectral ordering. Trained models may yield different A.',
        'Conditions below are epoch-five medians; AUC uses the inherited epoch-1-to-5 trapezoidal definition.',
        'Single-seed observations do not establish statistical significance.', '']
    for _, r in paired.iterrows():
        winner = 'inverse' if r.delta_inverse_minus_forward > 0 else ('forward' if r.delta_inverse_minus_forward < 0 else 'tie')
        lines.append(f'- {r.source}, beta={r.beta:g}: final-accuracy winner={winner}; inverse−forward accuracy={r.delta_inverse_minus_forward:+.4f}, '
                     f'AUC={r.inverse_auc-r.forward_auc:+.4f}, clip fraction={r.inverse_clip_fraction-r.forward_clip_fraction:+.4f}, '
                     f'mean clip factor={r.inverse_mean_clip_factor-r.forward_mean_clip_factor:+.4f}, '
                     f'update relative error={r.inverse_update_rel_error-r.forward_update_rel_error:+.4f}, '
                     f'update cosine={r.inverse_update_cos-r.forward_update_cos:+.4f}.')
    return '\n'.join(lines) + '\n'


def main():
    table = pd.DataFrame([summarize(s.directory, s.source, s.beta, 'inverse') for s in cfg.GRID] + reference_rows())
    assert len(table) == 17
    paired = paired_table(table)
    cfg.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.RESULTS_ROOT / 'summary.csv', index=False)
    paired.to_csv(cfg.RESULTS_ROOT / 'paired_summary.csv', index=False)
    plots(table, cfg.RESULTS_ROOT / 'plots')
    (cfg.RESULTS_ROOT / 'analysis.md').write_text(report(paired))


if __name__ == '__main__':
    main()

"""Damping curves with reused centers and explicit read-only reference rows."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1c import config as cfg
from expm1c.analyze_beta import beta_table, select_betas, summarize, reference, plots
import pandas as pd


def lambda_table(selected, centers):
    rows = [summarize(s.directory, s.source, s.beta, s.damping, 'phase_b')
            for s in cfg.lambda_grid(selected)]
    for source, beta in selected.items():
        center = centers[(centers.source == source) & (centers.beta == beta)]
        assert len(center) == 1
        rows.append(center.iloc[0].to_dict())
    table = pd.DataFrame(rows).sort_values(['source', 'lambda']).reset_index(drop=True)
    for source in cfg.SOURCES:
        part = table[table.source == source]
        assert len(part) == 4 and set(part['lambda']) == set(cfg.LAMBDA_CANDIDATES)
        assert set(part.beta) == {selected[source]}
    return table


def final_table(table, selected):
    rows = []
    for source in cfg.SOURCES:
        winner = table[table.source == source].sort_values(
            ['final_accuracy', 'accuracy_auc', 'mean_clip_factor', 'lambda'],
            ascending=[False, False, False, True]).iloc[0].to_dict()
        rows.append(winner | {'role': 'tuned_best', 'beta_star': selected[source],
                              'best_lambda': winner['lambda']})
        rows.append(reference(source) | {'role': 'expm1b_reference', 'beta_star': selected[source]})
    directory = cfg.REPO_ROOT / 'expm1/results/runs/vit_dp_sgd_none_seed42'
    rows.append(summarize(directory, 'none', None, cfg.DAMPING, 'expm1_reference') |
                {'role': 'dp_sgd_reference'})
    return pd.DataFrame(rows)


def analyze():
    selected = cfg.selected_betas()
    centers = beta_table()
    assert selected == select_betas(centers)
    table = lambda_table(selected, centers)
    final = final_table(table, selected)
    table.to_csv(cfg.RESULTS_ROOT / 'lambda_summary.csv', index=False)
    final.to_csv(cfg.RESULTS_ROOT / 'final_summary.csv', index=False)
    plots(table, 'lambda', cfg.RESULTS_ROOT / 'lambda_plots')
    print(final[['role', 'source', 'beta', 'lambda', 'final_accuracy', 'accuracy_auc']].to_string(index=False))


if __name__ == '__main__':
    analyze()

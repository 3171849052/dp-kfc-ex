"""Eight fresh runs plus two read-only ExpM1c every-epoch references."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1d import config as cfg
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DRIFT = ('median_A_cos_prev', 'median_A_rel_change_prev',
         'median_Q_cos_prev', 'median_Q_rel_change_prev')


def summarize(directory, source, frequency, origin):
    completion = json.loads((directory / 'complete.json').read_text())
    configuration = json.loads((directory / 'config.json').read_text())
    assert completion['run_name'] == directory.name and completion['epochs'] == 5
    assert configuration['task'] == 'vit' and configuration['method'] == 'dp_kfm_a'
    assert configuration['seed'] == 42 and configuration['source'] == source
    assert configuration['beta'] == .1 and configuration['damping'] == 1e-4
    metrics = pd.read_csv(directory / 'metrics.csv').sort_values('epoch')
    assert metrics.epoch.tolist() == [1, 2, 3, 4, 5]
    final = metrics.iloc[-1]
    geometry = pd.read_csv(directory / 'geometry.csv')
    geometry = geometry[(geometry.epoch == 5) & (geometry.layer != 'identity')]
    steps = cfg.refresh_steps(frequency)
    row = dict(source=source, frequency=frequency, geometry_build_count=len(steps),
               final_accuracy=final.test_accuracy, accuracy_auc=final.accuracy_auc,
               final_test_loss=final.test_loss, clip_fraction=final.clip_fraction,
               mean_clip_factor=final.mean_clip_factor,
               matched_raw_p90_ratio=final.matched_norm_p90 / final.raw_norm_p90,
               median_condition_S=geometry.condition_S.median(),
               total_geometry_build_seconds=metrics.geometry_build_seconds.sum(),
               total_training_seconds=metrics.private_train_seconds.sum(),
               total_runtime_seconds=final.run_wall_time_seconds,
               # Mean gap between actual builds; once has no observed interval.
               mean_refresh_interval_steps=float(np.mean(np.diff(steps))) if len(steps) > 1 else np.nan,
               origin=origin, run_directory=str(directory),
               **dict.fromkeys(DRIFT, np.nan))
    if origin == 'expm1d':
        assert configuration['frequency'] == frequency
        refresh = pd.read_csv(directory / 'refresh.csv')
        assert refresh.global_step.tolist() == list(steps)
        assert completion['geometry_build_count'] == len(steps)
        assert completion['oracle_build_count'] == 5
        row.update({key: refresh[key].median() for key in DRIFT})
        for key in ('total_geometry_build_seconds', 'total_training_seconds', 'total_runtime_seconds'):
            row[key] = completion[key]
    else:
        assert origin == 'expm1c_reference' and frequency == 'every_epoch'
        assert configuration['geometry_rebuild_every_epochs'] == 1
    row['geometry_runtime_fraction'] = row['total_geometry_build_seconds'] / row['total_runtime_seconds']
    return row


def frequency_table():
    rows = [summarize(run.directory, run.source, run.frequency, 'expm1d') for run in cfg.GRID]
    rows += [summarize(cfg.reference_directory(source), source, 'every_epoch', 'expm1c_reference')
             for source in cfg.SOURCES]
    table = pd.DataFrame(rows).sort_values(['source', 'geometry_build_count']).reset_index(drop=True)
    assert len(table) == 10
    for source in cfg.SOURCES:
        assert set(table[table.source == source].frequency) == set(cfg.FREQUENCIES)
    return table


def plots(table, output):
    output.mkdir(parents=True, exist_ok=True)
    for filename, x, y in (
        ('accuracy_vs_build_count', 'geometry_build_count', 'final_accuracy'),
        ('auc_vs_build_count', 'geometry_build_count', 'accuracy_auc'),
        ('clipping_vs_build_count', 'geometry_build_count', 'clip_fraction'),
        ('accuracy_vs_geometry_runtime', 'total_geometry_build_seconds', 'final_accuracy'),
    ):
        fig, ax = plt.subplots()
        for source, curve in table.groupby('source'):
            curve = curve.sort_values(x)
            ax.plot(curve[x], curve[y], 'o-', label=source)
            for _, row in curve.iterrows():
                ax.annotate(row.frequency, (row[x], row[y]), fontsize=6)
        ax.set(xlabel=x, ylabel=y)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{filename}.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, DRIFT):
        for source, curve in table.groupby('source'):
            curve = curve.dropna(subset=[metric, 'mean_refresh_interval_steps']).sort_values('mean_refresh_interval_steps')
            ax.plot(curve.mean_refresh_interval_steps, curve[metric], 'o-', label=source)
        ax.set(xlabel='mean interval between builds (private steps)', ylabel=metric)
        ax.legend()
    fig.suptitle('Training geometry drift; once and reference drift unavailable')
    fig.tight_layout()
    fig.savefig(output / 'geometry_drift_vs_refresh_interval.png', dpi=160)
    plt.close(fig)


def analyze():
    table = frequency_table()
    cfg.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.RESULTS_ROOT / 'frequency_summary.csv', index=False)
    plots(table, cfg.RESULTS_ROOT / 'plots')
    for source, part in table.groupby('source'):
        best = part[part.final_accuracy == part.final_accuracy.max()]
        print(f'{source}: highest final accuracy {best.final_accuracy.iloc[0]:.6f}; frequency={", ".join(best.frequency)}')
    print('Single seed (42): small differences are not evidence of statistical significance.')


if __name__ == '__main__':
    analyze()

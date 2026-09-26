"""Compare 16 layer-normalized runs against read-only ExpM1 references."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1b import config as cfg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

GLOBAL_RESULTS = cfg.REPO_ROOT / 'expm1' / 'results'


def load_run(root: Path, name: str, normalization: str):
    directory = root / 'runs' / name
    complete = json.loads((directory / 'complete.json').read_text())
    assert complete['run_name'] == name and complete['epochs'] == 5
    frames = []
    for filename in ('metrics', 'geometry', 'layer_groups'):
        frame = pd.read_csv(directory / f'{filename}.csv')
        assert set(frame.epoch) == {1, 2, 3, 4, 5}, directory
        if filename == 'metrics':
            assert len(frame) == 5
        frame['run_name'] = name
        frame['normalization'] = normalization
        frames.append(frame)
    return frames


def write_outputs(metrics, geometry, groups, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    final = metrics.sort_values('epoch').groupby(['normalization', 'run_name']).tail(1).copy()
    final['matched_raw_p90_ratio'] = final.matched_norm_p90 / final.raw_norm_p90
    final.to_csv(output / 'final_summary.csv', index=False)
    metrics.to_csv(output / 'all_metrics.csv', index=False)
    geometry.to_csv(output / 'geometry_summary.csv', index=False)
    groups.to_csv(output / 'layer_group_summary.csv', index=False)
    pairs = []
    for spec in cfg.FORMAL_GRID:
        layer = final[(final.run_name == spec.name) & (final.normalization == 'layer')].iloc[0]
        glob = final[(final.run_name == spec.name) & (final.normalization == 'global')].iloc[0]
        sgd = final[(final.task == spec.task) & (final.method == 'dp_sgd')].iloc[0]
        row = dict(task=spec.task, method=spec.method, source=spec.source, beta=spec.beta,
                   dp_sgd_accuracy=sgd.test_accuracy, expm1_global_accuracy=glob.test_accuracy,
                   expm1b_layer_accuracy=layer.test_accuracy,
                   delta_layer_minus_global=layer.test_accuracy-glob.test_accuracy,
                   delta_layer_minus_sgd=layer.test_accuracy-sgd.test_accuracy)
        for metric in ('clip_fraction', 'mean_clip_factor', 'matched_raw_p90_ratio'):
            row[metric+'_global'] = glob[metric]
            row[metric+'_layer'] = layer[metric]
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(output / 'paired_summary.csv', index=False)

    for task in cfg.TASKS:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, metric in zip(axes, ('test_accuracy', 'clip_fraction', 'mean_clip_factor')):
            part = final[(final.task == task) & (final.method != 'dp_sgd')]
            for (method, source, norm), curve in part.groupby(['method', 'source', 'normalization']):
                curve = curve.sort_values('beta')
                ax.plot(curve.beta, curve[metric], marker='o', linestyle='-' if norm == 'layer' else '--',
                        label=f'{method}/{source}/{norm}')
            reference = final[(final.task == task) & (final.method == 'dp_sgd')].iloc[0]
            ax.axhline(reference[metric], color='black', label='DP-SGD')
            ax.set(xlabel='beta', ylabel=metric, title=task, xticks=cfg.BETAS)
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / f'{task}_accuracy_clipping_vs_beta.png', dpi=160)
        plt.close(fig)

    last_groups = groups[groups.epoch == 5]
    for task in cfg.TASKS:
        for method in cfg.METHODS:
            for source in cfg.SOURCES:
                fig, axes = plt.subplots(2, 2, figsize=(15, 9))
                for col, beta in enumerate(cfg.BETAS):
                    selected = last_groups[(last_groups.task == task) & (last_groups.level == 'group') &
                        (((last_groups.method == method) & (last_groups.source == source) &
                          (last_groups.beta == beta)) | (last_groups.method == 'dp_sgd'))]
                    for ax, metric in zip(axes[:, col], ('snr', 'noise_energy_share')):
                        table = selected.pivot(index='group', columns='normalization', values=metric)
                        table.plot.bar(ax=ax)
                        ax.set(title=f'{task}/{method}/{source}/beta={beta}', ylabel=metric)
                        if metric == 'snr':
                            ax.set_yscale('log')
                fig.tight_layout()
                fig.savefig(output / f'{task}_{method}_{source}_snr_noise_share.png', dpi=160)
                plt.close(fig)

    focus_name = 'vit_dp_kfm_a_pink_beta0.25_seed42'
    focus = metrics[(metrics.run_name == focus_name) |
                    ((metrics.task == 'vit') & (metrics.method == 'dp_sgd'))]
    focus.to_csv(output / 'vit_kfm_a_pink_beta025_diagnostics.csv', index=False)
    for frame, suffix in ((geometry, 'geometry'), (groups, 'layer_groups')):
        frame[(frame.run_name == focus_name) | ((frame.task == 'vit') & (frame.method == 'dp_sgd'))].to_csv(
            output / f'vit_kfm_a_pink_beta025_{suffix}.csv', index=False)
    print(final[(final.run_name == focus_name) | ((final.task == 'vit') & (final.method == 'dp_sgd'))][
        ['normalization', 'test_accuracy', 'clip_fraction', 'mean_clip_factor',
         'matched_raw_p90_ratio', 'total_noise_rms', 'clip_cos', 'signal_cos']].to_string(index=False))


def analyze():
    loaded = []
    for spec in cfg.FORMAL_GRID:
        loaded.append(load_run(cfg.RESULTS_ROOT, spec.name, 'layer'))
        loaded.append(load_run(GLOBAL_RESULTS, spec.name, 'global'))
    for task in cfg.TASKS:
        loaded.append(load_run(GLOBAL_RESULTS, f'{task}_dp_sgd_none_seed42', 'dp_sgd'))
    frames = [pd.concat([run[index] for run in loaded], ignore_index=True) for index in range(3)]
    write_outputs(*frames, cfg.RESULTS_ROOT)


if __name__ == '__main__':
    analyze()

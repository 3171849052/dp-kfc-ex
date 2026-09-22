"""Read six measured runs; no training, imputation, or synthetic result rows."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36c.config import cfg
from exp36b.analyze import family_columns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = {
    'DP-AdamW': 'exp36b/results/runs/dp_adamw',
    'Pink alpha=.25': 'exp35b/results/vit/runs/dp_kfc_a_alpha025',
    'Oracle alpha=.25': 'exp36c/results/runs/oracle_dp_kfc_a_alpha025',
    'Pink alpha=.5': 'exp35/results/vit/runs/dp_kfc_a_mix05',
    'Oracle alpha=.5': 'exp36c/results/runs/oracle_dp_kfc_a_alpha05',
    'Oracle raw A-only': 'exp36b/results/runs/oracle_dp_kfc_a',
}


def read_run(relative):
    directory = cfg.ROOT.parent / relative
    frame = pd.read_csv(directory / 'metrics.csv')
    config = json.loads((directory / 'config.json').read_text())
    assert config['model'] == cfg.MODEL_NAME
    assert (config['seed'], config['epochs'], config['logical_batch_size'],
            config['physical_batch_size']) == (42, 5, 256, 128)
    assert frame.epoch.tolist() == [1, 2, 3, 4, 5]
    return frame


def main():
    frames = {label: read_run(path) for label, path in RUNS.items()}
    summary = pd.DataFrame([dict(label=label, source=RUNS[label], seed=42,
        privacy_status='oracle_non_private_geometry' if label.startswith('Oracle') else 'standard_dp',
        nominal_dp_epsilon=3, final_accuracy=frame.test_accuracy.iloc[-1],
        best_accuracy=frame.test_accuracy.max(), final_test_loss=frame.test_loss.iloc[-1])
        for label, frame in frames.items()])
    summary.to_csv(cfg.RESULTS / 'summary.csv', index=False)
    differences = []
    for alpha, suffix in ((.25, '.25'), (.5, '.5')):
        oracle, pink = (frames[f'{kind} alpha={suffix}'].set_index('epoch') for kind in ('Oracle', 'Pink'))
        for epoch in oracle.index:
            differences.append(dict(alpha=alpha, epoch=epoch,
                oracle_accuracy=oracle.loc[epoch, 'test_accuracy'],
                pink_accuracy=pink.loc[epoch, 'test_accuracy'],
                accuracy_difference=oracle.loc[epoch, 'test_accuracy']-pink.loc[epoch, 'test_accuracy'],
                best_accuracy_difference=oracle.loc[:epoch, 'test_accuracy'].max()-pink.loc[:epoch, 'test_accuracy'].max()))
    comparison = pd.DataFrame(differences)
    comparison.to_csv(cfg.RESULTS / 'oracle_vs_pink.csv', index=False)
    fields = ['test_accuracy', 'test_loss', 'clip_fraction', 'mean_clip_factor',
              'transformed_norm_p90', 'transformed_norm_p99']
    fields += [c for c in frames['DP-AdamW'] if c.startswith('group_norm_')]
    figures = cfg.RESULTS / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for field in fields:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, frame in frames.items():
            ax.plot(frame.epoch, frame[field], marker='o', label=label)
        ax.set(xlabel='Epoch', ylabel=field, title='Oracle uses non-private geometry; nominal training epsilon=3')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f'{field}.png', dpi=150)
        plt.close(fig)
    baseline = frames['DP-AdamW'].set_index('epoch')
    ratios = []
    for label, frame in frames.items():
        for family, columns in family_columns(frame).items():
            assert columns, family
            for _, row in frame.iterrows():
                numerator = np.sqrt(np.square(row[columns].astype(float)).sum())
                denominator = np.sqrt(np.square(baseline.loc[row.epoch, columns].astype(float)).sum())
                ratios.append(dict(label=label, epoch=int(row.epoch),
                    family='attention_out' if family == 'attention_out_proj' else family,
                    transformed_norm=numerator, dp_adamw_norm=denominator, ratio=numerator/denominator))
    pd.DataFrame(ratios).to_csv(cfg.RESULTS / 'layer_family_norm_ratios.csv', index=False)
    print(summary.to_string(index=False))
    print(comparison[comparison.epoch == 5].to_string(index=False))
    print('Accuracy differences are fractions (multiply by 100 for percentage points).')
    print('Oracle geometry is non-private: nominal epsilon is not end-to-end DP.')


if __name__ == '__main__':
    main()

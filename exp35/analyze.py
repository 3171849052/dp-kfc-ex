"""Publish analysis only after all eight five-epoch runs are complete."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp35 import ROOT
from exp35.config import grid
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_complete():
    runs = {}
    for dataset in ('mnist', 'vit'):
        for method in grid(dataset):
            directory = ROOT / 'results' / dataset / 'runs' / method
            config = json.loads((directory / 'config.json').read_text())
            frame = pd.read_csv(directory / 'metrics.csv')
            assert config['method'] == method and config['epochs'] == 5
            assert frame.epoch.tolist() == [1,2,3,4,5], str(directory)
            assert frame.method.eq(method).all()
            assert frame.parameters_finite.all() and frame.parameters_updated.all()
            expected_steps = frame.epoch * ((60000 if dataset == 'mnist' else 50000) // 256)
            for field in ('logical_steps', 'optimizer_steps', 'noise_steps', 'accountant_steps'):
                assert frame[field].eq(expected_steps).all(), (directory, field)
            runs[dataset, method] = frame
    return runs


def main():
    runs = load_complete()  # No plots or summaries are written before this gate.
    summaries = []
    for dataset in ('mnist', 'vit'):
        destination = ROOT / 'results' / dataset
        summary = pd.DataFrame([dict(runs[dataset,m].iloc[-1], dataset=dataset) for m in grid(dataset)])
        summary.to_csv(destination / 'summary.csv', index=False)
        summaries.append(summary)
        for field in ('test_accuracy', 'clip_fraction', 'mean_clip_factor', 'transformed_norm_p90', 'transformed_norm_p99'):
            fig, ax = plt.subplots(figsize=(7,4))
            for method in grid(dataset):
                frame = runs[dataset,method]
                ax.plot(frame.epoch, frame[field], marker='o', label=method)
            ax.set(xlabel='Epoch', ylabel=field, xticks=range(1,6), title=dataset.upper())
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / f'{field}.png', dpi=180)
            plt.close(fig)
    pd.concat(summaries, ignore_index=True).to_csv(ROOT / 'results' / 'summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8,5))
    for method in grid('vit'):
        frame = runs['vit',method]
        ax.scatter(frame.clip_fraction, frame.test_accuracy, label=method)
        for row in frame.itertuples():
            ax.annotate(str(row.epoch), (row.clip_fraction, row.test_accuracy))
    ax.set(xlabel='clip_fraction', ylabel='test_accuracy', title='ViT: labels indicate epoch')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'results' / 'vit' / 'clip_fraction_vs_test_accuracy.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

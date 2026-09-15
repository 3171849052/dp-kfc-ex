"""Render formal factor geometry; no optimizer accuracy comparison."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parent/'results/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).resolve().parent/'results/formal'
    out = root/'plots'
    out.mkdir(exist_ok=True)
    factors = pd.read_csv(root/'factors.csv')
    ratios = pd.read_csv(root/'transformed_conditions.csv')
    spectra = pd.read_csv(root/'generalized_spectra.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, side in zip(axes, ('A', 'G')):
        for epoch, frame in factors[factors.side == side].groupby('epoch'):
            ax.plot(frame.layer, frame.condition_F, marker='o', label=f'F epoch {epoch}')
            ax.plot(frame.layer, frame.condition_B, marker='x', linestyle='--', label=f'B epoch {epoch}')
        ax.set(title=side, yscale='log', ylabel='Condition number')
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'condition_vs_depth.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ('matrix_cosine', 'top_k_overlap', 'shape_error')):
        for (layer, side), frame in factors.groupby(['layer', 'side']):
            ax.plot(frame.epoch, frame[metric], marker='o', label=f'{layer}/{side}')
        ax.set(title=metric, xlabel='Epoch'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'alignment_vs_epoch.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, ((layer, side), frame) in zip(axes.flat, spectra.groupby(['layer', 'side'], sort=False)):
        for epoch, sub in frame.groupby('epoch'):
            ax.plot(sub['index'], sub.eigenvalue, label=f'Epoch {epoch}')
        ax.set(title=f'{layer}/{side}', yscale='log', xlabel='Eigenvalue index'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'generalized_spectrum.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, side in zip(axes, ('A', 'G')):
        for (layer, q), frame in ratios[ratios.side == side].groupby(['layer', 'q']):
            ax.plot(frame.epoch, frame.condition_ratio, marker='o', label=f'{layer} q={q}')
        ax.axhline(1, color='black', linestyle=':')
        ax.set(title=side, yscale='log', xlabel='Epoch', ylabel='Transformed / original condition'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'transformed_condition_ratio.png'); plt.close(fig)


if __name__ == '__main__':
    main()

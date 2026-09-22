"""Offline plots; private diagnostics never select a training setting."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36.runtime import ROOT, setup


def main():
    setup()
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layers = pd.read_csv(ROOT/'results/layer_metrics.csv')
    groups = pd.read_csv(ROOT/'results/group_metrics.csv')
    assert sorted(layers.epoch.unique()) == list(range(6))
    destination = ROOT/'results/figures'
    destination.mkdir(parents=True, exist_ok=True)
    pink = layers[layers.source == 'pink']
    order = pink[pink.epoch == 0].layer.tolist()
    for metric in ('cos_A', 'cos_P'):
        table = pink.pivot(index='layer', columns='epoch', values=metric).loc[order]
        fig, ax = plt.subplots(figsize=(8,18))
        im = ax.imshow(table, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set(yticks=range(len(table)), yticklabels=table.index, xticks=range(6), xlabel='Epoch', title=f'pink {metric}')
        ax.tick_params(axis='y', labelsize=6)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(destination/f'pink_{metric}_heatmap.png', dpi=180)
        plt.close(fig)
    families = ['patch_embed','attention_qkv','attention_out','mlp_fc1','mlp_fc2','head']
    for metric in ('cos_A', 'cos_P', 'rel_frob_A', 'rel_frob_P'):
        fig, axes = plt.subplots(2,3,figsize=(14,8))
        for ax, group in zip(axes.flat, families):
            for source, frame in groups[groups.family == group].groupby('source'):
                ax.plot(frame.epoch, frame[metric], 'o-', label=source)
            ax.set(title=group, xlabel='Epoch', ylabel=metric, xticks=range(6))
        axes.flat[0].legend()
        fig.tight_layout()
        fig.savefig(destination/f'group_{metric}.png', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    for ax, metric in zip(axes, ('cos_A','cos_P')):
        for epoch, frame in groups[groups.source == 'pink'].groupby('epoch'):
            frame = frame.set_index('family').loc[families]
            ax.plot(families, frame[metric], 'o-', label=f'epoch {epoch}')
        ax.set(ylabel=metric, title='Pink alignment by family')
        ax.tick_params(axis='x', rotation=35)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(destination/'pink_family_alignment.png', dpi=180)
    plt.close(fig)
    representatives = ['patch_embed','blocks.0.attn.q_proj','blocks.5.attn.out_proj',
                       'blocks.5.mlp.fc1','blocks.11.mlp.fc2','head']
    for epoch in (0,5):
        spectra = np.load(ROOT/f'results/spectra/epoch_{epoch}.npz')
        fig, axes = plt.subplots(2,3,figsize=(14,8))
        for ax, layer in zip(axes.flat, representatives):
            for source in ('private_ref','private_replica','pink','white'):
                values = spectra[source+'::'+layer][::-1]
                ax.semilogy(np.arange(1,len(values)+1), values, label=source)
            ax.set(title=layer, xlabel='Eigenvalue rank', ylabel='Eigenvalue')
        axes.flat[0].legend()
        fig.suptitle(f'Epoch {epoch}')
        fig.tight_layout()
        fig.savefig(destination/f'spectrum_epoch_{epoch}.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()

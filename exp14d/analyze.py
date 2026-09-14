"""Final-epoch summaries and empirical-whitening plots."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(rows, whitening, profiles, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, layers, profile = map(pd.DataFrame, (rows, whitening, profiles))
    metrics.to_csv(output/'metrics.csv', index=False)
    layers.to_csv(output/'whitening.csv', index=False)
    profile.to_csv(output/'whitening_profile.csv', index=False)
    final = metrics.sort_values('epoch').groupby(['beta', 'seed'], as_index=False).tail(1)
    final.to_csv(output/'summary.csv', index=False)
    columns = ['test_accuracy', 'test_loss', 'train_loss', 'epsilon', 'clip_fraction',
               'transformed_norm_p50', 'transformed_norm_p90', 'transformed_norm_p99',
               'scale_match', 'W_global', 'top10_energy_share', 'directions_for_90pct',
               'probe_sample_count']
    summary = final.groupby('beta')[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(output/'beta_summary.csv')
    means = final.groupby('beta')[columns].mean()

    def plot(x, y, filename, xlabel, ylabel, annotate=False):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(means.index if x == 'beta' else means[x], means[y], 'o-')
        ax.scatter(final.beta if x == 'beta' else final[x], final[y], alpha=.4, s=18)
        if annotate:
            for beta, row in means.iterrows():
                ax.annotate(f'{beta:g}', (row[x], row[y]), xytext=(4, 4), textcoords='offset points')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(output/filename, dpi=160)
        plt.close(fig)

    plot('beta', 'W_global', 'isotropy_vs_beta.png', 'beta', 'Global isotropy W')
    plot('W_global', 'test_accuracy', 'accuracy_vs_isotropy.png', 'Global isotropy W', 'Test accuracy', True)
    plot('W_global', 'clip_fraction', 'clip_fraction_vs_isotropy.png', 'Global isotropy W', 'Clip fraction', True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ('top10_energy_share', 'directions_for_90pct'):
        ax.plot(means.index, means[key], 'o-', label=key)
    ax.set(xlabel='beta', ylabel='Direction-count weighted layer metric')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'energy_concentration_vs_beta.png', dpi=160)
    plt.close(fig)
    final_profile = profile.merge(final[['beta', 'seed', 'epoch']], on=['beta', 'seed', 'epoch'])
    curves = final_profile.groupby(['beta', 'direction_percentile']).normalized_energy.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for beta, curve in curves.groupby(level='beta'):
        ax.plot(curve.index.get_level_values('direction_percentile'), curve, label=f'beta={beta:g}')
    ax.axhline(1, color='black', linestyle='--', label='Perfect isotropy')
    ax.set(xlabel='Direction percentile (descending energy)', ylabel='e / mean(e)', yscale='log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'ranked_energy_profile.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).parent/'results/formal')
    output = parser.parse_args().output
    save(*(pd.read_csv(output/name).to_dict('records') for name in
           ('metrics.csv', 'whitening.csv', 'whitening_profile.csv')), output)

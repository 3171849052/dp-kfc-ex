"""Complete-grid, seed-paired descriptive analysis. No significance claims."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp23 import HERE
from exp23.config import METHODS, SEEDS, EPOCHS, RESEARCH_ONLY


def contrasts(wide):
    pa = wide.a_public_fashion-wide.a_public_cifar10
    pf = wide.full_public_fashion-wide.full_public_cifar10
    result = {'Penalty_A': pa, 'Penalty_Full': pf, 'Interaction': pf-pa,
              'Fashion_A_minus_Full': wide.a_public_fashion-wide.full_public_fashion,
              'CIFAR_A_minus_Full': wide.a_public_cifar10-wide.full_public_cifar10}
    for kind in ('a', 'full'):
        for source in ('fashion', 'cifar10'):
            result[f'{kind}_{source}_minus_pink'] = wide[f'{kind}_public_{source}']-wide[f'{kind}_pink']
    return pd.DataFrame(result)


def analyze(smoke):
    output = HERE/'results'/('smoke' if smoke else 'formal')
    seeds = (42,) if smoke else SEEDS
    frames, geometries, summaries, configs = [], [], [], []
    for seed in seeds:
        for method in METHODS:
            directory = output/'runs'/f'{method}_{seed}'
            frame = pd.read_csv(directory/'metrics.csv')
            configuration = json.loads((directory/'config.json').read_text())
            assert frame.epoch.tolist() == list(range(1, (1 if smoke else EPOCHS)+1))
            assert frame.method.eq(method).all() and frame.seed.eq(seed).all()
            assert configuration['smoke'] == smoke and configuration['calibration_samples'] == 2560
            expected = 0 if method == 'dp_sgd' else 2560
            assert frame.builder_samples.eq(expected).all()
            assert frame.builder_forward_calls.eq(0 if method == 'dp_sgd' else 10).all()
            assert frame.builder_vjp_calls.eq(10 if method.startswith('full_') else 0).all()
            assert frame.builder_reverse_vectors.eq(2560 if method.startswith('full_') else 0).all()
            assert frame.builder_backward_calls.eq(0).all()
            assert frame.batches.eq(2 if smoke else 234).all()
            assert frame.noise_multiplier.gt(0).all()
            if not smoke:
                assert frame.epsilon.iloc[-1] <= 1.
            if method.startswith('a_'):
                np.testing.assert_allclose(frame.scale_match**2*frame.m_p, frame.m_reference, rtol=1e-12)
            if method != 'dp_sgd':
                geometry = pd.read_csv(directory/'geometry.csv')
                assert len(geometry) == 4*len(frame)
                assert np.isfinite(geometry[['cosA', 'relative_frobenius_A']]).all().all()
                if method.startswith('full_'):
                    assert np.isfinite(geometry[['cosG', 'relative_frobenius_G']]).all().all()
                geometries.append(geometry)
            frames.append(frame)
            summaries.append(json.loads((directory/'summary.json').read_text()))
            configs.append(configuration)
        paired_configs = [c for c in configs if c['seed'] == seed]
        for key in ('initialization_hash', 'final_noise_rng_hash', 'noise_multiplier'):
            assert len({c[key] for c in paired_configs}) == 1
        if smoke:
            assert all(c['private_batch_hashes'] == paired_configs[0]['private_batch_hashes'] for c in paired_configs)
    metrics, geometry, summary = pd.concat(frames), pd.concat(geometries), pd.DataFrame(summaries)
    summary.to_csv(output/'summary.csv', index=False)
    metrics.to_csv(output/'metrics.csv', index=False)
    geometry.to_csv(output/'geometry.csv', index=False)
    columns = [c for c in summary.select_dtypes('number') if c != 'seed']
    grouped = summary.groupby(['method', 'source'])[columns].agg(['mean', 'std'])
    grouped.columns = [f'{metric}_{"sample_std" if stat == "std" else stat}' for metric, stat in grouped.columns]
    grouped.to_csv(output/'method_source_summary.csv')
    pairs = []
    for metric in ('final_accuracy', 'best_accuracy', 'accuracy_auc'):
        wide = summary.pivot(index='seed', columns='method', values=metric).loc[list(seeds)]
        deltas = contrasts(wide)
        deltas.to_csv(output/f'paired_{metric}_by_seed.csv')
        for name, values in deltas.items():
            pairs.append(dict(metric=metric, comparison=name, mean=values.mean(), sample_std=values.std(ddof=1), n=len(values)))
    paired = pd.DataFrame(pairs)
    paired.to_csv(output/'paired_summary.csv', index=False)
    geometry.groupby(['method', 'source', 'epoch', 'layer'])[
        ['cosA', 'relative_frobenius_A', 'cosG', 'relative_frobenius_G', 'fisher_cosine_proxy']
    ].agg(['mean', 'std']).to_csv(output/'geometry_summary.csv')

    def save(name):
        plt.tight_layout()
        plt.savefig(output/f'{name}.png', dpi=160)
        plt.close()

    plt.figure(figsize=(10, 5))
    for method, group in metrics.groupby('method'):
        curve = group.groupby('epoch').test_accuracy.agg(['mean', 'std'])
        plt.plot(curve.index, curve['mean'], marker='o', label=method)
    plt.xlabel('Epoch'); plt.ylabel('Test accuracy'); plt.legend(fontsize=7)
    save('accuracy_curves')
    plt.figure(figsize=(7, 4))
    for index, kind in enumerate(('a', 'full')):
        means = [summary[summary.method.eq(f'{kind}_public_{s}')].final_accuracy.mean() for s in ('fashion', 'cifar10')]
        plt.bar(np.arange(2)+index*.35, means, width=.35, label=kind)
    plt.xticks(np.arange(2)+.175, ['Fashion (matched)', 'CIFAR-10 (mismatched)'])
    plt.ylabel('Final accuracy'); plt.legend(); save('matched_mismatched_final_accuracy')
    penalties = paired[(paired.metric == 'final_accuracy') & paired.comparison.isin(['Penalty_A', 'Penalty_Full', 'Interaction'])]
    plt.figure(figsize=(7, 4)); plt.bar(penalties.comparison, penalties['mean'])
    for name, values in contrasts(summary.pivot(index='seed', columns='method', values='final_accuracy')).items():
        if name in penalties.comparison.tolist():
            plt.scatter([name]*len(values), values, color='black', s=15)
    plt.axhline(0, color='gray'); plt.ylabel('Accuracy difference'); save('mismatch_penalty')
    for factor in ('A', 'G'):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for col, layer in enumerate(geometry.layer.unique()):
            for method, group in geometry[geometry.layer.eq(layer)].groupby('method'):
                if factor == 'G' and not method.startswith('full_'):
                    continue
                for row, metric in enumerate((f'cos{factor}', f'relative_frobenius_{factor}')):
                    curve = group.groupby('epoch')[metric].mean()
                    axes[row, col].plot(curve.index, curve, marker='o', label=method)
                    axes[row, col].set_title(f'{layer}: {metric}')
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(RESEARCH_ONLY); save(f'{factor}_alignment')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ('clip_fraction', 'transformed_norm_mean', 'transformed_norm_p90', 'transformed_norm_p99')):
        for method, group in metrics.groupby('method'):
            curve = group.groupby('epoch')[metric].mean()
            ax.plot(curve.index, curve, marker='o', label=method)
        ax.set_title(metric)
    axes[0].legend(fontsize=6); fig.suptitle(RESEARCH_ONLY); save('clipping_norm_tail')
    report = ['# Exp23 '+('SMOKE — pipeline verification only' if smoke else 'descriptive results'),
              'No strong statistical significance claims; sample std uses ddof=1. Accuracy is a fraction.',
              RESEARCH_ONLY,
              'AUC: trapezoid over observed epochs 1..5; one-epoch smoke AUC is zero.',
              'Timing includes cold-start work; oracle time is separate. DP-SGD has no calibration.',
              '\nFinal accuracy paired comparisons:\n',
              paired[paired.metric.eq('final_accuracy')].to_string(index=False),
              '\nSmoke results must not be used for formal conclusions.' if smoke else '\nOnly three paired seeds: descriptive statistics.']
    (output/'report.md').write_text('\n\n'.join(report)+'\n')
    print(f'Validated {len(summary)} runs; analysis written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    analyze(parser.parse_args().smoke)

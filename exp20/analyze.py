"""Seed-aligned descriptive analysis with paired bootstrap intervals."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ['MPLCONFIGDIR'] = str(HERE/'.cache/matplotlib')
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp20.config import POWERS, SEEDS, POWER_ORDER, EPOCHS, FORMAL_RUN_COUNT

METRICS = ('final_accuracy', 'best_accuracy', 'accuracy_auc', 'clip_fraction', 'norm_p99',
           'scale_match', 'transformed_condition_proxy')
PAIRED = ('final_accuracy', 'accuracy_auc', 'clip_fraction', 'norm_p99')
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20020


def bootstrap(deltas):
    """Resample paired seed differences, never independent treatment samples."""
    values = np.asarray(deltas, dtype=float)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return np.percentile(values[indices].mean(axis=1), [2.5, 97.5])


def paired_analysis(summary, seeds):
    reference = summary[summary.p.eq(.5)].set_index('seed').loc[list(seeds)]
    rows = []
    for p in POWERS:
        group = summary[summary.p.eq(p)].set_index('seed').loc[list(seeds)]
        for metric in PAIRED:
            delta = (group[metric]-reference[metric]).to_numpy()
            lo, hi = bootstrap(delta)
            rows.append(dict(p=p, reference_p=.5, metric=metric, paired_mean_delta=delta.mean(),
                sample_std=np.std(delta, ddof=1) if len(seeds)>1 else np.nan,
                ci95_low=lo, ci95_high=hi,
                **{f'delta_seed_{seed}': value for seed, value in zip(seeds, delta)}))
    return pd.DataFrame(rows)


def analyze(output, smoke):
    seeds = (42,) if smoke else SEEDS
    frames, layers, configs = [], [], []
    for seed in seeds:
        for p in POWERS:
            directory = output/'runs'/f'{p}_{seed}'
            frame = pd.read_csv(directory/'metrics.csv')
            assert frame.epoch.tolist() == list(range(1, (1 if smoke else EPOCHS)+1))
            assert frame.p.eq(p).all() and frame.seed.eq(seed).all()
            np.testing.assert_allclose(frame.scale_match**2*frame.m_p, frame.m_reference, rtol=1e-12)
            assert frame.builder_vjp_calls.eq(0).all() and frame.builder_reverse_vectors.eq(0).all()
            assert frame.batches.eq(1 if smoke else 234).all()
            config = json.loads((directory/'config.json').read_text())
            assert config['smoke'] == smoke and config['seed'] == seed and config['p'] == p
            assert config['SEEDS'] == list(SEEDS) and config['epochs'] == (1 if smoke else EPOCHS)
            assert config['POWER_ORDER'] == {str(s): list(POWER_ORDER[s]) for s in SEEDS}
            layer = pd.read_csv(directory/'layer_norm_diagnostics.csv')
            assert layer.p.eq(p).all() and layer.seed.eq(seed).all()
            assert set(layer.epoch) == set(frame.epoch)
            frames.append(frame); layers.append(layer); configs.append(config)
    assert len(frames) == (8 if smoke else FORMAL_RUN_COUNT)
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for (p, seed), g in df.groupby(['p', 'seed'], sort=True):
        g = g.sort_values('epoch')
        rows.append(dict(p=p, seed=seed, final_accuracy=g.test_accuracy.iloc[-1],
            best_accuracy=g.test_accuracy.max(), accuracy_auc=np.trapezoid(g.test_accuracy, g.epoch),
            clip_fraction=g.clip_fraction.mean(), norm_p99=g.transformed_norm_p99.mean(),
            scale_match=g.scale_match.mean(), transformed_condition_proxy=g.transformed_condition_proxy.mean()))
    summary = pd.DataFrame(rows)
    power = summary.groupby('p')[list(METRICS)].agg(['mean', 'std'])
    power.columns = [f'{metric}_{"sample_std" if stat=="std" else stat}' for metric, stat in power.columns]
    power = power.reset_index()
    paired = paired_analysis(summary, seeds)
    df.to_csv(output/'metrics.csv', index=False)
    pd.concat(layers, ignore_index=True).to_csv(output/'layer_norm_diagnostics.csv', index=False)
    spectral = [c for c in df if c.startswith(('operator_gain_', 'transformed_eig_')) or c in ('transformed_condition_proxy', 'm_p', 'm_reference', 'scale_match')]
    df[['p', 'seed', 'epoch']+spectral].to_csv(output/'spectrum_summary.csv', index=False)
    summary.to_csv(output/'summary.csv', index=False)
    power.to_csv(output/'power_summary.csv', index=False)
    paired.to_csv(output/'paired_summary.csv', index=False)
    (output/'config.json').write_text(json.dumps(dict(smoke=smoke, seeds=seeds,
        power_order={s: POWERS if smoke else POWER_ORDER[s] for s in seeds}, expected_runs=len(frames), runs=configs,
        accuracy_auc='Trapezoid integral over observed epochs 1..5; smoke one epoch = 0',
        epoch_reduction='clip fraction, norm p99, scale and spectral spread: arithmetic mean over epochs',
        comparison='Seed-aligned differences relative to p=0.5; sample std ddof=1',
        bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, unit='paired seed delta', interval='percentile 95%')), indent=2)+'\n')
    charts = [('final_accuracy_vs_p', 'p', 'final_accuracy'), ('accuracy_auc_vs_p', 'p', 'accuracy_auc'),
        ('clip_fraction_vs_p', 'p', 'clip_fraction'), ('norm_p99_vs_p', 'p', 'norm_p99'),
        ('accuracy_vs_clip_fraction', 'clip_fraction', 'final_accuracy'),
        ('accuracy_vs_norm_p99', 'norm_p99', 'final_accuracy'), ('scale_factor_vs_p', 'p', 'scale_match'),
        ('transformed_spectral_spread_vs_p', 'p', 'transformed_condition_proxy')]
    def save(name):
        plt.tight_layout(); plt.savefig(output/f'{name}.png', dpi=160); plt.close()
    for filename, x, y in charts:
        plt.figure(figsize=(7, 4))
        xx = power.p if x == 'p' else power[x+'_mean']
        plt.errorbar(xx, power[y+'_mean'], yerr=None if smoke else power[y+'_sample_std'],
                     xerr=None if smoke or x=='p' else power[x+'_sample_std'], fmt='o-', capsize=3)
        if x != 'p':
            for i, p in enumerate(power.p):
                plt.annotate(f'p={p:g}', (xx.iloc[i], power[y+'_mean'].iloc[i]))
        plt.xlabel(x); plt.ylabel(y); plt.title('Smoke only' if smoke else '5 seeds: mean ± sample std')
        save(filename)
    for p, g in df.groupby('p'):
        epoch = g.groupby('epoch').test_accuracy.agg(['mean', 'std'])
        plt.errorbar(epoch.index, epoch['mean'], yerr=None if smoke else epoch['std'], label=f'p={p:g}')
    plt.xlabel('Epoch'); plt.ylabel('Test accuracy'); plt.legend(); save('accuracy_vs_epoch_by_p')
    for metric, filename in [('final_accuracy', 'paired_accuracy_delta_vs_p'), ('accuracy_auc', 'paired_auc_delta_vs_p')]:
        g = paired[paired.metric.eq(metric)]
        plt.plot(g.p, g.paired_mean_delta, 'o')
        plt.vlines(g.p, g.ci95_low, g.ci95_high)
        plt.axhline(0, color='gray'); plt.xlabel('p'); plt.ylabel(f'{metric}: p − 0.5')
        plt.title('Paired mean and bootstrap 95% CI' if not smoke else 'Smoke only'); save(filename)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, PAIRED):
        for seed in seeds:
            g = summary[summary.seed.eq(seed)].set_index('p')
            ax.plot([0, 1], g.loc[[.375, .5], metric], 'o-', label=str(seed))
        ax.set_xticks([0, 1], ['p=.375', 'p=.5']); ax.set_ylabel(metric)
    axes.flat[0].legend(title='Paired seed'); save('p0375_vs_p05_by_seed')
    def best(metric, minimum=False):
        v = power[metric+'_mean']; return power.loc[v.eq(v.min() if minimum else v.max()), 'p'].tolist()
    lines = ['# Exp20 paired power sweep',
        'SMOKE ONLY: no formal utility conclusions.' if smoke else 'Five paired seeds; descriptive evidence only. Bootstrap intervals from 20,000 resamples of seed differences. Five seeds do not justify strong significance claims.',
        '## Mean performance and clipping geometry',
        f'Best final accuracy: {best("final_accuracy")}. Best accuracy AUC: {best("accuracy_auc")}.',
        f'Lowest clip fraction: {best("clip_fraction", True)}; lowest norm p99: {best("norm_p99", True)}. These are separate geometry criteria, not a single utility optimum.',
        power.to_csv(index=False), '## p=.375 − p=.5',
        'Individual seed deltas, paired means, sample standard deviations and bootstrap 95% intervals:',
        paired[paired.p.eq(.375)].to_csv(index=False),
        'Raw paired results:', summary[summary.p.isin([.375, .5])].to_csv(index=False)]
    for metric in ('final_accuracy', 'accuracy_auc'):
        g = power[power.p.isin([.375, .5])].set_index('p')
        lines.append(f'{metric} stability (smaller sample std): '+', '.join(f'p={p}: {g.loc[p, metric+"_sample_std"]:.6g}' for p in (.375, .5)))
        stds = g[metric+'_sample_std']
        if not smoke:
            lines.append(f'More stable {metric} by sample std: {stds[stds.eq(stds.min())].index.tolist()} (ties listed).')
        r = paired[paired.p.eq(.375)&paired.metric.eq(metric)].iloc[0]
        wins = sum(r[f'delta_seed_{s}']>0 for s in seeds)
        lines.append(f'p=.375 improves {metric} in {wins}/{len(seeds)} seeds; mean delta {r.paired_mean_delta:+.6g}, CI [{r.ci95_low:+.6g}, {r.ci95_high:+.6g}]. '+('Positive in every observed seed.' if wins==len(seeds) else 'Not consistently positive across observed seeds.'))
    lines += ['## Activation anisotropic geometry: p>0 versus p=0', 'Scale matching controls synthetic global RMS, not all private norms or optimization effects. p=0 is scalar identity, not ordinary DP-SGD.']
    base = summary[summary.p.eq(0)].set_index('seed').loc[list(seeds)]
    for p in POWERS[1:]:
        g = summary[summary.p.eq(p)].set_index('seed').loc[list(seeds)]
        delta = (g.final_accuracy-base.final_accuracy).to_numpy(); lo, hi = bootstrap(delta)
        lines.append(f'p={p}: final accuracy delta mean {delta.mean():+.6g}, CI [{lo:+.6g}, {hi:+.6g}]; positive in {(delta>0).sum()}/{len(seeds)} seeds. '+('Consistent observed support for geometry utility.' if (delta>0).all() else 'No uniformly positive geometry benefit across seeds.'))
    lines += ['## Above p=.5: norm tail ↑, clipping ↑, accuracy ↓']
    ref = summary[summary.p.eq(.5)].set_index('seed').loc[list(seeds)]
    for p in ( .625, .75, 1.):
        delta = summary[summary.p.eq(p)].set_index('seed').loc[list(seeds), list(PAIRED)]-ref[list(PAIRED)]
        joint = (delta.norm_p99>0)&(delta.clip_fraction>0)&(delta.final_accuracy<0)
        lines.append(f'p={p}: joint pattern in {joint.sum()}/{len(seeds)} seeds; mean deltas {delta.mean().to_dict()}. '+('Consistent observed pattern.' if joint.all() else 'Pattern is not consistent across all seeds.')+' This association does not establish causation.')
    region = all(p in (.375, .5) for p in best('final_accuracy')+best('accuracy_auc'))
    candidate = best('final_accuracy')[0]
    candidate_rows = summary[summary.p.eq(candidate)].set_index('seed').loc[list(seeds)]
    under_count = int((candidate_rows.final_accuracy > base.final_accuracy).sum())
    over_counts = []
    for p in (.625, .75, 1.):
        high = summary[summary.p.eq(p)].set_index('seed').loc[list(seeds)]
        over_counts.append(int(((high.final_accuracy < candidate_rows.final_accuracy) &
                               (high.norm_p99 > candidate_rows.norm_p99) &
                               (high.clip_fraction > candidate_rows.clip_fraction)).sum()))
    stable_shape = region and under_count == len(seeds) and all(n == len(seeds) for n in over_counts)
    lines += ['## Moderate optimum and under/over-conditioning',
        f'Both mean-utility maxima lie in p≈.375–.5: {region}. '+('This is a candidate optimum region within this sweep.' if region else 'The two mean-utility maxima do not jointly support that optimum region.'),
        f'Using mean-final-accuracy winner p={candidate} as the candidate sweet spot: improvement over p=0 in {under_count}/{len(seeds)} seeds; higher tail/clipping and lower accuracy at p=.625,.75,1 in {over_counts} seeds respectively. Consistent full pattern across observed seeds: {stable_shape}. This selection is descriptive and made after observing the sweep.',
        'The p=0 comparisons and above-.5 joint counts quantify the proposed under-conditioning → sweet spot → over-preconditioning pattern; mixed seed signs limit its stability. Mean maxima alone do not establish a universal optimum.',
        'Timing: algorithm = builder + private training; diagnostic CPU transfers and evaluation excluded; CUDA internal breakdown uses deferred Events. Runtime is secondary; small single-machine fluctuations do not establish power-specific speedups.',
        'Biases use augmented activations. Spectral quantiles weight each eigenvalue once; RMS moments additionally weight layer output dimension. RDP uses the unchanged shuffled fixed-batch convention; clipping diagnostics are unnoised research measurements.']
    (output/'report.md').write_text('\n\n'.join(lines)+'\n')
    print(f'Validated {len(frames)} runs / {len(df)} epochs; analysis saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(HERE/'results'/'smoke' if args.smoke else HERE/'results', args.smoke)

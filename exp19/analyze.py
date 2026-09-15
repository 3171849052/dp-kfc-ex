"""Paired descriptive analysis; smoke outputs never enter formal summaries."""
import sys
import os
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
from exp19.config import METHODS, SEEDS, EPOCHS

PAIRS = ((0, 1), (1, 2), (2, 3), (0, 3))


def bootstrap(values):
    rng = np.random.default_rng(19019)
    return np.quantile(rng.choice(values, (20000, len(values)), replace=True).mean(1), [.025, .975])


def analyze(output, smoke):
    frames, layers, configs = [], [], []
    for seed in ((42,) if smoke else SEEDS):
        for method in METHODS:
            directory = output/'runs'/f'{method}_{seed}'
            f = pd.read_csv(directory/'metrics.csv')
            assert f.epoch.tolist() == list(range(1, (1 if smoke else EPOCHS)+1))
            assert f.method.eq(method).all() and f.seed.eq(seed).all()
            frames.append(f)
            layers.append(pd.read_csv(directory/'layer_norm_diagnostics.csv'))
            configs.append(json.loads((directory/'config.json').read_text()))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output/'metrics.csv', index=False)
    pd.concat(layers, ignore_index=True).to_csv(output/'layer_norm_diagnostics.csv', index=False)
    (output/'config.json').write_text(json.dumps(dict(smoke=smoke, runs=configs,
        accuracy_auc='Trapezoid integral over observed epochs 1..5; one-epoch smoke = 0',
        bootstrap='20000 paired seed resamples, percentile 95% CI; sample std ddof=1',
        total_peak='maximum of build/train phase peaks; evaluation excluded'), indent=2)+'\n')
    keys = ['method', 'seed', 'epoch']
    for filename, columns in {
        'timing_breakdown': [c for c in df if c.endswith('_seconds') or c in ('seconds_per_batch', 'private_samples_per_second')],
        'memory_summary': [c for c in df if c.endswith('_bytes') or c == 'stored_scalar_count'],
        'builder_budget': [c for c in df if c.startswith('builder_') or c in ('curvature_backward_seconds', 'operator_state_bytes', 'stored_scalar_count')],
    }.items():
        df[keys+columns].to_csv(output/f'{filename}.csv', index=False)
    summary = []
    for (method, seed), g in df.groupby(['method', 'seed'], sort=False):
        g = g.sort_values('epoch')
        summary.append(dict(method=method, seed=seed, final_accuracy=g.test_accuracy.iloc[-1],
            best_accuracy=g.test_accuracy.max(), accuracy_auc=np.trapezoid(g.test_accuracy, g.epoch),
            algorithm_seconds=g.algorithm_epoch_seconds.sum(), private_seconds=g.private_train_seconds.sum(),
            build_seconds=g.preconditioner_build_seconds.sum(), curvature_seconds=g.curvature_backward_seconds.sum(),
            peak_allocated_bytes=g.total_peak_cuda_allocated_bytes.max(),
            peak_reserved_bytes=g.total_peak_cuda_reserved_bytes.max(),
            train_peak_bytes=g.train_peak_cuda_allocated_bytes.max(),
            clip_fraction=g.clip_fraction.mean(), norm_p99=g.transformed_norm_p99.mean()))
    s = pd.DataFrame(summary)
    s.to_csv(output/'summary.csv', index=False)
    ms = s.groupby('method').agg({c: ['mean', 'std'] for c in s if c not in ('method', 'seed')})
    ms.columns = [f'{c}_{"sample_std" if a == "std" else a}' for c, a in ms.columns]
    ms.to_csv(output/'method_summary.csv')
    paired = []
    for a, b in PAIRS:
        x, y = [s[s.method.eq(METHODS[i])].set_index('seed').sort_index() for i in (a,b)]
        assert x.index.equals(y.index)
        for metric in [c for c in s if c not in ('method', 'seed')]:
            delta = (y[metric]-x[metric]).to_numpy()
            lo, hi = bootstrap(delta) if not smoke else (np.nan, np.nan)
            record = dict(comparison=f'{METHODS[b]} - {METHODS[a]}', metric=metric,
                mean=delta.mean(), sample_std=pd.Series(delta).std(), ci95_low=lo, ci95_high=hi,
                n=len(delta))
            if metric == 'final_accuracy':
                record['paired_accuracy_delta'] = json.dumps(dict(zip(map(str, x.index), delta.tolist())))
            paired.append(record)
    paired = pd.DataFrame(paired)
    paired.to_csv(output/'paired_summary.csv', index=False)

    def finish(name, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output/f'{name}.png', dpi=160)
        plt.close()

    for name, columns in [('accuracy_vs_epoch', ['test_accuracy']),
                          ('clipping_vs_epoch', ['clip_fraction', 'mean_clip_factor']),
                          ('norm_quantiles', ['transformed_norm_p50', 'transformed_norm_p90', 'transformed_norm_p99'])]:
        plt.figure(figsize=(9, 5))
        for method, g in df.groupby('method'):
            means = g.groupby('epoch')[columns].mean()
            for col in columns:
                plt.plot(means.index, means[col], marker='o', label=f'{method}: {col}')
        finish(name, 'Epoch', 'Accuracy' if name.startswith('accuracy') else 'Value')
    plt.figure(figsize=(8, 5))
    for method, g in df.groupby('method'):
        g = g.sort_values(['seed', 'epoch']).copy()
        g['cumulative_seconds'] = g.groupby('seed').algorithm_epoch_seconds.cumsum()
        means = g.groupby('epoch')[['cumulative_seconds', 'test_accuracy']].mean()
        plt.plot(means.cumulative_seconds, means.test_accuracy, marker='o', label=method)
    finish('accuracy_vs_algorithm_time', 'Mean cumulative algorithm seconds', 'Mean accuracy')
    for name, columns, divisor in [
        ('timing_breakdown', ['preconditioner_build_seconds', 'private_train_seconds', 'evaluation_seconds'], 1),
        ('builder_breakdown', ['synthetic_generation_seconds', 'activation_forward_seconds', 'curvature_backward_seconds', 'factor_accumulation_seconds', 'matrix_function_seconds'], 1),
        ('cuda_memory_breakdown', ['build_peak_cuda_allocated_bytes', 'build_peak_cuda_reserved_bytes', 'train_peak_cuda_allocated_bytes', 'train_peak_cuda_reserved_bytes'], 2**20)]:
        (df.groupby('method')[columns].mean()/divisor).plot.bar(figsize=(10, 6), rot=15)
        finish(name, 'Method', 'MiB (mean epoch peak)' if divisor > 1 else 'Mean seconds per epoch')
    for name, col, divisor in [('accuracy_vs_peak_memory', 'peak_allocated_bytes', 2**20), ('accuracy_vs_runtime', 'algorithm_seconds', 1)]:
        plt.figure(figsize=(8, 5))
        for method, g in s.groupby('method'):
            plt.scatter(g[col]/divisor, g.final_accuracy, label=method)
        finish(name, 'Peak allocated MiB' if divisor > 1 else 'Total algorithm seconds', 'Final accuracy')
    plt.figure(figsize=(10, 5))
    for i, (a,b) in enumerate(PAIRS):
        x,y = [s[s.method.eq(METHODS[j])].set_index('seed').final_accuracy for j in (a,b)]
        plt.scatter(np.full(len(x), i), (y-x)*100, label=f'M{b} - M{a}')
    plt.axhline(0, color='gray', linewidth=1)
    finish('paired_accuracy_delta', 'Attribution comparison', 'Paired final accuracy delta (percentage points)')

    means = s.groupby('method').mean(numeric_only=True)
    lines = ['# Exp19 results', '', 'SMOKE ONLY: one zero-noise batch, one seed. No formal utility or performance conclusion.' if smoke else
        'Five paired seeds; descriptive estimates and paired percentile bootstrap intervals. No strong significance claims.', '',
        'All clipping and layerwise norm statistics are unnoised research diagnostics, not a DP release.', '',
        '## Attribution chain', '']
    effects = ['Ghost backend', 'Remove C (including its scaling effect)', 'Power .5 → .25 with self-contained A RMS matching', 'Headline M3 versus M0']
    for (a,b), effect in zip(PAIRS, effects):
        row = paired[(paired.comparison == f'{METHODS[b]} - {METHODS[a]}') & (paired.metric == 'final_accuracy')].iloc[0]
        x, y = means.loc[METHODS[a]], means.loc[METHODS[b]]
        lines += [f'### M{b} − M{a}: {effect}', '',
            f'Final accuracy: {x.final_accuracy:.6f} → {y.final_accuracy:.6f}; paired delta {row["mean"]*100:+.4f} pp; '
            f'95% CI [{row.ci95_low*100:.4f}, {row.ci95_high*100:.4f}] pp.',
            f'Algorithm runtime: {x.algorithm_seconds:.3f} → {y.algorithm_seconds:.3f} s ({(1-y.algorithm_seconds/x.algorithm_seconds)*100:+.2f}% reduction). '
            f'Private training: {x.private_seconds:.3f} → {y.private_seconds:.3f} s ({(1-y.private_seconds/x.private_seconds)*100:+.2f}% reduction).',
            f'Peak allocated memory: {x.peak_allocated_bytes/2**20:.2f} → {y.peak_allocated_bytes/2**20:.2f} MiB; '
            f'train peak: {x.train_peak_bytes/2**20:.2f} → {y.train_peak_bytes/2**20:.2f} MiB ({(1-y.train_peak_bytes/x.train_peak_bytes)*100:+.2f}% reduction).',
            f'Mean epoch clipping fraction: {x.clip_fraction:.6f} → {y.clip_fraction:.6f}; mean epoch norm p99: {x.norm_p99:.6f} → {y.norm_p99:.6f}.', '']
    lines += ['## Forward-only curvature budget', '']
    for method in METHODS:
        g = df[df.method.eq(method)]
        lines.append(f'- {method}: per-epoch forward/VJP/reverse-vector/sample counts '
            f'{g.builder_forward_calls.iloc[0]}/{g.builder_vjp_calls.iloc[0]}/{g.builder_reverse_vectors.iloc[0]}/{g.builder_samples.iloc[0]}; '
            f'mean backward {g.curvature_backward_seconds.mean():.6f}s; build {g.preconditioner_build_seconds.mean():.6f}s.')
    full = df[df.method.eq(METHODS[1])]
    for method in METHODS[2:]:
        g = df[df.method.eq(method)]
        lines.append(f'- {method} versus M1: build time reduction '
            f'{(1-g.preconditioner_build_seconds.mean()/full.preconditioner_build_seconds.mean())*100:+.2f}%; '
            f'curvature backward saved {full.curvature_backward_seconds.mean()-g.curvature_backward_seconds.mean():.6f} s/epoch (100%); '
            f'{full.builder_reverse_vectors.iloc[0]} reverse vectors eliminated per epoch.')
    lines += ['', 'M2/M3 eliminate all synthetic reverse vectors and curvature backward calls. '
              'Actual build savings are reported above and include A accumulation/eigendecomposition.', '',
              'Accuracy AUC integrates observed epoch 1–5 accuracies. Memory peaks exclude evaluation; reserved memory includes allocator caching within each fresh process. '
              'Private timing includes the common research diagnostics. Internal Exact/Ghost phases are explanatory, not cross-backend rankings. '
              'M3−M2 identifies power plus the required RMS matching jointly. Identity C is undamped identity. '
              'RDP accounting follows the repository shuffled fixed-batch convention; research CSVs are not privacy-protected releases.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Analysis saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(HERE/'results'/'smoke' if args.smoke else HERE/'results', args.smoke)

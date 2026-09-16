"""Summarize measured phases and paired speedups; accuracy is only a sanity check."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def analyze(smoke=False, output=None):
    root = Path(output).resolve() if output else HERE/'results'
    source = root/'smoke' if smoke and output is None else root
    paths = sorted((source/'runs').glob('*/metrics.csv'))
    if not paths:
        raise RuntimeError(f'No measurements in {source}')
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    errors_path = root/'correctness.json'
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else {}
    frame['gradient_max_abs_error'] = frame.method.map({k: v['gradient_max_abs_error'] for k, v in errors.items()})
    frame['gradient_relative_l2_error'] = frame.method.map({k: v['gradient_relative_l2_error'] for k, v in errors.items()})
    frame.to_csv(root/'summary.csv', index=False)
    metrics = ['seconds_per_batch', 'private_samples_per_second', 'peak_cuda_allocated_bytes',
        'peak_cuda_reserved_bytes', 'gradient_max_abs_error', 'gradient_relative_l2_error',
        'first_pass_seconds', 'norm_seconds', 'second_pass_seconds', 'bk_reconstruction_seconds',
        'aggregate_transform_seconds', 'noise_step_seconds', 'bk_cache_bytes', 'temporary_per_sample_grad_bytes']
    summary = frame.groupby('method')[metrics].mean()
    summary.to_csv(root/'method_summary.csv')
    lines = ['# Exp21 '+('smoke' if smoke else 'formal')+' report', '',
        'MNIST / SimpleCNN, A-only power=0.25, damping=1e-3. Accuracy is a sanity check only.', '',
        'Smoke uses seed 42, 3 private batches of 256, 1 synthetic batch, 1 epoch, noise=0.' if smoke else
        'Formal protocol: five paired seeds, five epochs, epsilon=1, delta=1e-5.', '',
        '| method | s/batch | samples/s | allocated MiB | reserved MiB | gradient relative L2 error |',
        '|---|---:|---:|---:|---:|---:|']
    for method, row in summary.iterrows():
        lines.append(f'| {method} | {row.seconds_per_batch:.6f} | {row.private_samples_per_second:.1f} | {row.peak_cuda_allocated_bytes/2**20:.1f} | {row.peak_cuda_reserved_bytes/2**20:.1f} | {row.gradient_relative_l2_error:.3g} |')
    lines += ['', '## Paired speedups', '', 'Ratios are reference time / candidate time, averaged over aligned seed/epoch pairs.', '']
    for candidate, reference in [('bk','ghost2'), ('bk_gd','bk'), ('bk_gd','exact'), ('bk_gd','fast2')]:
        a = frame[frame.method.eq(candidate)].set_index(['seed','epoch']).seconds_per_batch
        b = frame[frame.method.eq(reference)].set_index(['seed','epoch']).seconds_per_batch
        if not a.index.equals(b.index):
            raise RuntimeError('Unpaired measurement rows')
        lines.append(f'- {candidate} vs {reference}: {(b/a).mean():.3f}x')
    lines += ['', '## GD profiling', '',
        'CUDA event phases below are milliseconds per batch; wall time also includes data loading/transfer and Python work outside the event phases.', '',
        '| method | first pass ms | norm ms | reconstruction ms | wall ms | cache MiB | first-pass parameter grad count |',
        '|---|---:|---:|---:|---:|---:|---:|']
    per_batch = frame.copy()
    phases = ['first_pass_seconds', 'norm_seconds', 'bk_reconstruction_seconds',
              'second_pass_seconds', 'aggregate_transform_seconds', 'noise_step_seconds']
    per_batch[phases] = per_batch[phases].div(per_batch.batches, axis=0)
    grouped = per_batch.groupby('method').mean(numeric_only=True)
    for method, row in grouped.iterrows():
        lines.append(f'| {method} | {row.first_pass_seconds*1000:.3f} | {row.norm_seconds*1000:.3f} | {row.bk_reconstruction_seconds*1000:.3f} | {row.seconds_per_batch*1000:.3f} | {row.bk_cache_bytes/2**20:.3f} | {row.first_pass_parameter_grad_count:.0f} |')
    bk, gd = grouped.loc['bk'], grouped.loc['bk_gd']
    first_change = gd.first_pass_seconds/bk.first_pass_seconds-1
    total_change = gd.seconds_per_batch/bk.seconds_per_batch-1
    lines += ['', f'BK+GD vs BK: first-pass change {first_change:+.1%}; total wall time change {total_change:+.1%}.',
        f'Absolute phase differences (GD minus BK): first pass {(gd.first_pass_seconds-bk.first_pass_seconds)*1000:+.3f} ms/batch, norm {(gd.norm_seconds-bk.norm_seconds)*1000:+.3f}, reconstruction {(gd.bk_reconstruction_seconds-bk.bk_reconstruction_seconds)*1000:+.3f}.',
        f'Wall time outside the sum of measured CUDA phases: BK {(bk.seconds_per_batch-bk[phases].sum())*1000:.3f} ms/batch; GD {(gd.seconds_per_batch-gd[phases].sum())*1000:.3f} ms/batch. This residual includes unprofiled work and host/launch gaps, not a pure CPU-time measurement.',
        'The timing regions are identical across BK and GD. A first-pass improvement does not imply an end-to-end improvement in a three-batch diagnostic run.', '']
    if total_change > 0:
        event_delta = gd[phases].sum()-bk[phases].sum()
        wall_delta = gd.seconds_per_batch-bk.seconds_per_batch
        lines += [f'In this run GD is slower by {wall_delta*1000:.3f} ms/batch: {event_delta*1000:+.3f} ms is the change in the sum of CUDA phases and {(wall_delta-event_delta)*1000:+.3f} ms is the residual change. These measurements localize the difference by phase, but do not establish a causal CPU/GPU scheduling explanation.', '']
    gd_rows = frame[frame.method.eq('bk_gd')]
    assert (gd_rows.first_pass_parameter_grad_count == 0).all()
    assert (gd_rows.backward_calls == 1).all() and (~gd_rows.input_gradient_computed).all()
    assert gd.bk_cache_bytes <= bk.bk_cache_bytes
    lines += ['', '## Correctness and memory', '',
        'Gradient errors above come from a separate real MNIST batch of 256 against naive sample backward; they are not timing-path estimates. FP32 convolution tolerances: rtol=5e-4, atol=3e-5 for gradients.', '']
    if smoke:
        ref = torch.load(source/'runs/exact_42/smoke_state.pt', weights_only=True)
        comparison = {}
        for method in frame.method.unique():
            state = torch.load(source/f'runs/{method}_42/smoke_state.pt', weights_only=True)
            diff = max((state['model'][n]-v).abs().max().item() for n, v in ref['model'].items())
            for n, v in ref['model'].items():
                torch.testing.assert_close(state['model'][n], v, rtol=5e-4, atol=3e-5)
            torch.testing.assert_close(state['norms'], ref['norms'], rtol=5e-4, atol=1e-4)
            torch.testing.assert_close(state['factors'], ref['factors'], rtol=5e-4, atol=1e-5)
            comparison[method] = dict(final_parameter_max_abs_error=diff)
            lines.append(f'- {method}: after 3 SGD steps, max parameter error vs exact = {diff:.3g}.')
        (source/'state_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    for _, row in frame.iterrows():
        if row.method in ('bk', 'bk_gd'):
            assert row.second_pass_seconds == 0 and row.fallback_layer_count == 0 and not row.requires_second_backward
        assert row.cache_empty_after_step
        allocations = json.loads(row.batch_end_allocated_bytes)
        # Saved norm/factor/loss diagnostics grow linearly, by ~2 KiB per batch.
        lines.append(f'- {row.method}, seed {row.seed}, epoch {row.epoch}: batch-end allocated MiB = '+', '.join(f'{v/2**20:.3f}' for v in allocations)+f'; strategy {row.layer_strategies}.')
    lines += ['', '## Transformer integration', '']
    for path in sorted((root/'integration').glob('*.json')):
        result = json.loads(path.read_text())
        label = path.stem
        lines.append(f'- {label}: fallback layers={result["fallback_layer_count"]}, backward calls={result["backward_calls"]}, second backward={result["requires_second_backward"]}, anchor={result["gd_anchor_module"]}.')
    lines += ['', 'All recorded BK caches are empty after each step. Reserved memory includes CUDA allocator caching and disposable warmup; allocated memory is the training-phase peak. BK cache bytes count actual distinct retained tensor storage (views deduplicated); temporary-gradient bytes count gradient payloads, not allocator peak.', '',
        'Small smoke measurements are diagnostic and do not establish formal speed or utility claims. No formal jobs were started by the smoke workflow.', '']
    (root/'report.md').write_text('\n'.join(lines))
    print(summary[['seconds_per_batch','private_samples_per_second','peak_cuda_allocated_bytes']].to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.smoke, args.output)

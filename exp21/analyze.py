"""Summarize measured phases and paired speedups; accuracy is only a sanity check."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def analyze(smoke=False):
    root = HERE/'results'
    source = root/'smoke' if smoke else root
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
    lines += ['', 'All recorded BK caches are empty after each step. Reserved memory includes CUDA allocator caching and disposable warmup; allocated memory is the training-phase peak. Cache/temporary-gradient byte fields count tensor payloads (views may overlap), not allocator peak.', '',
        'Small smoke measurements are diagnostic and do not establish formal speed or utility claims. No formal jobs were started by the smoke workflow.', '']
    (root/'report.md').write_text('\n'.join(lines))
    print(summary[['seconds_per_batch','private_samples_per_second','peak_cuda_allocated_bytes']].to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    analyze(parser.parse_args().smoke)

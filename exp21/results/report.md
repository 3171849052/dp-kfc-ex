# Exp21 smoke report

MNIST / SimpleCNN, A-only power=0.25, damping=1e-3. Accuracy is a sanity check only.

Smoke uses seed 42, 3 private batches of 256, 1 synthetic batch, 1 epoch, noise=0.

| method | s/batch | samples/s | allocated MiB | reserved MiB | gradient relative L2 error |
|---|---:|---:|---:|---:|---:|
| bk | 0.037177 | 6886.0 | 128.6 | 228.0 | 3.82e-07 |
| bk_gd | 0.053130 | 4818.3 | 128.6 | 228.0 | 3.82e-07 |
| exact | 0.055875 | 4581.6 | 810.5 | 1214.0 | 4.33e-07 |
| fast2 | 0.060099 | 4259.7 | 668.7 | 820.0 | 7.31e-07 |
| ghost2 | 0.054700 | 4680.0 | 128.6 | 230.0 | 7.32e-07 |

## Paired speedups

Ratios are reference time / candidate time, averaged over aligned seed/epoch pairs.

- bk vs ghost2: 1.471x
- bk_gd vs bk: 0.700x
- bk_gd vs exact: 1.052x
- bk_gd vs fast2: 1.131x

## Correctness and memory

Gradient errors above come from a separate real MNIST batch of 256 against naive sample backward; they are not timing-path estimates. FP32 convolution tolerances: rtol=5e-4, atol=3e-5 for gradients.

- bk: after 3 SGD steps, max parameter error vs exact = 7.45e-09.
- bk_gd: after 3 SGD steps, max parameter error vs exact = 7.45e-09.
- exact: after 3 SGD steps, max parameter error vs exact = 0.
- fast2: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- ghost2: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- bk, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "fast", "conv1": "fast"}.
- bk_gd, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "fast", "conv1": "fast"}.
- exact, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {}.
- fast2, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "fast", "fc1": "fast", "conv2": "fast", "conv1": "fast"}.
- ghost2, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "ghost", "conv1": "ghost"}.

All recorded BK caches are empty after each step. Reserved memory includes CUDA allocator caching and disposable warmup; allocated memory is the training-phase peak. Cache/temporary-gradient byte fields count tensor payloads (views may overlap), not allocator peak.

Small smoke measurements are diagnostic and do not establish formal speed or utility claims. No formal jobs were started by the smoke workflow.

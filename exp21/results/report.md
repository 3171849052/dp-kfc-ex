# Exp21 smoke report

MNIST / SimpleCNN, A-only power=0.25, damping=1e-3. Accuracy is a sanity check only.

Smoke uses seed 42, 3 private batches of 256, 1 synthetic batch, 1 epoch, noise=0.

| method | s/batch | samples/s | allocated MiB | reserved MiB | gradient relative L2 error |
|---|---:|---:|---:|---:|---:|
| bk | 0.037673 | 6795.3 | 127.0 | 224.0 | 3.82e-07 |
| bk_gd | 0.037384 | 6847.9 | 138.5 | 222.0 | 3.82e-07 |
| exact | 0.039043 | 6557.0 | 810.6 | 1208.0 | 4.32e-07 |
| fast2 | 0.056932 | 4496.6 | 663.7 | 814.0 | 7.31e-07 |
| ghost2 | 0.082224 | 3113.4 | 127.0 | 224.0 | 7.32e-07 |

## Paired speedups

Ratios are reference time / candidate time, averaged over aligned seed/epoch pairs.

- bk vs ghost2: 2.183x
- bk_gd vs bk: 1.008x
- bk_gd vs exact: 1.044x
- bk_gd vs fast2: 1.523x

## GD profiling

CUDA event phases below are milliseconds per batch; wall time also includes data loading/transfer and Python work outside the event phases.

| method | first pass ms | norm ms | reconstruction ms | wall ms | cache MiB | first-pass parameter grad count |
|---|---:|---:|---:|---:|---:|---:|
| bk | 1.834 | 4.359 | 0.364 | 37.673 | 55.578 | 8 |
| bk_gd | 1.013 | 4.821 | 0.375 | 37.384 | 55.578 | 0 |
| exact | 3.729 | 10.036 | 0.000 | 39.043 | 0.000 | 8 |
| fast2 | 2.408 | 6.073 | 0.000 | 56.932 | 55.578 | 8 |
| ghost2 | 2.425 | 30.311 | 0.000 | 82.224 | 55.578 | 8 |

BK+GD vs BK: first-pass change -44.8%; total wall time change -0.8%.
Absolute phase differences (GD minus BK): first pass -0.821 ms/batch, norm +0.462, reconstruction +0.012.
Wall time outside the sum of measured CUDA phases: BK 28.564 ms/batch; GD 28.734 ms/batch. This residual includes unprofiled work and host/launch gaps, not a pure CPU-time measurement.
The timing regions are identical across BK and GD. A first-pass improvement does not imply an end-to-end improvement in a three-batch diagnostic run.


## Correctness and memory

Gradient errors above come from a separate real MNIST batch of 256 against naive sample backward; they are not timing-path estimates. FP32 convolution tolerances: rtol=5e-4, atol=3e-5 for gradients.

- bk: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- bk_gd: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- exact: after 3 SGD steps, max parameter error vs exact = 0.
- fast2: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- ghost2: after 3 SGD steps, max parameter error vs exact = 7.45e-09.
- bk, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "fast", "conv1": "fast"}.
- bk_gd, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "fast", "conv1": "fast"}.
- exact, seed 42, epoch 1: batch-end allocated MiB = 20.021, 20.023, 20.026; strategy {}.
- fast2, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "fast", "fc1": "fast", "conv2": "fast", "conv1": "fast"}.
- ghost2, seed 42, epoch 1: batch-end allocated MiB = 20.017, 20.020, 20.022; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "ghost", "conv1": "ghost"}.

## Transformer integration

- bert_bk: fallback layers=0, backward calls=1, second backward=False, anchor=None.
- bert_bk_gd: fallback layers=0, backward calls=1, second backward=False, anchor=bert.embeddings.word_embeddings.
- gpt2_bk: fallback layers=0, backward calls=1, second backward=False, anchor=None.
- gpt2_bk_gd: fallback layers=0, backward calls=1, second backward=False, anchor=transformer.wte.
- llama_bk: fallback layers=0, backward calls=1, second backward=False, anchor=None.
- llama_bk_gd: fallback layers=0, backward calls=1, second backward=False, anchor=model.embed_tokens.
- tinyvit_bk_gd_identity: fallback layers=2, backward calls=2, second backward=True, anchor=None.
- tinyvit_bk_gd_partial: fallback layers=2, backward calls=2, second backward=True, anchor=None.
- tinyvit_bk_identity: fallback layers=2, backward calls=2, second backward=True, anchor=None.
- tinyvit_bk_partial: fallback layers=2, backward calls=2, second backward=True, anchor=None.
- tinyvit_exact_identity: fallback layers=2, backward calls=2, second backward=True, anchor=None.
- tinyvit_exact_partial: fallback layers=2, backward calls=2, second backward=True, anchor=None.

All recorded BK caches are empty after each step. Reserved memory includes CUDA allocator caching and disposable warmup; allocated memory is the training-phase peak. BK cache bytes count actual distinct retained tensor storage (views deduplicated); temporary-gradient bytes count gradient payloads, not allocator peak.

Small smoke measurements are diagnostic and do not establish formal speed or utility claims. No formal jobs were started by the smoke workflow.

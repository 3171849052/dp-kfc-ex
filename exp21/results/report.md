# Exp21 smoke report

MNIST / SimpleCNN, A-only power=0.25, damping=1e-3. Accuracy is a sanity check only.

Smoke uses seed 42, 3 private batches of 256, 1 synthetic batch, 1 epoch, noise=0.

| method | s/batch | samples/s | allocated MiB | reserved MiB | gradient relative L2 error |
|---|---:|---:|---:|---:|---:|
| bk | 0.056094 | 4563.8 | 127.0 | 224.0 | 3.82e-07 |
| bk_gd | 0.054985 | 4655.8 | 126.2 | 222.0 | 3.82e-07 |
| exact | 0.056666 | 4517.7 | 810.6 | 1208.0 | 4.32e-07 |
| fast2 | 0.057488 | 4453.1 | 259.9 | 422.0 | 7.31e-07 |
| ghost2 | 0.060041 | 4263.7 | 150.4 | 274.0 | 7.31e-07 |

## Paired speedups

Ratios are reference time / candidate time, averaged over aligned seed/epoch pairs.

- bk vs ghost2: 1.070x
- bk_gd vs bk: 1.020x
- bk_gd vs exact: 1.031x
- bk_gd vs fast2: 1.046x

## GD profiling

CUDA event phases below are milliseconds per batch; wall time also includes data loading/transfer and Python work outside the event phases.

| method | first pass ms | norm ms | reconstruction ms | wall ms | cache MiB | first-pass parameter grad count |
|---|---:|---:|---:|---:|---:|---:|
| bk | 2.600 | 4.895 | 0.773 | 56.094 | 55.578 | 8 |
| bk_gd | 1.685 | 5.294 | 0.769 | 54.985 | 55.578 | 0 |
| exact | 4.893 | 10.200 | 0.000 | 56.666 | 0.000 | 8 |
| fast2 | 7.622 | 0.109 | 0.000 | 57.488 | 0.000 | 0 |
| ghost2 | 13.130 | 0.009 | 0.000 | 60.041 | 0.000 | 0 |

BK+GD vs BK: first-pass change -35.2%; total wall time change -2.0%.
Absolute phase differences (GD minus BK): first pass -0.915 ms/batch, norm +0.399, reconstruction -0.004.
Wall time outside the sum of measured CUDA phases: BK 42.583 ms/batch; GD 42.101 ms/batch. This residual includes unprofiled work and host/launch gaps, not a pure CPU-time measurement.
The timing regions are identical across BK and GD. A first-pass improvement does not imply an end-to-end improvement in a three-batch diagnostic run.


## Correctness and memory

Gradient errors above come from a separate real MNIST batch of 256 against naive sample backward; they are not timing-path estimates. FP32 convolution tolerances: rtol=5e-4, atol=3e-5 for gradients.

- bk: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- bk_gd: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- exact: after 3 SGD steps, max parameter error vs exact = 0.
- fast2: after 3 SGD steps, max parameter error vs exact = 1.49e-08.
- ghost2: after 3 SGD steps, max parameter error vs exact = 7.45e-09.
- bk, seed 42, epoch 1: batch-end allocated MiB = 20.018, 20.021, 20.023; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "fast", "conv1": "fast"}.
- bk_gd, seed 42, epoch 1: batch-end allocated MiB = 20.018, 20.021, 20.023; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "fast", "conv1": "fast"}.
- exact, seed 42, epoch 1: batch-end allocated MiB = 20.021, 20.023, 20.026; strategy {}.
- fast2, seed 42, epoch 1: batch-end allocated MiB = 20.018, 20.021, 20.023; strategy {"fc2": "fast", "fc1": "fast", "conv2": "fast", "conv1": "fast"}.
- ghost2, seed 42, epoch 1: batch-end allocated MiB = 20.018, 20.021, 20.023; strategy {"fc2": "ghost", "fc1": "ghost", "conv2": "ghost", "conv1": "ghost"}.

## Transformer integration

- bert_bk: fallback layers=0, backward calls=1, second backward=False, anchor=None.
- bert_bk_gd: fallback layers=0, backward calls=1, second backward=False, anchor=bert.embeddings.word_embeddings.
- gpt2_bk: fallback layers=0, backward calls=1, second backward=False, anchor=None.
- gpt2_bk_gd: fallback layers=0, backward calls=1, second backward=False, anchor=transformer.wte.
- llama_bk: fallback layers=0, backward calls=1, second backward=False, anchor=None.
- llama_bk_gd: fallback layers=0, backward calls=1, second backward=False, anchor=model.embed_tokens.
- tinyvit_bk_gd_identity: fallback layers=2, backward calls=3, second backward=True, anchor=patch_embed.
- tinyvit_bk_gd_partial: fallback layers=2, backward calls=3, second backward=True, anchor=patch_embed.
- tinyvit_bk_identity: fallback layers=2, backward calls=3, second backward=True, anchor=patch_embed.
- tinyvit_bk_partial: fallback layers=2, backward calls=3, second backward=True, anchor=patch_embed.
- tinyvit_exact_identity: fallback layers=2, backward calls=3, second backward=True, anchor=patch_embed.
- tinyvit_exact_partial: fallback layers=2, backward calls=3, second backward=True, anchor=patch_embed.

All recorded BK caches are empty after each step. Reserved memory includes CUDA allocator caching and disposable warmup; allocated memory is the training-phase peak. BK cache bytes count actual distinct retained tensor storage (views deduplicated); temporary-gradient bytes count gradient payloads, not allocator peak.

Small smoke measurements are diagnostic and do not establish formal speed or utility claims. No formal jobs were started by the smoke workflow.

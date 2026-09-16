# Exp22 — paired CIFAR-10 TinyViT DP geometry

This experiment compares `dp_sgd`, A-only DP-KFC with `power=0.4`, and full
synthetic KFAC under one logical-batch DP protocol.  The model is the
repository TinyViT converted from packed `nn.MultiheadAttention` to explicit
`q_proj`, `k_proj`, `v_proj`, and `out_proj` Linear maps.

All geometry and clipping code is local to this directory.  A physical batch
is only a memory chunk: clipped sums are accumulated over eight chunks, then
one Gaussian noise draw, one AdamW update, and one RDP accountant step are
performed for each logical batch.

Run the tests and the short three-method smoke protocol with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m pytest \
  exp22/test_exp22.py -q -o cache_dir=exp22/.pytest_cache
conda run -n curve bash exp22/run_all.sh --smoke
conda run -n curve env PYTHONPATH=src:. python -B exp22/analyze.py --smoke
```

The formal 15-job protocol is intentionally not started by this implementation.

## BK implementation

Linear sample gradients are never materialized as `[B,Dout,Din]`. Linear
norms use Ghost Gram identities, and clipped aggregates are reconstructed
directly from transformed activation/backprop factors. The production Linear
strategy is `bk_ghost` (tile size 8). LayerNorm uses analytic
`identity_analytic` geometry, while `pos_embed` uses direct
`identity_direct` geometry.

## Memory metrics

`bk_cache_bytes` is the peak retained activation/backprop BK cache payload
for one physical chunk; the persistent logical aggregate is excluded.
`temporary_per_sample_grad_bytes` is a legacy field name for the peak
temporary BK workspace payload, excluding both retained BK cache and the
persistent logical parameter aggregate. It includes Ghost Gram workspace,
direct aggregate reconstruction workspace, and analytic LayerNorm tensors.
`fallback_temporary_grad_bytes` measures fallback gradient workspace and is
zero for the current explicit-attention TinyViT.

## Counters

`logical_steps`, `optimizer_steps`, `noise_events`, and `backward_calls` are
epoch-local; the last is the number of physical private reverse calls.
`accountant_steps` and `epsilon` are cumulative through the current epoch.
One logical batch has one optimizer step, one Gaussian noise event, and one
RDP accountant step.

Builder counters distinguish `builder_vjp_calls` (backward/VJP invocations),
`builder_reverse_vectors` (synthetic sample reverse vectors processed), and
`builder_samples` (synthetic samples). For the default full-KFC calibration,
these are 80, 2560, and 2560 respectively; A-only uses 80 forward calls and
zero reverse calls, and DP-SGD leaves all builder counters at zero.

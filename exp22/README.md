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
conda run -n curve env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  pytest -q --cache-dir exp22/.pytest_cache exp22/test_exp22.py
conda run -n curve bash exp22/run_all.sh --smoke
conda run -n curve env PYTHONPATH=src:. python -B exp22/analyze.py --smoke
```

The formal 15-job protocol is intentionally not started by this implementation.


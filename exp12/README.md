# Exp12: data-free curvature diagnostics

Exp12 freezes MNIST SimpleCNN (random initialization by default), never trains,
and never reports final accuracy. Private MNIST images, without labels, enter
only the diagnostic oracle. These diagnostics are not a DP release.

- KFAC-U-1 is the existing DP-KFC synthetic KFAC baseline: pink noise, uniform
  labels, sum-reduction score vectors, bias-augmented activations, and spatial
  averaging for convolutions. No factor ridge is added before measurement.
- KFAC-M-k samples model probabilities: Monte-Carlo true Fisher.
- KFLR propagates the nine columns of the softmax Hessian square root: an exact
  GGN **C factor**, combined with the same Kronecker approximation as KFAC.
- KFRA averages the output Hessian before explicit weight/ReLU/MaxPool curvature
  recursion. For convolutions it retains spatially varying channel blocks but
  discards cross-location covariance. It averages joint gate/argmax indicators
  over probes and decouples them from the propagated expected curvature. This
  spatial block approximation is additional to expectation decoupling; it is
  not full spatial KFRA and may account for a gap against KFLR. No VJPs are used.

Every synthetic estimator receives the identical cached pink-noise tensors.
A factors use sample-weighted averages (and spatial averages for convolution).
Parameters are frozen; input gradients permit VJPs without parameter gradients.
The fixed private subset is selected with seed + 2, synthetic inputs with seed + 1.
`--checkpoint PATH` accepts a raw SimpleCNN state_dict.

## Run

```bash
conda run -n curve pytest -q exp12/test_exp12.py
conda run -n curve python exp12/run_diagnostics.py --smoke
# Full experiment: run manually after inspecting smoke output.
conda run -n curve python exp12/run_budget_sweep.py
```

Defaults: 10 batches of 256 synthetic inputs, 2560 private images, damping
0.001, k = 1/3/9 and label seeds 0/1/2. Smoke uses four synthetic and four
private images, with two label seeds. MNIST must already exist in repository
`exp9/data/`. Both runners accept `--device`, `--output`, `--seed`, `--label-seeds`,
`--synthetic-batches`, `--batch-size`, and `--private-samples`.

## Outputs and interpretation

`results/` holds config.json, correctness.json (pytest session outcome),
curvature_error.csv, whitening.csv, mc_convergence.csv, compute_budget.csv,
and summary.csv. Summary is an unweighted layer/seed mean; consult per-layer
rows before drawing conclusions. MC error compares C against synthetic KFLR;
other errors compare against private KFLR. Multiple seeds quantify sampling
variation; four smoke samples cannot establish a hypothesis.

Kronecker Frobenius metrics never materialize a Kronecker matrix. Whitening uses
inverse square roots of damped estimator factors. **Whitening eigenvalues are
floored at 1e-7 times each factor's maximum eigenvalue** for finite CSV values.
Thus `condition_number` and log spread describe the floored spectrum. `A_rank`,
`C_rank` and dimensions identify singular/truncated spectra: a true singular
oracle has infinite condition number, regardless of this finite report. The
log spread uses population variance and the identity Var(X+Y)=Var(X)+Var(Y).

Compute rows distinguish batched `vjp_calls` from per-example `reverse_vectors`:
KFLR and k=9 both cost nine vectors per input; KFRA costs zero. Build time includes
activation factor accumulation, forward passes and curvature construction, but
excludes probe generation and diagnostic metrics. CUDA peaks are absolute process
allocated/reserved peaks (including resident caches/references), not isolated
incremental estimator memory. KFRA currently retains activations for the cache;
its zero VJP count alone does not establish a memory or time advantage.

Uniform scores have expectation
`E_u[(p-e_y)(p-e_y)^T] = diag(u)-u u^T + (p-u)(p-u)^T`,
whereas model-sampled scores give `diag(p)-p p^T`. Increasing k removes variance,
not this difference. At near-uniform random initialization, bias may be small.

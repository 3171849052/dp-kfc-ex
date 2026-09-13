# Exp12: frozen-model curvature diagnostics

Exp12 does not train, run DP training, attach Ghost Clipping, or compare final
accuracy. Private MNIST inputs are used without labels only by the diagnostic
oracle; they never feed back into synthetic estimators. Diagnostics are not a DP
release. All synthetic methods receive the **same cached pink-noise inputs**, using
the existing DP-KFC generator. Parameters remain frozen.

## Estimators

- **KFAC-U-1/3/9**: uniform labels and unscaled score VJPs. U1 is the existing
  DP-KFC synthetic KFAC baseline. A regression compares every SimpleCNN A/C factor
  to `KFACRecorder` + `compute_covariances(eps=0)` on identical inputs and labels.
- **KFAC-M-1/3/9**: categorical model-probability labels, Monte-Carlo true Fisher.
- **KFLR**: column-wise VJPs of the softmax Hessian square root,
  `H = diag(p) - ppᵀ`. Nine columns for MNIST; exact GGN C factors combined with a
  Kronecker approximation, not an exact full parameter GGN.
- **KFRA**: standard expected preactivation GGN recursion for sequential
  Linear/ReLU MLPs, used in correctness tests. Start with `C_L = E[H]`, then use
  `C_l = E[D_l W_nextᵀ C_next W_next D_l]`.
- **KFRA-block**: the SimpleCNN recursive approximation, retaining spatially
  varying channel blocks and dropping cross-location covariance. Its gap from
  KFLR includes both **recursive expectation approximation** and **spatial
  cross-location truncation**. It is not full spatial KFRA. No heuristic repairs
  are applied to its factors.

A includes bias augmentation; convolutions average over samples and spatial
positions, as in existing DP-KFC. No ridge is added before comparison. A Gram
sums and position counts accumulate per batch. C uses sample weighting and,
for convolution, spatial averaging. KFAC/KFLR release each batch's graph.
KFRA/KFRA-block use one forward per batch to accumulate A, output Hessian sums,
and joint ReLU/MaxPool gate sums; a second stage recurses only on these compact
statistics. They do not save intermediate activations for the full probe cache
or forward the cache again. MaxPool routes follow PyTorch argmax and ReLU's zero
subgradient. The CNN recursion is specifically for repository SimpleCNN (28x28).

## Run

```bash
conda run -n curve pytest -q exp12/test_exp12.py
conda run -n curve python exp12/run_diagnostics.py --smoke
# Full runs are manual:
conda run -n curve python exp12/run_budget_sweep.py
conda run -n curve python exp12/run_budget_sweep.py --checkpoint PATH
```

Both runners accept the same CLI arguments, explicitly forwarded by the sweep
entry point: `--checkpoint`, `--synthetic-batches`, `--batch-size`,
`--private-samples`, `--label-seeds`, `--seed`, `--device`, `--output`,
`--warmup`, `--repeats`, and `--damping`.
Defaults: 10 synthetic batches of 256, 2560 private images, damping 1e-3,
label seeds 0/1/2, two warm-ups and five measurements **per estimator/seed**.
Smoke uses four synthetic and four private images, two label seeds, and retains
the same warm-up/repeat defaults. MNIST is read from existing `exp9/data/`.
Synthetic generation uses seed + 1, private subset selection uses seed + 2.
`--checkpoint PATH` expects a raw SimpleCNN state_dict; none is invented or trained.
Without it, only random-init diagnostics run. At random initialization p is often
near uniform, so U and M may be almost identical. Assess systematic uniform-label
bias preferentially using a trained frozen checkpoint.

Uniform scores have expectation
`E_u[(p-e_y)(p-e_y)ᵀ] = diag(u)-uuᵀ + (p-u)(p-u)ᵀ`;
model-sampled scores give `diag(p)-ppᵀ`. Increasing k removes sampling variance,
not this difference. Non-uniform tiny-model tests check both limiting expectations.
A tiny smoke is not evidence that a hypothesis holds, nor must every finite-seed
layer error decrease monotonically with k.

## Scientific outputs

`results/config.json` records the run; `correctness.json` records pytest outcome.

- `curvature_error.csv`: per estimator, seed, and layer; A/C cosine and relative
  Frobenius errors plus Kronecker error against private KFLR, without materializing
  any Kronecker matrix.
- `whitening.csv`: `floored_condition_number`, `floored_log_eigenvalue_spread`,
  `A_effective_rank`, `C_effective_rank`, dimensions, and `spectral_floor_ratio`.
  Whitening applies inverse square roots of damped estimator factors to oracle
  factors. Each resulting spectrum is floored at 1e-7 times its largest eigenvalue.
  These are floored diagnostics; a truly singular oracle has **infinite true
  condition number**. Log spread uses population variance of pairwise summed logs.
- `mc_convergence.csv`: per-layer/per-seed C errors against synthetic KFLR.
- `summary.csv`: estimator × layer means over label seeds, plus explicitly marked
  `global_mean` rows. Global means are only a convenient unweighted layer summary;
  per-layer and individual-seed results remain available and should guide inference.

## Benchmark outputs

Each estimator/seed independently completes warm-up before measurement. Remaining
synthetic jobs use fixed-seed shuffled order; references are built first and are
also independently warmed. Before each measured build, previous GPU outputs are
deleted, garbage is collected, the CUDA cache is emptied, and CUDA is synchronized.
Resident model and input caches are included in the recorded baseline. Reference
and diagnostic factor outputs live on CPU, outside the timed region.

`benchmark_raw.csv` contains all repeat timings and allocated/reserved baselines,
absolute peaks, and peak increments. CUDA is synchronized before and after timing.
`compute_budget.csv` reports median/min/max build time; `build_median_seconds`
(alias `median_build_seconds`) is the primary timing statistic, never first-run
time. Memory summaries use the maximum observed value over repeats. The primary
memory metric is `peak_allocated_increment = peak_allocated_absolute -
baseline_allocated`. Reserved memory is an allocator metric, auxiliary only,
and cannot establish the primary algorithm-memory conclusion. Warm-up, garbage
collection, cache clearing, probe generation, CPU transfers and scientific metrics
are excluded from build timing. All methods include A construction in build cost.

Counts are **per build**, excluding warm-up/repeat multipliers:

- `forward_calls`: number of synthetic/private input batches.
- `vjp_calls`: batched `autograd.grad` calls; k per KFAC batch or nine per KFLR batch.
- `reverse_vectors`: k × samples for KFAC, nine × samples for KFLR.
- KFRA/KFRA-block have zero VJPs/vectors; lower cost must be established from actual
  wall time and memory, not inferred from this count.
- `synthetic_samples`: synthetic inputs processed; zero for the private oracle.
  `samples` counts processed inputs for either path.

Correctness checks cover Hessian factors, Fisher/GGN equivalence, one/two/deeper
MLP recursion, CNN local weight/reshape/gate recursion against independent
Jacobians, original DP-KFC U1 equivalence, shared A and single-pass streaming,
MC convergence and non-uniform U/M limits, and oracle import isolation.

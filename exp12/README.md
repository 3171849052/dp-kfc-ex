# Exp12: frozen-model curvature diagnostics

The curvature diagnostic path does not train, run DP training, attach Ghost
Clipping, or compare final accuracy. The optional SGD trajectory producer below
is a separate checkpoint-generation path. Private MNIST inputs are used without labels only by the diagnostic
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

## Generate frozen checkpoints with ordinary SGD

The optional training entry point is separate from curvature diagnostics. It uses
MNIST, the same normalization, repository SimpleCNN, and ordinary **non-private**
SGD: lr=0.5, momentum=0, weight decay=0, batch size=256, shuffled batches, seed=42,
CUDA by default, five epochs. It uses no curvature estimator or Ghost Clipping.
Training is only for producing model states; high accuracy or monotone confidence
is not guaranteed with this learning rate.

```bash
conda run -n curve python exp12/train_trajectory.py
# Only two updates; outputs go to exp12/checkpoints/smoke/.
conda run -n curve python exp12/train_trajectory.py --smoke
```

Checkpoints are raw `model.state_dict()` files in `exp12/checkpoints/`, named
`sgd_seed42_step000000.pt`, `sgd_seed42_step000050.pt`, etc., plus
`sgd_seed42_final.pt`. Save steps are 0/50/100/250/500/1000/2000 and final; steps
beyond training are skipped. Five MNIST epochs have 1175 updates, so step 2000 is
skipped. Final always has its own file. If it coincides with a scheduled step, the
trajectory keeps only the scheduled row; otherwise it adds a final row. The sweep
asserts that (seed, step) is unique before launching any diagnostics. No optimizer is saved. `--epochs`, `--max-steps`, `--seed`, `--device`,
`--diagnostic-samples`, `--test-samples` and `--output` can be set explicitly;
output must remain inside Exp12.

`metadata.json` records settings and fixed diagnostic indices. `trajectory.csv`
has one row per checkpoint. `epoch` is fractional completed epochs. `train_loss`
is checkpoint cross entropy on a **fixed training diagnostic subset**, including
step 0, rather than a last-minibatch loss. Prediction confidence, entropy and KL
use that same subset (2560 train images, seed + 2, matching default oracle subset
selection). `test_accuracy` is a fraction over all 10000 test images by default.
Smoke uses 32 diagnostic images and 64 test images. These evaluations never
update the model or choose its optimizer settings. KL is measured on MNIST inputs,
not synthetic pink noise; interpret its relationship to synthetic curvature error
accordingly. Entropy is in nats; normalized entropy divides by log(10), and
`mean_kl_to_uniform = log(10) - mean_prediction_entropy`.

Training exits after saving checkpoints and never launches diagnostics.

```bash
conda run -n curve python exp12/run_budget_sweep.py --checkpoint exp12/checkpoints/sgd_seed42_step000500.pt --output exp12/results/sgd_step500
# Explicit opt-in: processes every row in the saved trajectory.
conda run -n curve python exp12/run_trajectory_sweep.py
```

The trajectory runner reads `exp12/checkpoints/trajectory.csv`, invokes existing
curvature diagnostics separately per checkpoint, and writes to
`exp12/results/trajectory/<checkpoint_name>/`. `--trajectory` and `--output` select
other paths; remaining diagnostic options (e.g. `--device`, `--seed`, `--smoke`)
pass through unchanged. Use the same diagnostic seed across checkpoints to share
synthetic probes/private subset selection across model states.

`trajectory_summary.csv` joins checkpoint step/epoch and domain-specific entropy/KL
with estimator × layer
means over label seeds: C error against **synthetic KFLR**, Kronecker error against
**private KFLR**, floored condition number, and median build time. The build column
is the mean over seeds of each seed's measured median. KFLR self-error is zero;
KFRA-block is included. The additional `C_relative_error_vs_KFLR` column in each
`curvature_error.csv` supplies this comparison without changing any estimator or
existing metric definition. Per-seed source rows remain in each checkpoint folder.


## Prediction domains and CUDA execution

`state_metrics.json` records the four `synthetic_*` prediction statistics, computed
in double precision directly from the **same cached pink-noise tensors** passed to
the curvature estimators. No probes are regenerated. The statistics pass is outside
benchmark timing and reverse/forward budget counts, and does not change the frozen
model. The checkpoint path is recorded when supplied.

Trajectory summaries preserve the four real-data statistics under `mnist_*` names
and include all four `synthetic_*` statistics. The primary mechanism comparison is
**`synthetic_mean_kl_to_uniform` vs `C_relative_error_vs_KFLR`**. MNIST-domain KL
only describes the model's training state; it should not substitute for probe-domain
KL when interpreting synthetic curvature bias. Original trajectory.csv field names
remain unchanged for compatibility.

Training and diagnostic entry points set cuDNN benchmark=False,
cuDNN deterministic=True, CUDA matmul TF32=False, and cuDNN TF32=False.
Diagnostics initialize CUDA with a small matrix-multiply VJP before measurement.
Autograd multithreading is disabled within these entry points so reverse-mode work
uses the caller thread with its initialized CUDA context, avoiding lazy cuBLAS
context creation in an autograd worker. Warnings are not suppressed. Existing
per-estimator warm-ups and synchronized repeated measurements remain in place.
These settings improve same-environment repeatability; they do not promise bitwise
agreement across hardware or PyTorch versions.

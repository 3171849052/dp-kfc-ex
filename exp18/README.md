# Exp18: Lightweight DP-KFC

Independent research prototype. All code, logs, caches and results live here;
Exp12/13/14/14b and existing MNIST data are imported/read without modification.
No formal training was requested during implementation: only pytest and one tiny
end-to-end smoke (ten methods, one four-example private batch each).

## Run

```bash
conda run -n curve bash exp18/run_all.sh
```

The script runs all **10** methods and automatically updates CSVs, plots and
`results/pareto_summary.md` after each method. No `full` or `identity` training
method is exposed. Existing output in `exp18/results/` is overwritten on rerun.

```bash
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$PWD/exp18/.cache" MPLCONFIGDIR="$PWD/exp18/.cache/matplotlib" CUDA_CACHE_PATH="$PWD/exp18/.cache/cuda" conda run -n curve python -m pytest exp18/test_exp18.py -q -o cache_dir=exp18/.pytest_cache
conda run -n curve bash exp18/run_all.sh --smoke
```

Smoke writes to `results/smoke/`, uses real MNIST, SimpleCNN, synthetic pink
noise, all production builders, Ghost Clipping, SGD, evaluation and analysis.
It uses four private examples, four synthetic examples, one epoch, zero noise;
noise ordering is independently checked with nonzero noise in pytest. Smoke
accuracy and retention are not scientific evidence. Formal runs include a tiny
disposable warmup for each action path, outside measured results and isolated
from RNG state.

## Fixed protocol

MNIST / `dp_kfac.models.SimpleCNN`, seed 42, five epochs, batch 256,
shuffle and drop_last, SGD lr .5 / momentum 0 / weight decay 0, epsilon 1,
delta 1e-5, clip bound 1, damping .001, beta .25. One shared noise calibration
uses Exp14b's fixed-batch shuffled RDP convention, 234 steps/epoch and 1170
steps total. This retains the existing accounting convention, including its
sampling assumptions. Each rebuild uses 10 × 256 synthetic pink-noise samples,
KFAC-U uniform labels. No private samples or gradients enter the builder/scale.

Epochs in Exp18 outputs and refresh schedules are zero-based. Synthetic and
label seeds use `42+10000+(epoch+1)` and `42+20000+(epoch+1)`, matching Exp14b's
one-based loop. Sketch RNG is a separate generator seeded
`42+30000+(epoch+1)`. Initialization, shuffle generator and private noise generator
(`42+40000`) come directly from Exp14b/Exp13. Replayed low-rank passes use identical
synthetic samples and labels; neither consumes the private noise stream.

| Method | A | C | Rebuild epochs |
|---|---|---|---|
| diag | diagonal | diagonal | 0–4 |
| a_only | full | identity | 0–4 |
| c_only | identity | full | 0–4 |
| fullA_diagC | full | diagonal | 0–4 |
| diagA_fullC | diagonal | full | 0–4 |
| rank4 / rank8 / rank16 | low-rank + tail | low-rank + tail | 0–4 |
| refresh2 | full | full | 0, 2, 4 |
| frozen | full | full | 0 |

## Implementation and mathematics

`run_exp18.py` installs scoped adapters on the imported Exp14b runner: only
configuration, synthetic cache/build calls and optional diagnostics are replaced.
Its training loop, accounting, initialization, evaluation and timers are reused
without copying. Lazy adapters retain the operator, skip synthetic generation
and report zero builder budget on reuse epochs. Optional old geometry and state
metric forwards are disabled; Exp18 records its own scale and low-rank diagnostics.

`builders.py` reuses Exp12 `forward`, `layers` and KFAC-U VJP semantics. Full lazy
methods directly call the existing `estimate`. Compressed methods accumulate
only requested statistics: sums of squares for diagonal, covariance only for
requested full sides, nothing for identity. Biases are augmented as in Exp13.
The spatial normalization is the same as Exp12 for the fixed-size synthetic batches.

`lowrank.py` makes two streaming passes. First accumulate `Y = sum Xᵀ(XΩ)` and
trace, with Gaussian Ω width `min(d,r+4)`, and compute `Q=qr(Y)`. Replay the same
samples to accumulate `B=sum (XQ)ᵀ(XQ)/N`. Its top r eigenpairs yield
`U=QV`, lambda, and `tau=(trace(F)-sum(lambda))/(d-r)`. For r>=d, accumulate the
exact factor as specified. For r<d, no full covariance is formed; if oversampling
reaches d, the range basis and projected matrix can have width d, but are still
obtained via the range finder. No dense reconstructed factor is stored or applied.

`operators.py` implements identity, diagonal, dense and low-rank actions:
`f(F)x=f(tau)x+U[(f(lambda)-f(tau))*(Uᵀx)]`, with
`f(t)=(t+.001)^(-.25)`. Dense factors use double eigendecomposition; low-rank
projected arithmetic is double, and applied powers/bases are float32. The
existing Exp14b PSD eigenvalue zero-floor convention is retained. A tiny negative
tail caused by trace subtraction is floored only after asserting it lies within
`1e-12*trace`; larger negative tails fail. There are no identity fallbacks.

For each nonidentity side, compute
`h_b=sum lambda*(lambda+damping)^(-2b)` including `(d-r)` copies of tau.
Identity actions have `h_b=d` for both powers. Thus dropped sides remain exactly
identity, rather than applying a damping-dependent scalar to I. Sum products
`m_b=sum_layers h_b(A)*h_b(C)`, then global `s=sqrt(m_.5/m_.25)`.
Both transformed activation and backprop actions get sqrt(s); the aggregate
gets s exactly once. Scale uses only the compressed structure's own spectrum.
These are structure-predicted moments, not measured private gradient moments.

Exp13 Structured Ghost Clipping computes transformed per-example **global**
norms using spatial Gram tiles, clips, aggregates, transforms once, adds Gaussian
noise, divides by batch size, then updates SGD. No per-example parameter gradients
are materialized in production. Tiny explicit gradients occur only in tests.

## Outputs and interpretation

- `metrics.csv`: each epoch accuracy/loss, clipping statistics, norm quantiles,
  epsilon/steps, build/private/evaluation/algorithm time, peak CUDA allocation,
  operator payload bytes/scalars, cumulative rebuild count, forward/VJP/reverse
  vector budgets, raw/reference moments and RMS matching.
- `summary.csv`, `method_summary.csv`: final rows plus summed time/budgets and
  peak memory. Algorithm time means build + private training, consistent with
  Exp14b; evaluation and wall time are separate. Low-rank's replay counts in all
  budgets. Reuse epochs include only adapter overhead in build time.
- `lowrank_diagnostics.csv`: each rebuilt layer/side rank, d, tau, captured trace.
- `config.json`, `baseline_references.csv`, `pareto_summary.csv` / `.md`.
- `accuracy_vs_operator_state_bytes.png`,
  `accuracy_vs_total_algorithm_seconds.png`, `clipping_norms.png`.

Operator bytes count retained action tensors plus global scale and scalar tail
powers, not Python object metadata, model parameters, or temporary builder
workspace. `stored_scalar_count` uses the same definition. CUDA peak allocation
captures live tensor workspace as well as model/operator state. Lazy rebuilding
keeps the old operator alive until replacement, and its peak reflects that.

Historical references are read from seed=42 rows, never copied into training
metrics: **full = .9570** at beta=.25 from `exp14b/results/summary.csv`;
**identity = .9221** at beta=0 from `exp14/results/summary.csv`.
Exp14b beta=0 is rescaled and is not identity. Utility retention is
`(acc_method-.9221)/(.9570-.9221)`, without clipping to [0,1]. Historical baselines
are horizontal utility references on plots; their operator state bytes were not
reported in these CSVs. Pareto flags compare new methods only. Historical runtime
is recorded separately and subject to separate-run timing/hardware differences.

After all formal methods finish, the report quantifies diagonal retention,
A-only vs C-only and the hybrid comparison, rank 4→8→16 increments, refresh/frozen
accuracy differences from full, and memory/runtime Pareto membership. Single-seed
comparisons indicate hypotheses rather than establish statistical equivalence.

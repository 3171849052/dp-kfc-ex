# exp10b: Equil scale parameterization

All new source and output for this experiment live here. Historical exp7/9/10 are unchanged.
At the user's explicit request, the standalone training entry also loses Equil
scale_min/scale_max defaults, validation, YAML fields, run-name cap suffix and
algorithm hard clamp. Its Fisher estimator, stabilization and DP clipping are unchanged.
Run from the repository root in conda environment `curve`.

## Fixed configuration and reuse

`config.py` imports exp10's constants, including exp9's training constants:
MNIST (existing exp1/data cache; exp10's exp9/data path is absent on this server), SimpleCNN, seeds 42 and 7, 5 epochs,
batch size 256, SGD lr=0.5, momentum=0, weight decay=0, epsilon=1, delta=1e-5,
max grad norm=1, synthetic batch size=256, 10 synthetic batches per epoch,
8 Rademacher probes, tau=1. No hyperparameter sweep and no scale-bound config.
The resolved configuration is written to results/config.json.

Reuse: exp10.layerwise_statistics, exp7.fisher_statistics, exp9.pink_batches,
the existing SimpleCNN, Opacus GradSampleModule, clipping/noise, evaluation,
RDP accountant, and exp10 CSV aggregation. The runner retains exp10's seeded
loaders and nested fork_rng streams (synthetic seed+10000+epoch;
probes seed+30000+epoch). Each method rebuilds at every epoch on its own model.

## Operators

The estimator is unchanged. Compute gamma=tau*median(e) across all trainable
coordinates, s=(e+gamma)^(-1/2), then subtract the coordinate-weighted mean log
scale before exponentiation. Reductions use float64; applied gains use the
model dtype. There is no [0.1,10] clamp or substitute bound in the config or
operator algorithm. The clipping factor cap at 1 remains part of DP clipping.

Every Linear/Conv weight is flattened to [out, flattened input] and its bias
appended as the final column. From the SAME module-elementwise target S:

- Module-Elementwise retains S.
- Factorized uses log r=row_mean(log S), log c=column_mean(log S)-mean(log S).
  It removes any residual global weighted log mean from log r. Only r and c
  are retained for private training; gradients receive separate row and column
  multiplies, with the final column factor applied to bias.
- Layer-Scalar uses exp(mean(log S)) per module and removes the global mean
  weighted by the module coordinate counts.

Full-Fisher uses the full-model Hutchinson projection and the same unbounded
stabilization/normalization. DP-SGD applies the identity.

No Ghost Clipping: exact grad_sample -> scale -> global per-example clipping
-> isotropic Gaussian noise -> SGD. The sigma=0 smoke run reports infinite
epsilon intentionally; it is a functional test, not a private run.

## Diagnostics and DOF

operator_approximation.csv identifies source_method (the model trajectory)
separately from method (the projected operator). Each module-wise source build
generates all three projections from identical statistic and target objects.
SHA256 columns verify this identity. This does not force different training
trajectories to use the same numerical Fisher after their models diverge.
Plots compare all projections on the Module-Elementwise source trajectory.

The augmented shapes are conv1=16x10, conv2=32x145, fc1=128x1569, fc2=10x129.
There are 206922 trainable coordinates. Effective operator DOF:
- Module-Elementwise and Full-Fisher: 206922-1 = 206921.
- Factorized: sum(out+augmented_input-1)-1 = 2034.
- Layer-Scalar: 4-1 = 3.
Per-module DOF removes the factorization gauge; the global row additionally
removes the single GM constraint. Stored scalar counts before constraints are
206922, 2039, and 4 respectively (factorized includes 4 redundant gauges).
DP-SGD is fixed identity, DOF=0.

Diagnostics: log-scale RMSE, Pearson and tie-aware Spearman, GM, gain
min/p1/p10/median/p90/p99/max, and log10(max/min). Correlation with a constant
vector is undefined: per-module Layer-Scalar and DP-SGD correlations are blank,
with correlation_defined=false, rather than fabricated zeros. Global scalar
correlations remain defined when module gains differ.

Build time includes diagnostic projection, hashing and correlation computation;
private train time excludes these. This common analysis overhead is not an
estimate of optimized operator construction time. Dense factorized matrices
exist only transiently for diagnostics/tests, never in the private update.
Unnoised training diagnostics follow exp10 and are analysis-only.

## Commands

```bash
source /HOME/sysu_qling/sysu_qling_3/miniconda3/etc/profile.d/conda.sh
conda activate curve
export PYTHONDONTWRITEBYTECODE=1
python -m unittest exp10b.test_exp10b exp10b.test_standalone -v > exp10b/results/tests.log 2>&1
python -u exp10b/run_exp10b.py --smoke > exp10b/results/smoke.log 2>&1
python -u exp10b/run_exp10b.py > exp10b/results/full_run.log 2>&1
```

Quick tests cover exp10 estimator identity and unclipped numeric equivalence,
unbounded gains, row/column least-squares residuals, scalar constancy, global
weighted GM, shared target/statistic hashes, RNG restoration, and exact
sigma=0 clipped private updates against a dense reference. The smoke run
tests all five methods and CSV/plot generation. Full completion checks 10 runs,
50 epoch rows, 500 approximation rows, finite metrics, global GM, and paired
accuracy differences. The gain anomaly flag is a reporting-only threshold
of more than six decades; it never clips or changes the algorithm.

Outputs: metrics.csv, summary.csv, operator_approximation.csv, accuracy.png,
complexity_vs_accuracy.png (log DOF x-axis), approximation_by_layer.png,
verification.txt, config.json, logs, and smoke/ outputs.

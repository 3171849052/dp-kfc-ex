# Exp11: exact two-pass structured Ghost Clipping

Run from the repository root:

```bash
conda run -n curve pytest -q -o cache_dir=exp11/results/pytest_cache exp11/test_exp11.py
conda run -n curve python exp11/run_exp11.py --smoke
conda run -n curve python exp11/run_exp11.py
```

The default runs DP-SGD, Factorized Equil and DP-KFC, each Exact/Ghost,
seeds 42 and 7, five epochs. All outputs are under `exp11/results/`;
smoke uses `results/smoke/`, one 256-example MNIST batch per method,
one epoch, seed 42 and zero noise. No gradient accumulation.

Configuration follows exp10b: SGD lr=.5, momentum=0, weight_decay=0,
epsilon=1, delta=1e-5, C=1; Equil tau=1, 8 probes, 10 synthetic
batches of 256, no hard clamp. We reuse exp10b's statistic, scale function
and Factorized Operator constructor directly, omitting its audit machinery.
KFC calls exp6's covariance/inverse-square-root builder (10 synthetic
batches, damping=.001). Builders run once per epoch on synthetic inputs;
only the Factorized Equil build uses GSM in Ghost methods. KFC builds on
the ordinary model without per-sample gradients.

All private training batches are exactly 256 (`drop_last=True`): 234 batches
per full MNIST epoch, discarding the shuffled last 96 examples. Sigma is
calibrated for 1170 steps. Accounting retains the existing experiments'
RDP/sample-rate convention (256/60000 with shuffled fixed batches); this
is not a new analysis of shuffled sampling. Train loss and clipping diagnostics
are unnoised research metrics, not private releases.

For each layer, augment unfolded activations with a coordinate of ones.
Transform backprops by L and activations by R^T. The exact identity
`||sum_t b_t a_t^T||_F^2 = sum_st <b_s,b_t><a_s,a_t>` provides the Gram
backend. Each layer uses Gram when `T² <= d_out * d_in_augmented`, with
spatial rows tiled by 32. Otherwise it computes `B' @ A'^T`, reduces its
squared Frobenius norm, and immediately releases the temporary matrix.
Neither backend stores per-sample matrices on parameters or across layers.
SimpleCNN uses the matrix backend for both convolutions and Gram for both
Linear layers. Both backends use transformed activations/backprops and are
exact; spatial tiling is not batch accumulation.
Linear layers with one position reduce to the outer-product norm identity.
All layer norms contribute before global clipping. The second ordinary
backward weights each loss by its detached clip factor, then applies LGR
once to the aggregate, adds isotropic Gaussian noise once, divides by 256,
and steps SGD and the accountant once. Bias participates in the same
augmented matrix, including KFC's weight/bias mixing.

Both paths retain the repository's `C/(norm+1e-6)` numerical clipping
convention. Exact uses GSM and the original Factorized/KFC per-sample
transforms as the reference. Private Ghost backward uses only plain autograd
and activation/output hooks. It neither reads nor writes grad_sample.
The current runtime covers the model's non-shared Linear and groups=1 Conv2d
layers, without stochastic layers.

Tests compare float32 norms, clip factors, zero-noise aggregates, bias
coordinates and one SGD update on SimpleCNN, with the real builders.
They also check that Ghost retains no grad_sample or captured activations.
Tolerance is elementwise `atol=2e-6, rtol=2e-4`; the error report records
max absolute and relative infinity-norm errors (max error / max reference
magnitude), avoiding misleading relative errors at zero coordinates.

`metrics.csv` records epoch utility, clipping, epsilon, build/train time,
throughput and CUDA peaks. Primary CUDA peaks cover private training;
separate build and total peaks expose synthetic-builder costs. The allocator
cache is emptied only at phase boundaries, consistently for both modes.
Ghost first-pass time includes forward/backward, norm and clip computation;
second-pass time includes the second forward and weighted backward.
Aggregate-transform time is separate. Timings synchronize CUDA; they are
phase wall times, not kernel-only profiles. Exact Ghost-specific times are zero.

`paired_accuracy.csv` is an equivalence diagnostic, not an algorithm ranking.
`ghost_comparison.csv` isolates Factorized Equil Ghost vs DP-KFC Ghost:
compare final test accuracy/loss, private train time and phase costs, and
private/total allocated and reserved peaks. With only two seeds, utility
differences are exploratory. `summary.csv` contains final rows per run.

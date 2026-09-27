# ExpM1c: ViT-Tiny / CIFAR10 spectral tuning

Only `vit`, `dp_kfm_a`, public/pink, seed 42. Training, BK clipping,
Poisson sampling, transforms, calibration RNG, AdamW, accounting, and
975 optimizer steps are copied from ExpM1b. Changes are beta and damping.
All runtime caches, logs, and new results stay under this directory;
datasets are read from repository `data/` with `download=False`.
The offline pretrained checkpoint is copied from the existing Exp30 cache.

For each affine layer, `Q = tau * (A + lambda I)^beta`, with
`tau = d_A / Tr((A + lambda I)^beta)`. Identity parameters retain identity
geometry. The optimizer receives raw globally clipped gradient sums;
metric factors are used only to calculate norms. Noise is matched to Q,
with expected energy `sigma^2 C^2 * layer_dimension`. The noisy sum is
always divided by 256. The inherited clipping numerical stabilizer is
unchanged. No G factors are constructed or consumed.

`geometry.csv` reports `trace_R` for the input-side R and additionally
`trace_R_block` for its output-repeated block; ExpM1b called the latter
`trace_R`. `trace_S_per_dim` remains one. The inherited spectral diagnostics
(including `A_condition_raw`, which measures damped A before the beta power)
and fixed private A alignment diagnostics are retained. The alignment
sample is diagnostic only; there are no oracle training runs. Unnoised
private diagnostics make these research results unsuitable as a DP release.

## Schedule

| Phase | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| A, lambda=.001 | public then pink, beta=.1 | public then pink, beta=.2 | public then pink, beta=.3 | public then pink, beta=.4 |
| B, selected beta per source | public then pink, lambda=.0001 | public then pink, lambda=.01 | public lambda=.1 | pink lambda=.1 |

Phase A has eight new runs. `analyze_beta.py` reads their completion markers
and the two ExpM1b beta=.25 runs, then writes `results/beta_summary.csv`
and `results/selected_beta.json` with numeric `public` and `pink` values.
Selection maximizes epoch-five test accuracy, then accuracy AUC, then
epoch-five mean clip factor, then prefers smaller beta. Each source is
selected independently from .1, .2, .25, .3, .4.

Phase B loads this JSON and verifies it against completed Phase A results.
It adds six runs at lambda=.0001, .01, .1. The .001 center is reused from
Phase A or the ExpM1b reference. No old run directories are copied.
`analyze.py` writes `lambda_summary.csv` (eight rows) and
`final_summary.csv` (two tuned optima, two ExpM1b references, one ExpM1
DP-SGD reference). Best lambda uses the same descending accuracy/AUC/clip
ordering, then smaller lambda for a deterministic final tie.

Summary clipping and spectral statistics use epoch five; spectral medians
exclude identity parameters. AUC uses the unchanged ExpM1b trapezoidal
calculation over epochs 1–5. Plots live in `results/beta_plots/` and
`results/lambda_plots/`, including accuracy, AUC, clipping, mean clip factor,
and accuracy versus median condition. Damping axes are logarithmic.

Each new run path includes source, beta, lambda, and seed, under
`results/beta/` or `results/lambda/`. Logs use the same names under `logs/`.
A GPU executes its assigned runs serially; four GPU processes run in
parallel. Any worker failure produces a nonzero phase exit and prevents
subsequent stages. Existing run directories or logs cause failure. There
is no resume, recovery, fallback, or dynamic scheduling.

## Lightweight checks and launch

Checks run CPU toy models and synthetic analysis fixtures, read existing
reference CSVs, and verify shell syntax; they do not initialize a formal
ViT model or train an experiment.

```bash
conda run --no-capture-output -n curve python -B expm1c/checks.py
bash -n expm1c/run_beta.sh
bash -n expm1c/run_lambda.sh
bash -n expm1c/run_all.sh
```

Full launch (not executed during implementation):

```bash
conda run --no-capture-output -n curve bash expm1c/run_all.sh
```

## Additional small-damping runs

`run_lambda_extra.sh` adds four runs using the existing selected_beta.json
without changing beta or repeating earlier runs. GPU 0: public lambda=1e-5;
GPU 1: pink lambda=1e-5; GPU 2: public lambda=1e-6; GPU 3: pink lambda=1e-6.
Results remain under results/lambda with source/beta/lambda/seed names;
worker logs are under logs/lambda_extra. The original two-stage analysis
and grid remain unchanged; these additional runs are a separate extension.

# Exp20: activation exponent sweep

Fixed DP-KFC-A + Exp19 Structured Ghost, `p = 0, .125, .25, .375, .5, .625, .75, 1`, seed 42 only. Each power runs in a fresh process. Formal execution is opt-in:

```bash
conda run -n curve bash exp20/run_all.sh
```

Smoke (eight fresh processes, one epoch, one 256-example private batch, one 256-example synthetic batch, sigma zero):

```bash
conda run -n curve bash exp20/run_all.sh --smoke
```

## Protocol and implementation

MNIST / SimpleCNN; 5 epochs; shuffled drop-last batches of 256; 234 steps/epoch, 1170 total. SGD lr=.5, momentum=weight_decay=0, epsilon=1, delta=1e-5, clipping bound=1, damping=.001. Formal noise calibrated with the same RDP/shuffled fixed-batch convention as Exp19. Each process first performs a disposable action-path warmup without optimizer/accountant steps.

`methods.py` ports the Exp19 forward-only A builder: no-grad forward, Conv2d/Linear inputs only, bias augmentation, no C, labels, synthetic backward or VJP. It reads Exp19's corrected activation capture and Structured Ghost implementation without changing them. Every epoch builds from 10×256 pink-noise images using only the current model. Synthetic RNG forks only the selected CUDA device and preserves the CPU generator; private Gaussian noise has its own generator. Initialization/shuffle use seed 42, synthetic uses seed+10000+epoch, private noise uses seed+40000. Data is read from existing `exp1/data` with download disabled.

For each current A, double-precision eigendecomposition gives nonnegative eigenvalues lambda (numerical negatives clamped to zero):

- `m_p = sum_l d_out,l sum_j lambda_j (lambda_j+.001)^(-2p)`.
- `m_reference = m_0.5`; `scale_match = sqrt(m_reference/m_p)`.
- p=.5 has exactly scale=1; p=0 uses an exact identity matrix multiplied by the matched scalar.
- Both Ghost norm activations and the final aggregate apply the same operator. First backward computes global transformed norms by spatial Gram identity, second computes the clipping-weighted ordinary sum; transform aggregate once, add Gaussian noise, divide by batch size, SGD.

No production `grad_sample` is created. Spectral quantiles pool augmented activation eigenvalues equally across layers. Gain is `s_p(lambda+.001)^(-p)`; transformed spectrum is `s_p² lambda(lambda+.001)^(-2p)`; condition proxy is p90/p10, infinite for zero p10. Moment output-dimension weighting is separate from spectral quantile weighting.

## Outputs

Formal outputs: `exp20/results/`; smoke outputs: `exp20/results/smoke/`. Each contains per-run CSV/config files plus `metrics.csv`, `summary.csv`, `power_summary.csv`, `spectrum_summary.csv`, `layer_norm_diagnostics.csv`, `config.json`, `report.md`, and all nine requested PNGs. Per-run smoke snapshots are also saved. Logs and caches stay under exp20.

AUC is the trapezoid integral over observed epochs 1..5 (zero for one-epoch smoke). Clip fraction and p99 comparisons average the per-epoch statistics. All four deltas subtract p=.5. Single-seed descriptive comparisons only, no bootstrap or significance claims. Smoke results cannot support formal utility conclusions.

Build and private training have synchronized phase boundaries; internal profiling records asynchronous CUDA events. CPU diagnostic transfer/postprocessing and evaluation are separately timed. Algorithm epoch time is build+private training. Peak CUDA memory is max(build, train), excluding evaluation. Layer diagnostics are each layer's share of total transformed squared norm. Private norm diagnostics are research measurements, not privatized releases.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$PWD/exp20/.cache" \
MPLCONFIGDIR="$PWD/exp20/.cache/matplotlib" \
CUDA_CACHE_PATH="$PWD/exp20/.cache/cuda" \
conda run -n curve python -m pytest exp20/test_exp20.py -q -o cache_dir=exp20/.pytest_cache
```

Tests require CUDA and existing MNIST. They verify independent eigendecomposition and RMS matching for all powers; exact identity at p=0; forward-only budget and output lifetime; RNG pairing/isolation; Ghost against independent real-MNIST per-sample gradients for all powers; noise/divide/SGD ordering; asynchronous profiling; and production diagnostic timing. Failure is explicit; no fallback path is provided.

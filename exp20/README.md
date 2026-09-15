# Exp20: five-seed paired power sweep

Unchanged A-only DP-KFC-A algorithm, powers `(0, .125, .25, .375, .5, .625, .75, 1)`;
paired seeds `(42, 7, 123, 2024, 3407)`. Initialization, private shuffle,
DP Gaussian noise and synthetic pink-noise schedules depend on seed, never power.
Forward-only builder, global RMS matching, Structured Ghost, optimizer, accounting,
and profiling are unchanged.

Formal execution uses cyclic power offsets `(0, 2, 4, 6, 1)` for the five seeds.
Each power occupies five distinct positions; per-position counts are at most one.
The order is recorded in `results/config.json` before execution and in final metadata.
Each `(p, seed)` runs in a fresh subprocess and rewrites all five epochs.
The default always runs all 40 jobs, with no resume logic. Existing aggregates are
moved to `previous_aggregate/` so stale reports are not mistaken for the new sweep.
Analysis requires all 40 runs and 200 epoch rows before writing aggregate results.

`summary.csv` contains one row per power/seed; `power_summary.csv` contains means
and sample standard deviations. `paired_summary.csv` aligns seeds against p=.5,
records individual differences and uses 20,000 paired-delta bootstrap resamples
(fixed RNG seed 20020) for percentile 95% intervals. Smoke sample std is undefined.
The report distinguishes final accuracy, accuracy AUC and clipping geometry, with
a dedicated .375 versus .5 comparison. Five seeds support descriptive, cautious
interpretation rather than strong significance claims. AUC is the trapezoidal
integral over observed epochs 1–5. Geometry metrics average epochs within seed
before seed aggregation. Plot error bars show sample std; paired plots show CIs.

```bash
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$PWD/exp20/.cache" MPLCONFIGDIR="$PWD/exp20/.cache/matplotlib" CUDA_CACHE_PATH="$PWD/exp20/.cache/cuda" conda run -n curve python -m pytest exp20/test_exp20.py -q -o cache_dir=exp20/.pytest_cache
conda run -n curve bash exp20/run_all.sh --smoke
conda run -n curve bash exp20/run_all.sh
```

Smoke runs seed 42 only, all eight powers, one epoch, one private and one synthetic
batch, sigma=0, under `results/smoke/`. All writes remain inside `exp20/`.
Algorithm time includes builder and private training, excludes diagnostic CPU
transfers and evaluation. Internal CUDA timings use deferred Events. Small runtime
fluctuations do not imply power-specific speedups.

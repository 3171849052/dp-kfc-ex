# ExpM1d: geometry refresh frequency

Copied from ExpM1c at `cd85dbdca8738a620f5765ea0b07c1364622ac9b`, with the same ViT-Tiny full fine-tuning, transforms, calibration construction, Poisson sampling, RDP accountant, BK global clipping, raw clipped optimizer signal, and matched A-only noise. Fixed beta=.1, lambda=1e-4, seed=42, epsilon=3, delta=1e-5, C=1; 5 × 195 steps.

| GPU | First run | Second run |
| --- | --- | --- |
| 0 | public four_per_epoch | pink once |
| 1 | pink four_per_epoch | public once |
| 2 | public twice_per_epoch | pink every_2_epochs |
| 3 | pink twice_per_epoch | public every_2_epochs |

Builds occur before private updates at zero-based global steps: once `{0}`, every_2_epochs `{0,390,780}`, twice_per_epoch at each epoch's local `{0,98}`, four_per_epoch at local `{0,49,98,147}`. Epoch numbers are one-based; refresh indices are zero-based. Geometry persists across epoch boundaries when no refresh is scheduled. Refresh preserves model, optimizer, accountant and noise generator. Calibration uses `seed+10000+epoch`, independent of refresh slot.

Every_epoch uses only the two existing `expm1c/results/lambda/vit_dp_kfm_a_{public,pink}_beta0.1_lambda0.0001_seed42` directories, read-only. They are neither copied nor submitted to workers.

Private oracle estimation and the inherited full geometry/alignment diagnostics occur once at each epoch's start, including epochs without a training refresh. Layer/group metrics aggregate all private steps as in ExpM1c. Oracle factors are discarded before private steps. `refresh.csv` compares each new training geometry against the preceding training geometry: Frobenius cosine and relative change for A and Q, then medians across affine layers. No oracle enters these comparisons. Identity blocks remain S=I.

For direct comparison, `geometry_build_seconds` retains ExpM1c's timing scope: calibration iteration and factor estimation, excluding Shape construction and research diagnostics. Total geometry build time sums this field. Training time excludes all within-epoch refresh work. Runtime follows ExpM1c's epoch-loop timing scope (including oracle, evaluation, and diagnostics; excluding initial model/data setup). The runtime fraction is factor-estimation time divided by that runtime. These are research diagnostics, not a DP release.

Analysis requires all completed runs and references and produces ten summary rows and five plots. Clipping, loss, accuracy and condition metrics follow the inherited final-epoch convention (geometry at epoch start); AUC follows ExpM1c's trapezoidal integration over epoch accuracies. Drift summaries take medians across refresh events after the first. Reference drift is NaN because its A/Q matrices were not saved; once also has no previous geometry. The drift plot uses the mean actual gap between builds and omits unavailable points. Single-seed differences are not statistically significant evidence.

All new outputs and caches stay under `expm1d/`. Datasets come only from root `data/` with `download=False`. The existing Exp30 pretrained checkpoint is read and copied into the local ExpM1d cache, as in ExpM1c. No resume, retry or dynamic scheduling.

Lightweight checks (no formal ViT training):

```bash
conda run --no-capture-output -n curve python -B expm1d/checks.py
bash -n expm1d/run_all.sh
```

Launch all eight runs; analysis runs only if all four queues succeed:

```bash
conda run --no-capture-output -n curve bash expm1d/run_all.sh
```

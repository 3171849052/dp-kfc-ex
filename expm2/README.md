# ExpM2: inverse DP-KFM-A

Copied from ExpM1c's fixed per-epoch ViT implementation. Only `expm2/` is written.
Beta remains positive: 0.1, 0.2, 0.3, 0.4; `geometry_power=inverse`.
Each beta has public and pink runs (seed 42, lambda 0.001), for exactly eight runs.
GPU 0/1/2/3 owns beta 0.1/0.2/0.3/0.4 respectively, public then pink serially.
The four queues run concurrently. Any queue failure prevents analysis and returns nonzero.
No resume, retry, checkpoint recovery or dynamic scheduling.

For each augmented affine activation covariance A, R=(A+lambda I)^(-beta),
tau=d_A/Tr(R), Q=tau R, S=Q tensor I_out. Metric factor is
(A+lambda I)^(beta/2)/sqrt(tau); noise factor is
sqrt(tau)(A+lambda I)^(-beta/2). Identity parameters use S=I.
One global sample coefficient clips the raw gradient; optimizer receives the
raw clipped sum plus matched noise divided by 256. Each layer has Tr(S)=d_layer;
expected noise energy is sigma² C² d_layer. No G is constructed or used.
A eigenvalue diagnostics describe undamped A; q extrema use damped A.
BK's inherited 1e-6 numerical stabilizer in the clipping denominator is retained.

Protocol: pretrained vit_tiny_patch16_224.augreg_in21k_ft_in1k, full fine-tuning,
AdamW lr=1e-4, weight decay=.01, betas=(.9,.999), eps=1e-8;
5 epochs, 195 Poisson steps/epoch, expected batch 256, physical batch 128,
C=1, epsilon=3, delta=1e-5. Geometry rebuilds at each epoch start.
Transforms, calibration, accountant, RNG streams, BK and all private oracle
diagnostics are inherited. Diagnostics are research-only, not a DP release.
Datasets are read from repository `data/` with download=False. Existing Exp30
pretrained weights are copied into the local cache as in ExpM1c.

Analysis reads eight matching `expm1c/results/beta/` forward runs and
`expm1/results/runs/vit_dp_sgd_none_seed42` without copying or modifying them.
It requires all runs complete and writes summary.csv, paired_summary.csv,
six plots and analysis.md below `expm2/results/`. Summaries use epoch-five
metrics and inherited epoch-1-to-5 accuracy AUC. Signed beta exists only for
visualization. For identical A, forward and inverse have equal condition
numbers and reversed spectrum ordering; their trained A may differ.
Comparisons report observed utility and clipping/update distortion without
claiming single-seed statistical significance.

Checks (CPU toy models only):

```bash
conda run --no-capture-output -n curve python -B expm2/checks.py
bash -n expm2/run_all.sh
```

Full experiment:

```bash
conda run --no-capture-output -n curve bash expm2/run_all.sh
```

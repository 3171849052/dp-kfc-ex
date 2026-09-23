# ExpM1 — matched-covariance DP-KFM

ExpM1 compares DP-SGD, the repository's standard DP-KFC, DP-KFM, and
DP-KFM-A under remove-one adjacency.  It is isolated from earlier experiments:
code, copied model cache, logs, results, matplotlib/CUDA caches, and temporary
files all live below `expm1/`.  Datasets are read-only below the repository
`data/` directory, and every torchvision dataset is opened with
`download=False`.

## Mechanisms

For affine layer `l`, geometry supplies bias-augmented activation factor `A_l`
and, for full methods, backpropagation factor `G_l`.  Damping is fixed at
`lambda=1e-3`.  DP-KFM uses

```
R_l(beta) = (A_l + lambda I)^beta kron (G_l + lambda I)^beta
```

and DP-KFM-A uses

```
R_l(beta) = (A_l + lambda I)^beta kron I.
```

Parameters without geometry use an identity block.  One global normalization,
never a per-layer normalization, sets

```
tau = d_total / sum_l trace(R_l),       S_l = tau R_l,
sum_l trace(S_l) = d_total.
```

Thus all beta values have matched total Gaussian energy
`E ||z||^2 = sigma^2 C^2 d_total`.  The formal beta grid is
`{0.25, 0.5, 0.75, 1.0}`.  Beta zero is not a formal condition; the CPU checks
verify exact degeneration to DP-SGD.

The BK implementation deliberately separates two branches.  The metric branch
uses `(A+lambda I)^(-beta/2)` and, for full DP-KFM, the corresponding G factor
inside the Ghost Gram identity.  It produces one global matched norm and one
clip factor per example.  The signal branch reconstructs both `sum_i g_i` and
`sum_i c_i g_i` from raw activations and raw backprops.  Metric factors never
enter the DP-KFM or DP-KFM-A optimization gradient.  Bias remains the final
constant-one activation coordinate; convolution uses unfold without forming
full per-example parameter gradients.

Noise is sampled from the matching covariance.  DP-KFM uses

```
sigma C sqrt(tau) (G+lambda I)^(beta/2) E (A+lambda I)^(beta/2),
```

DP-KFM-A omits the G transform, and identity blocks receive
`sigma C sqrt(tau) E`.  The noisy sum is divided by the logical batch size
before the task optimizer step.  DP-SGD uses raw BK clipping plus isotropic
noise.  DP-KFC preserves the repository baseline: transformed factors are used
for both its norm and aggregate, followed by transformed-space isotropic noise.
All methods use the same `q`, `sigma`, number of steps, and RDP accountant for a
given task.

## Fixed tasks

| | MNIST + SimpleCNN | CIFAR-10 + pretrained ViT-Tiny |
| --- | --- | --- |
| Private data | `data/MNIST` | `data/cifar-10-batches-py` |
| Matched public data | `data/FashionMNIST` | `data/stl10_binary` |
| Model | `dp_kfac.models.SimpleCNN` | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` |
| Epochs | 5 | 5 |
| Logical / physical batch | 256 / 256 | 256 / 128 |
| Optimizer | SGD, lr 0.5, no momentum/decay | AdamW, lr 1e-4, wd .01, betas (.9,.999), eps 1e-8 |
| Privacy | epsilon 1, delta 1e-5, C 1 | epsilon 3, delta 1e-5, C 1 |
| Private steps | 234/epoch, 1170 total | 195/epoch, 975 total |

MNIST and FashionMNIST use `ToTensor` followed by normalization with
mean `0.1307` and standard deviation `0.3081`.  CIFAR-10 and STL10 use 224x224
bicubic resize, `ToTensor`, and channel-wise mean/std `(0.5,0.5,0.5)`, with no
augmentation.  Public labels are retained, including STL10's real ten-class
labels.  Pink images use isolated epoch-local RNG state and independent uniform
labels.

Geometry is rebuilt before every epoch from exactly 10 batches of 256 public or
pink samples.  It never receives a current private training batch.  A fixed
10x256 private oracle sample with seed 23000 is read only for diagnostics and
never affects geometry selection, clipping, updates, or randomness.

The pretrained checkpoint is copied from the existing offline Exp30 Hugging
Face cache into `expm1/.cache/huggingface`; network access is disabled.  The
fixed protocol does not search for replacement data or checkpoints.  In this
checkout the pretrained cache is present, but `data/stl10_binary` is currently
absent.  `checks.py` and any ViT public run therefore fail explicitly until the
required STL10 dataset is placed at that exact path; nothing is downloaded or
substituted.

## Formal grid and launcher

Each task has 19 conditions: one source-independent DP-SGD, pink/public DP-KFC,
eight pink/public x beta DP-KFM, and eight corresponding DP-KFM-A conditions.
Every condition uses seeds 42, 7, and 123.  This is 57 runs per task and **114
formal runs in total**.

`run_all.sh` contains a fixed round-robin mapping with exactly 38 runs on each
physical GPU 1, 2, and 3 (19 MNIST and 19 ViT per GPU).  The three GPU workers
run concurrently; runs assigned to one GPU execute serially, each in a fresh
`python -B expm1/worker.py` process whose internal device is `cuda:0`.  Before
launch, one non-training preflight process rejects any pre-existing artifact,
constructs MNIST/FashionMNIST/CIFAR-10/STL10 with `download=False`, checks their
sizes and ten-class labels, and stages/verifies the existing local ViT
checkpoint.  Only after that succeeds are formal directories created.  The
launcher never resumes or overwrites a formal run.  If any worker fails, all
worker statuses are collected and analysis is not started.

After all 114 `complete.json` markers exist, `analyze.py` validates the entire
grid before writing anything.  It checks five exact epochs per run, dataset and
batch protocol, optimizer/noise/accountant step counts, privacy parameters,
global trace normalization, factor availability, DP-KFM-A's absence of G,
ViT group coverage, and all research-only flags.  Missing or inconsistent data
causes a nonzero failure without a partial summary.  Analysis is staged under
`expm1/results/` and published only after every CSV and plot is produced.

The six tabular outputs are:

- `all_metrics.csv`
- `final_summary.csv`
- `beta_summary.csv`
- `paired_summary.csv`
- `geometry_summary.csv`
- `layer_group_summary.csv`

`paired_summary.csv` retains each seed's delta and reports mean plus sample
standard deviation for KFM-SGD, KFM-KFC, KFM-A-KFM, and public-pink comparisons.
Plots cover MNIST and ViT accuracy versus beta, epoch accuracy curves, clipping,
clipping and update distortion, layer/group SNR, oracle A/G mismatch versus the
KFM-A-KFM utility gap, geometry anisotropy, and runtime/memory.

## Research-only diagnostics

Per-example matched/raw norms, clip factors, raw and clipped aggregate
distortion, noisy-update distortion, layer/group SNR, and private-oracle A/G
alignment are unnoised private diagnostics.  Every corresponding row is marked
`research_only=true`.  They are for controlled research analysis only and must
not be described or released as differentially private outputs.  Utility and
privacy-accountant fields do not make those accompanying diagnostic rows safe
for release.

With conda environment `curve`, the only requested validation commands are:

```bash
conda run --no-capture-output -n curve python -B expm1/checks.py
bash -n expm1/run_all.sh
```

These checks do not launch the formal grid.  The full experiment command is:

```bash
conda run --no-capture-output -n curve bash expm1/run_all.sh
```

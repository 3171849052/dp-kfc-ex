# ExpM1

ExpM1 is the fixed comparison experiment for MNIST + SimpleCNN and CIFAR-10 + pretrained ViT-Tiny. It compares DP-SGD, DP-KFC, DP-KFM, and DP-KFM-A with pink and matched public geometry. The formal grid uses only seed `42`, beta `{0.25, 0.5, 0.75, 1.0}`, remove-one adjacency, the BK backend, conda environment `curve`, and physical GPUs `1,2,3`. There are 19 runs per task and 38 total runs, statically assigned 13/13/12.

Private data are MNIST and CIFAR-10. Matched public data are FashionMNIST and CIFAR100 images. CIFAR100 semantic labels are never used: full KFC/KFM use independent deterministic 10-way pseudo-labels only to form G, seeded by `seed + PUBLIC_LABEL_SEED_OFFSET + epoch`. DP-KFM-A builds A only and does not create, access, or backpropagate through G. Every torchvision dataset uses `download=False`; all data are read from repository-root `data/`.

The BK implementation performs one model forward and one backward per physical chunk. It retains raw activations and backpropagations, computes clipping factors from the chosen metric, and reconstructs the clipped raw aggregate analytically without a second backward or full per-example parameter-gradient tensor. Linear, convolution, LayerNorm, positional embedding, and class-token gradients use the cached values.

Diagnostics distinguish raw clipping (`clip_cos`, `clip_rel_error`), mechanism signal (`signal_cos`, `signal_rel_error`), and noisy private update (`update_cos`, `update_rel_error`). DP-SGD, DP-KFM, and DP-KFM-A send the raw clipped aggregate to the optimizer; DP-KFC sends its preconditioned clipped aggregate. SNR uses the actual clipped signal sent to the optimizer.

Private sampling is fixed-step Poisson/Bernoulli inclusion with expected batch size 256, `q = 256/N`, 234 steps per MNIST epoch, and 195 steps per ViT epoch. Realized batch sizes may vary, physical chunks are at most 256 for MNIST and 128 for ViT, and every step adds Gaussian noise and advances the matching RDP accountant. The noisy sum is always divided by expected batch size 256.

KFM geometry uses one symmetric eigendecomposition per factor per epoch. Shape diagnostics report the beta-dependent actual `S_beta` condition, log-eigenvalue spread, and effective rank, with global trace normalization. DP-SGD uses lightweight identity semantics with zero operator and factor state.

All generated outputs are under `expm1/`. Run the checks before any formal experiment:

```bash
conda run --no-capture-output -n curve python -B expm1/checks.py
bash -n expm1/run_all.sh
```

Formal results have not been run unless the output files are present and complete. The launcher refuses incomplete or existing formal artifacts and never resumes or downloads data.

For the interrupted grid at `b731c71`, audit with `python -B expm1/status.py`.
A complete run requires its config, exactly five metrics rows (epochs 1–5),
geometry and layer-group rows covering epochs 1–5, and a completion marker
with five epochs and the task's full accountant/optimizer/noise counts.
The initial audit is 18 complete / 20 incomplete; `--assert-initial` checks
that exact remaining set. The remaining launcher accepts progress within
that set on later invocations and fails if any originally complete run is
now incomplete.

After checks pass, explicitly launch remaining jobs with:

```bash
conda run --no-capture-output -n curve bash expm1/run_remaining.sh
```

This launcher preserves complete directories and logs, deletes only checked
incomplete directories/logs, and restarts those runs from epoch 1. It keeps
`cfg.GPU_RUNS` unchanged, runs each physical GPU's jobs serially, and runs
GPUs 1/2/3 in parallel. Per-run output goes to `expm1/logs/<run_name>.log`.
A lock prevents overlapping remaining launchers. A failed worker stops its
GPU queue and prevents analysis; completed jobs on other GPUs are retained.
There is no checkpoint resume. `run_all.sh` still requires empty formal outputs.

Only after every worker succeeds and a fresh audit confirms 38/38 complete
will analysis run. Existing aggregate CSVs, plots, and `analysis.log` do not
block this launcher. Analysis validates all runs and stages every output
before atomically replacing each aggregate file; per-run results are untouched.

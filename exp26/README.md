# Exp26

Run from the repository root:

```bash
conda run -n curve bash exp26/run_all.sh
```

This always reruns the complete 32-run grid and overwrites matching results;
there is no automatic skip/resume or second-stage search. Decide in the shell
whether to launch it when results already exist.

The current `scripts/paper/exp_distilbert_sst2.py` supplies the complete training
implementation, prompt MLM, data, structural synthetic geometry, BK clipping,
Adam (default weight_decay=0, no scheduler), and RDP accounting. Only base/none
and full/synthetic run, each at physical batch 128 and logical batch 1024.
Exp26 uses profile=False and collect_diagnostics=True: clipping/norm diagnostics
are retained without timing, CUDA peak-memory tracking, or profiling counters.
Completed run CSVs and summary.csv keep only hyperparameter-search fields.
No controlled batch-16 run or Explicit run is launched. The shell uses the
calling environment directly; the launch command above selects curve once.

Each combination resets seed 42 and creates a fresh model and optimizer, visiting
all 67,349 training examples for each of three epochs. C controls both global
per-example clipping and the pre-averaging Gaussian standard deviation sigma*C.
Sigma uses only epsilon=3, delta=1e-5, logical sample rate 1024/67349, and 198
logical steps. Each epoch has 66 logical steps and 527 physical steps; the whole
run has 198 optimizer/noise/accountant steps and 1581 physical steps. The
reference's shuffled-batch accounting convention is preserved unchanged.

Per-run CSVs contain three epoch rows. As in the reference, validation runs only
after epoch 3, so earlier accuracy/test_loss fields are NaN. Completed CSVs add
C, learning_rate and noise_std to the reference diagnostics. summary.csv contains
one row per completed combination: final-epoch validation and clipping/norm
diagnostics, with step counts accumulated over all three epochs. Per-epoch CSV
step counts describe that epoch; epsilon_spent is cumulative in both outputs.

Analysis writes both per-method rankings and ranking.csv, plus four heatmaps
(accuracy and clip_fraction for each method). Accuracy is a fraction in [0,1].
Ties use lower validation loss, then C and LR for deterministic ordering.

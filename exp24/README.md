# Exp24 — frozen ViT-Tiny DP linear probing

Exp24 loads `vit_tiny_patch16_224.augreg_in21k_ft_in1k`, converts the timm
model to explicit patch, q/k/v, and output Linear maps, and replaces the
1000-class classifier with a seeded `Linear(192, 10)`.  The complete
pretrained backbone is frozen.  The only trainable parameters are exactly
`head.weight` and `head.bias` (1,930 values), and the model remains in
`eval()` mode during private training.

The fixed comparison is `dp_sgd`, `dp_kfc_a_bk`, and `dp_kfc`, with seeds
42, 7, 123, 2024, and 3407.  Training is five epochs of CIFAR-10, using
AdamW with lr `1e-3`, weight decay `0.01`, and logical/physical batches
`256/256`.  Preprocessing is precisely Resize `(224,224)` with BICUBIC,
ToTensor, and Normalize `(0.5,0.5,0.5)` / `(0.5,0.5,0.5)`.

Only the trainable head participates in geometry:

- DP-SGD uses identity geometry.
- A-only uses `z=[CLS feature,1]`, `A` of shape `(193,193)`,
  `U_A=scale*(A+1e-3 I)^(-0.4)`, and Exp20 RMS matching.
- Full KFC uses `A` `(193,193)`, `G` `(10,10)`, and
  `g_tilde=U_G g U_A` with inverse square roots and damping `1e-3`.

The Ghost/BK path computes the global head transformed-gradient norm and
clipped aggregate without creating a `[B,10,193]` tensor.  Synthetic KFAC
calibration streams pink-noise images (`alpha=1`, `(3,224,224)`) in ten
batches of 256.  Full KFC uses continuous uniform synthetic labels; all
synthetic and private/noise RNG streams are isolated and paired by seed.

Opacus RDP computes the noise multiplier from epsilon `8`, delta `1e-5`,
sample rate `256/50000`, and the exact 975-step formal schedule.  Smoke
calibration uses its exact two-step schedule.  Every private logical batch
has one backward call, one clipped aggregate, one noise event, one AdamW
step, and one accountant step.

Run the tests and smoke validation with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m pytest \
  exp24/test_exp24.py -q -o cache_dir=exp24/.pytest_cache
rm -rf exp24/results/smoke
conda run -n curve bash exp24/run_all.sh --smoke
conda run -n curve env PYTHONPATH=src:. python -B exp24/analyze.py --smoke
```

Formal runs are intentionally not started by this implementation task.
Existing formal run directories are protected from overwrite with
`FileExistsError`.  Outputs belong under `results/runs/`; smoke outputs belong
under `results/smoke/`.

# Exp22 — pretrained ViT-Tiny full-parameter DP fine-tuning

The production model is loaded with
`timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)`.
The ImageNet classifier is replaced by a seeded 10-class head. All parameters
are trainable, including patch embedding, all 12 transformer blocks, final
LayerNorm, position embedding and CLS token. No randomly initialized backbone,
model fallback, automatic batch reduction or frozen layers are used.

The [timm model card](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)
describes the ImageNet-21k pretraining and ImageNet-1k fine-tuning. CIFAR-10 train
and test images use `resolve_model_data_config` / `create_transform` with
`is_training=False`: deterministic bicubic resize/center crop to 224 and the
checkpoint's normalization. The resolved configuration is saved with each run.

For Exp21-style BK/Ghost, the non-overlapping patch convolution is copied into
an equivalent Linear over `unfold` patches. Each packed QKV is split into three
Linears; timm's attention, GELU MLP, residuals, LayerNorms and CLS pooling remain
intact. These are fixed, weight-preserving representation conversions. All 74
Linears receive geometry in the KFC methods. LayerNorm uses analytic identity
geometry. Position and CLS gradients are recovered from the per-example
backprop at the input of `pos_drop`, with identity geometry for both tokens.
Tests compare converted pretrained logits with timm and BK with independent
single-example backward gradients.

## Fixed protocol

- Methods: `dp_sgd`, `dp_kfc_a_bk`, `dp_kfc`; seeds: `42, 7, 123, 2024, 3407`.
- CIFAR-10; 5 epochs; learning rate 1e-3; epsilon 3; delta 1e-5; clipping C=1.
- AdamW: betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01.
- Damping 1e-3; A-only power 0.4 with existing RMS matching.
- Logical batch 256; **fixed physical batch 8**, 32 chunks per logical batch.
  This replaces the old local physical batch 256 setting explicitly for the
  224px model; there is no runtime batch adaptation. Evaluation also uses 8.
- Each epoch rebuilds geometry from 10 batches of 256 synthetic images,
  processed in fixed chunks of 32. Pink noise uses alpha=1.0 at 224x224;
  full KFC uses uniformly random synthetic labels.
- Initialization and private shuffle use `seed`; synthetic images use
  `seed+10000+epoch`, labels `seed+20000+epoch`, DP noise `seed+40000`.
  Methods with the same seed have identical initial weights, sample order and
  Gaussian RNG stream. Synthetic RNG is isolated from the private RNG.
- Exactly one DP noise event, AdamW step and accountant step per logical batch.
  The existing RDP fixed-logical-batch accounting convention is retained
  (`shuffle=True`, `drop_last=True`, sample rate 256/50000 in formal runs).

Linear sample gradients are never materialized as `[B,Dout,Din]`. Norms use
Ghost Gram tiles of 8, and clipped aggregates are reconstructed from factors.
`bk_cache_bytes` records retained activation/backprop storage;
`temporary_per_sample_grad_bytes` records temporary BK workspace;
`fallback_temporary_grad_bytes` is always zero. Epoch-local counters are
`logical_steps`, `optimizer_steps`, `noise_events`, and `backward_calls`;
`accountant_steps` and `epsilon` are cumulative. Full calibration has 80 VJP
calls and 2560 reverse vectors; A-only has 80 forwards and no reverses.

## Validation

All commands use conda environment `curve` (tested with timm 1.0.29). Cache,
dataset, temporary test files and outputs stay under `exp22/`.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -B -m pytest \
  exp22/test_exp22.py -q -o cache_dir=exp22/.pytest_cache \
  --basetemp=exp22/.cache/pytest
conda run --no-capture-output -n curve bash exp22/run_all.sh \
  --smoke --output exp22/results/pretrained_smoke
conda run --no-capture-output -n curve python -B -m exp22.analyze \
  --smoke --output exp22/results/pretrained_smoke
```

Smoke runs the real pretrained model and all three methods, seed 42, one epoch,
two logical batches and 256 test images. It performs full synthetic calibration
and uses nonzero noise calibrated for the smoke schedule. It validates the
pipeline, not convergence. Existing random-TinyViT results are not comparable
and are left untouched. Use a separate fresh output directory for formal runs.
Formal experiments are never started by tests or smoke.

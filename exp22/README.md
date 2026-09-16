# Exp22 — pretrained ViT-Tiny full-parameter DP fine-tuning

- Pretrained model: `vit_tiny_patch16_224.augreg_in21k_ft_in1k`.
- Pretraining: ImageNet-21k pretrain + ImageNet-1k fine-tune.
- Downstream: CIFAR-10 full-model DP fine-tuning, all 5,526,346 parameters trainable.
- Input: 224x224 RGB; ViT-Tiny/16, embed_dim=192, 12 blocks, CLS-token pooling.
- Epochs: **10**. Logical/physical batch: **256 / 256**.
- Optimizer: **AdamW lr=1e-4**, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01.

The production backbone is loaded with
`timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)`.
See the [timm model card](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)
for checkpoint provenance. There is no head-only training, LoRA, adapter,
freezing, layer-wise LR decay, warmup, LR schedule or method-specific LR.

## Exact explicit conversion

`model.py` copies the patch Conv2d into `patch_embed: Linear(768,192)` over
`unfold` patches in the original channel/kernel order. Each packed QKV becomes
`attn.q_proj`, `attn.k_proj`, `attn.v_proj`; `attn.proj` becomes `attn.out_proj`.
`mlp.fc1`, `mlp.fc2`, norm1/norm2, CLS/position parameters and final norm are
copied. The explicit 1000-class model is compared against the pretrained timm
source before testing the seeded replacement `Linear(192,10)` CIFAR head.

There are exactly **74 Linear maps** and **25 LayerNorms**. The same seed gives
identical backbone and head for all methods. Across seeds the pretrained
backbone is unchanged and the CIFAR head differs. Pooling is `norm(x)[:,0]`.

A parameter-free `token_hook` sits immediately after CLS concatenation and
position addition. Its `[B,197,192]` backprop gives the per-example position
gradient and the first token gives the CLS gradient. Both enter the same
global clipping norm and clipped reconstruction with identity geometry.

Train and test preprocessing is exactly `Resize((224,224), BICUBIC)`,
`ToTensor()`, `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`. No random augmentation
or center crop is used.

## Fixed DP and geometry protocol

Methods: `dp_sgd`, `dp_kfc_a_bk`, `dp_kfc`; seeds: `42, 7, 123, 2024, 3407`.
Epsilon=3, delta=1e-5, clipping C=1, damping=1e-3, A-only power=0.4.
Private shuffle and head initialization use `seed`; DP noise uses
`seed+40000`. Private shuffling uses a separate generator and `drop_last=True`.

Each logical batch is exactly one physical chunk: one backward call, one DP
noise event, one optimizer step and one accountant step. Formal runs have
`50000//256=195` steps/epoch and **1950 total steps**. The RDP noise multiplier
is recalculated from q=256/50000, steps=1950, epsilon=3, delta=1e-5. The existing
fixed-logical-batch RDP accounting convention is retained.

- DP-SGD: identity geometry.
- A-only: `U_A = scale*(A+1e-3 I)^(-0.4)`, `U_G=I`, existing Exp20 RMS matching.
- Full KFC: `U_A=(A+1e-3 I)^(-1/2)`, `U_G=(G+1e-3 I)^(-1/2)`,
  `g_tilde=U_G g U_A`, without extra full-KFC scaling.

Each epoch rebuilds the preconditioner for all 74 Linears. Synthetic images
are deterministic streaming pink noise, alpha=1.0, shape=(3,224,224): generate
one batch, update A/G, release it, then generate the next. There is no cache of
all ten image batches. A dedicated image generator uses `seed+10000+epoch`;
uniform synthetic labels use `seed+20000+epoch`. Both streams are continuous
within an epoch and isolated from private RNG. Synthetic logical and physical
batch sizes are both 256, with 10 batches / 2560 samples each epoch.

| Builder counter | DP-SGD | A-only | Full KFC |
| --- | ---: | ---: | ---: |
| forward calls | 0 | 10 | 10 |
| logical batches | 0 | 10 | 10 |
| VJP calls | 0 | 0 | 10 |
| reverse vectors | 0 | 0 | 2560 |
| samples | 0 | 2560 | 2560 |

## BK and metrics

The Exp21-style FP32 BK/Ghost path does not materialize `[B,Dout,Din]`
per-example Linear gradients. Norms use Ghost Gram tiles; aggregation uses
one vectorized einsum and takes bias from the augmented last column. There
is one logical parameter aggregate. LayerNorm gradients are analytic.

`layer_strategies`: Linear=`bk_ghost`, LayerNorm=`identity_analytic`,
position/CLS=`identity_direct`. Groups are `attention_qkv`, `attention_out`,
`mlp`, `patch_head`, `identity`.

`bk_cache_bytes` measures retained activation/backprop storage;
`temporary_per_sample_grad_bytes` measures temporary BK workspace;
`fallback_temporary_grad_bytes` is zero. CUDA peak allocated/reserved memory
is recorded per epoch. Logical/optimizer/noise/backward counters are
epoch-local; accountant steps and epsilon are cumulative.

The batch sizes are fixed. CUDA OOM propagates as an error; no fallback,
automatic batch reduction, dynamic model change or swallowed exception is used.

## Validation and hygiene

All commands use conda environment `curve` (timm 1.0.29). Tests cover pretrained
1000-class logits, patch/QKV conversion, exact module counts, all-Linear
geometry, CLS/position clipping, seed pairing, a per-example autograd oracle,
streaming RNG/release, no sample-weight allocation, step counters, privacy
calibration, and single/multiple-seed analysis. The isolated algebraic
Conv/Linear equivalence test uses FP64 to avoid FP32 kernel reduction-order
noise; pretrained logits and private BK checks remain FP32.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m pytest \
  exp22/test_exp22.py -q -o cache_dir=exp22/.pytest_cache
conda run -n curve bash exp22/run_all.sh --smoke
conda run -n curve env PYTHONPATH=src:. python -B exp22/analyze.py --smoke
```

Smoke uses seed=42, one epoch, two logical batches, 512 private samples and
256 test images. It uses the real model, full synthetic calibration and nonzero
DP noise calibrated for its two-step schedule. Each method must report
logical/backward/optimizer/noise/accountant steps all equal to 2, finite and
updated parameters, empty BK caches and zero fallback workspace. Accuracy is
reported only as a smoke diagnostic; it is not used to tune hyperparameters.

Current smoke CSV/JSON is under `results/smoke/`. Earlier `pretrained_smoke/`
and `formal_bs256/` outputs describe old configurations and are not validation
of this protocol. Preserve CSV/JSON; remove `results/*.log`, `.cache/`,
`.pytest_cache/` and local `__pycache__/` after validation. These paths are
ignored. Pretrained weights are downloaded again after cache cleanup.

Tests and smoke do not start formal experiments.

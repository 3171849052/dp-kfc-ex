"""Exp34 fixed 20-epoch protocol; shared constants from Exp22."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp34 import ROOT

# Keep the original clip_norm=1 experiment intact; this is the clip_norm=2
# rerun requested for Exp34.
RESULTS = ROOT / "results_clip2"
DATA_ROOT = ROOT.parent / "exp30" / "data"
SEED = 42
EPOCHS = 20
METHODS = ("dp_adamw", "dp_kfc", "dp_kfc_a")
EXP22_METHOD = {"dp_adamw": "dp_sgd", "dp_kfc": "dp_kfc", "dp_kfc_a": "dp_kfc_a_bk"}
LABELS = {"dp_adamw": "DP-AdamW", "dp_kfc": "DP-KFC", "dp_kfc_a": "DP-KFC-A"}
DAMPING_VALUES = (1e-3, 1e-2, 1e-1)
# Two-card schedule: keep each damping pair sequential and run the baseline
# after the 0.1 pair on the second card.
GPU_RUNS = {
    0: (("dp_kfc", 1e-3), ("dp_kfc_a", 1e-3),
        ("dp_kfc", 1e-2), ("dp_kfc_a", 1e-2)),
    1: (("dp_kfc", 1e-1), ("dp_kfc_a", 1e-1), ("dp_adamw", None)),
}
# Exp34's imported default is 1; this rerun intentionally uses clip_norm=2.
MAX_GRAD_NORM = 2.0
LOGICAL_STEPS_PER_EPOCH = TRAIN_SAMPLES // LOGICAL_BATCH_SIZE
TOTAL_STEPS = EPOCHS * LOGICAL_STEPS_PER_EPOCH


def run_name(method, damping):
    return method if method == "dp_adamw" else f"{method}_damping_{damping:g}"


def grid():
    return [(gpu, method, damping) for gpu, runs in GPU_RUNS.items() for method, damping in runs]

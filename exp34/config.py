"""Exp34 fixed 20-epoch protocol; shared constants from Exp22."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp34 import ROOT

RESULTS = ROOT / "results"
SEED = 42
EPOCHS = 20
METHODS = ("dp_adamw", "dp_kfc", "dp_kfc_a")
EXP22_METHOD = {"dp_adamw": "dp_sgd", "dp_kfc": "dp_kfc", "dp_kfc_a": "dp_kfc_a_bk"}
LABELS = {"dp_adamw": "DP-AdamW", "dp_kfc": "DP-KFC", "dp_kfc_a": "DP-KFC-A"}
DAMPING_VALUES = (1e-3, 1e-2, 1e-1)
GPU_RUNS = {i: (("dp_kfc", d), ("dp_kfc_a", d)) for i, d in enumerate(DAMPING_VALUES)}
GPU_RUNS[3] = (("dp_adamw", None),)
LOGICAL_STEPS_PER_EPOCH = TRAIN_SAMPLES // LOGICAL_BATCH_SIZE
TOTAL_STEPS = EPOCHS * LOGICAL_STEPS_PER_EPOCH


def run_name(method, damping):
    return method if method == "dp_adamw" else f"{method}_damping_{damping:g}"


def grid():
    return [(gpu, method, damping) for gpu, runs in GPU_RUNS.items() for method, damping in runs]

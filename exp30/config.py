"""Only damping varies; all training constants come from Exp22."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp30 import ROOT

RESULTS = ROOT / "results"
SEED = 42
METHODS = ("dp_kfc", "dp_kfc_a")
EXP22_METHOD = {"dp_kfc": "dp_kfc", "dp_kfc_a": "dp_kfc_a_bk"}
LABELS = {"dp_kfc": "DP-KFC", "dp_kfc_a": "DP-KFC-A"}
DAMPING_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
GPU_DAMPINGS = {
    0: DAMPING_VALUES[0:2],
    1: DAMPING_VALUES[2:4],
    2: DAMPING_VALUES[4:6],
    3: DAMPING_VALUES[6:7],
}


def run_name(method, damping):
    return f"{method}_damping_{damping:g}"


def grid():
    return [(gpu, method, damping) for gpu, dampings in GPU_DAMPINGS.items()
            for damping in dampings for method in METHODS]

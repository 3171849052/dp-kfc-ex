"""Fixed Exp30 training protocol and the four Exp33 formal methods."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp33 import ROOT

RESULTS = ROOT / "results"
SEED = 42
SYNTHETIC_PHYSICAL_BATCH_SIZE = 128
METHODS = ("dp_adamw", "dp_wiener_a", "dp_wiener_full", "dp_kfc_a_01")
GPU_METHODS = dict(enumerate(METHODS))
LABELS = dict(zip(METHODS, ("DP-AdamW", "DP-Wiener-A", "DP-Wiener-Full", "DP-KFC-A-0.1")))
DAMPING = 0.1

def grid():
    return list(GPU_METHODS.items())

"""Exp33 protocol with six fixed Wiener-A shrinkage strengths."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp33c import ROOT
DATA_ROOT = ROOT.parent / "exp30" / "data"

RESULTS = ROOT / "results"
SEED = 42
SYNTHETIC_PHYSICAL_BATCH_SIZE = 128
GAMMAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
EXP33_DP_ADAMW_ACCURACY = 0.9276
EXP33_WIENER_A_ACCURACY = 0.7180
UPDATE_GROUPS = ("patch_head", "attention_qkv", "attention_out", "mlp", "identity")
RUNS = {
    f"wiener_a_gamma_{gamma:g}": {"gamma": gamma, "gpu": gpu}
    for gamma, gpu in zip(GAMMAS, (0, 1, 2, 3, 0, 1))
}
GPU_RUNS = {gpu: tuple(name for name, spec in RUNS.items() if spec["gpu"] == gpu) for gpu in range(4)}


def grid():
    return [(gpu, name) for gpu, names in GPU_RUNS.items() for name in names]

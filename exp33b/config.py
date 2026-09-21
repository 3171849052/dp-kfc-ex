"""Exp33 protocol with eight fixed Wiener-A scale / Linear LR runs."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp33b import ROOT
DATA_ROOT = ROOT.parent / "exp30" / "data"

RESULTS = ROOT / "results"
SEED = 42
SYNTHETIC_PHYSICAL_BATCH_SIZE = 128
IDENTITY_LR = 1e-4
LINEAR_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
SCALE_MODES = ("none", "rms_match")
EXP33_DP_ADAMW_ACCURACY = 0.9276
UPDATE_GROUPS = ("patch_head", "attention_qkv", "attention_out", "mlp", "identity")
RUNS = {
    f"wiener_a_{tag}_lr_{label}": {"linear_lr": lr, "scale_mode": mode, "gpu": gpu}
    for gpu, (lr, label) in enumerate(zip(LINEAR_LRS, ("1e-4", "3e-4", "1e-3", "3e-3")))
    for mode, tag in (("none", "none"), ("rms_match", "rms"))
}
GPU_RUNS = {gpu: tuple(name for name, spec in RUNS.items() if spec["gpu"] == gpu) for gpu in range(4)}


def grid():
    return [(gpu, name) for gpu, names in GPU_RUNS.items() for name in names]

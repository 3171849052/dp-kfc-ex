"""Exp33 protocol with seven fixed Wiener-A noise temperatures."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp33d import ROOT
DATA_ROOT = ROOT.parent / "exp30" / "data"

RESULTS = ROOT / "results"
SEED = 42
SYNTHETIC_PHYSICAL_BATCH_SIZE = 128
BETAS = (1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
ADAM_BETAS = (0.9, 0.999)
UPDATE_GROUPS = ("patch_head", "attention_qkv", "attention_out", "mlp", "identity")
RUNS = {"dp_adamw": {"beta": None, "gpu": 3}}
RUNS.update({name: {"beta": beta, "gpu": 3} for name, beta in zip(
    ("wiener_a_beta_1", "wiener_a_beta_1e-2", "wiener_a_beta_1e-3",
     "wiener_a_beta_1e-4", "wiener_a_beta_1e-5", "wiener_a_beta_1e-6", "wiener_a_beta_1e-7"), BETAS)})


def grid():
    return [(3, name) for name in RUNS]

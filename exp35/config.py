"""Fixed four-run grids. No sweep or resume settings."""
from exp35 import ROOT, REPO
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    ADAM_EPS, EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    LOGICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE, ACCUMULATION_STEPS,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
DATA_ROOT = REPO / 'data'
RESULTS = ROOT / 'results' / 'vit'
SEED = 42
GPU = 2
DAMPING = 1e-3
VARIANTS = ('raw', 'norm', 'mix05')
METHODS = ('dp_adamw',) + tuple('dp_kfc_a_' + v for v in VARIANTS)
EXP22_METHOD = {m: 'dp_sgd' if m == 'dp_adamw' else 'dp_kfc_a_bk' for m in METHODS}

def grid(dataset):
    baseline = 'dp_adam' if dataset == 'mnist' else 'dp_adamw'
    return (baseline,) + tuple('dp_kfc_a_' + v for v in VARIANTS)

def run_name(method, damping):
    return method

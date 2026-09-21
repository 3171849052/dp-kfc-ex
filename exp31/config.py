"""Fixed Exp31 protocol and seven runs."""
from exp22.config import (
    MODEL_NAME, IMG_SIZE, EPOCHS, WEIGHT_DECAY, BETAS, ADAM_EPS,
    EPSILON, DELTA, MAX_GRAD_NORM, A_POWER, TRAIN_SAMPLES,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE, SYNTHETIC_ALPHA,
    SYNTHETIC_PHYSICAL_BATCH_SIZE,
)
from exp31 import ROOT
RESULTS = ROOT / 'results'
SEED = 42
LEARNING_RATE = 1e-3
LOGICAL_BATCH_SIZE = PHYSICAL_BATCH_SIZE = 256
ACCUMULATION_STEPS = 1
LORA_RANK = LORA_ALPHA = 8
LORA_DROPOUT = 0.0
DAMPING_VALUES = (1e-3, 1e-2, 1e-1)
METHODS = ('dp_adamw', 'dp_kfc', 'dp_kfc_a')
EXP22_METHOD = {'dp_adamw': 'dp_sgd', 'dp_kfc': 'dp_kfc', 'dp_kfc_a': 'dp_kfc_a_bk'}
LABELS = {'dp_adamw': 'DP-AdamW-LoRA', 'dp_kfc': 'DP-KFC-LoRA', 'dp_kfc_a': 'DP-KFC-A-LoRA'}
GPU_RUNS = {i: [('dp_kfc', d), ('dp_kfc_a', d)] for i, d in enumerate(DAMPING_VALUES)}
GPU_RUNS[3] = [('dp_adamw', None)]

def run_name(method, damping):
    return method + ('_lora' if damping is None else f'_lora_damping_{damping:g}')

def grid():
    return [(gpu, method, damping) for gpu, runs in GPU_RUNS.items() for method, damping in runs]

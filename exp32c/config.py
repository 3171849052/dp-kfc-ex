"""Fixed seven-run protocol."""
from exp32c import ROOT
RESULTS = ROOT / 'results'
MODEL_NAME = 'squeezenet1_1'
WEIGHTS = None
DATA_ROOT = ROOT.parent / 'exp32' / 'data'
SEED = 42
EPOCHS = 5
IMG_SIZE = 224
NUM_CLASSES = 10
TRAIN_SAMPLES = 50000
EPSILON = 3.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
LOGICAL_BATCH_SIZE = PHYSICAL_BATCH_SIZE = 256
ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
A_POWER = 0.4
A_SCALE = 1.0
SYNTHETIC_BATCHES = 10
SYNTHETIC_BATCH_SIZE = SYNTHETIC_PHYSICAL_BATCH_SIZE = 256
SYNTHETIC_ALPHA = 1.0
DAMPING_VALUES = (1e-3, 1e-2, 1e-1)
METHODS = ('dp_adam', 'dp_kfc', 'dp_kfc_a')
LABELS = dict(dp_adam='DP-Adam', dp_kfc='DP-KFC', dp_kfc_a='DP-KFC-A')
GPU_RUNS = {i: [('dp_kfc', d), ('dp_kfc_a', d)] for i, d in enumerate(DAMPING_VALUES)}
GPU_RUNS[3] = [('dp_adam', None)]

def run_name(method, damping):
    return method + ('' if damping is None else f'_damping_{damping:g}')

def grid():
    return [(gpu, method, damping) for gpu, runs in GPU_RUNS.items() for method, damping in runs]

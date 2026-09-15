"""Fixed Exp21 protocol."""
from exp20.config import (SEEDS, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM,
                         WEIGHT_DECAY, EPSILON, DELTA, MAX_GRAD_NORM, DAMPING,
                         SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE)
POWER = .25
METHODS = ('exact', 'fast2', 'ghost2', 'bk', 'bk_gd')
METHOD_ORDER = {s: METHODS[i:]+METHODS[:i] for i, s in enumerate(SEEDS)}
FORMAL_RUN_COUNT = 25

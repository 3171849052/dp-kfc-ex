"""Exactly the Exp14 settings; synthetic RMS matching is fixed for every beta."""
from exp14b.config import (BETAS, SEEDS, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MOMENTUM, WEIGHT_DECAY, EPSILON, DELTA, MAX_GRAD_NORM, DAMPING,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE)

ESTIMATORS = ('KFLR', 'KFRA-block')

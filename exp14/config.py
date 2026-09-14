"""Exp13 training settings; beta is the only experimental variable."""
from exp13.config import (SEEDS, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM,
    WEIGHT_DECAY, EPSILON, DELTA, MAX_GRAD_NORM, DAMPING,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE)

BETAS = (0., .25, .5, .75, 1.)

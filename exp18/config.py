"""Fixed Exp14b protocol, one seed, ten lightweight variants."""
from exp14b.config import (EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM,
    WEIGHT_DECAY, EPSILON, DELTA, MAX_GRAD_NORM, DAMPING,
    SYNTHETIC_BATCHES, SYNTHETIC_BATCH_SIZE)
SEEDS = (42,)
BETA = .25
OVERSAMPLING = 4
METHODS = ('diag', 'a_only', 'c_only', 'fullA_diagC', 'diagA_fullC',
           'rank4', 'rank8', 'rank16', 'refresh2', 'frozen')


def rebuild(method, epoch):
    return epoch == 0 or (method != 'frozen' and (method != 'refresh2' or epoch % 2 == 0))

"""Fixed exp10 configuration, with no scale bounds."""
from exp10.run_exp10 import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPSILON,
    DELTA, MAX_GRAD_NORM, SYNTHETIC_BATCH_SIZE, PROBES, TAU,
    PRECONDITIONER_BATCHES, SEEDS, LAYERS,
)
METHODS = ("DP-SGD", "Full-Fisher Equil", "Module-Elementwise Equil",
           "Factorized Equil", "Layer-Scalar Equil")
STRUCTURED = METHODS[2:]

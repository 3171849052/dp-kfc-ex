"""Fixed paired MNIST protocol."""
METHODS = ('M0_original', 'M1_full_ghost', 'M2_a_ghost_p05', 'M3_a_ghost_p025')
SEEDS = (42, 7, 123, 2024, 3407)
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = .5
MOMENTUM = 0
WEIGHT_DECAY = 0
EPSILON = 1
DELTA = 1e-5
MAX_GRAD_NORM = 1.
DAMPING = 1e-3
SYNTHETIC_BATCHES = 10
SYNTHETIC_BATCH_SIZE = 256

ORDER = {seed: tuple(METHODS[i] for i in indices) for seed, indices in {
    42: (0, 1, 2, 3), 7: (3, 2, 1, 0), 123: (1, 3, 0, 2),
    2024: (2, 0, 3, 1), 3407: (0, 2, 1, 3)}.items()}

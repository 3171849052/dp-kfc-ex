"""Fixed Exp19 protocol, five paired seeds and eight activation powers."""
POWERS = (0., .125, .25, .375, .5, .625, .75, 1.)
METHODS = POWERS
SEEDS = (42, 7, 123, 2024, 3407)
# Evenly spaced cyclic offsets: each power occupies five distinct positions.
POWER_ORDER = {seed: POWERS[offset:]+POWERS[:offset]
               for seed, offset in zip(SEEDS, (0, 2, 4, 6, 1))}
FORMAL_RUN_COUNT = len(POWERS)*len(SEEDS)
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

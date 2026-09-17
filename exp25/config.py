from pathlib import Path
from functools import lru_cache
from opacus.accountants.utils import get_noise_multiplier

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 7, 91, 23, 58)
CONDITIONS = ('dp_sgd_bk_gd',) + tuple(
    f'{prefix}_{source}_bk_gd'
    for prefix in ('dp_kfc_a', 'dp_kfc') for source in ('public', 'pink', 'oracle'))
EPOCHS, BATCH_SIZE, TRAIN_SAMPLES = 5, 256, 50000
STEPS_PER_EPOCH = TRAIN_SAMPLES // BATCH_SIZE
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
SAMPLE_RATE = BATCH_SIZE / TRAIN_SAMPLES
EPSILON, DELTA, CLIP, LR = 8., 1e-5, 1., 1e-3
DAMPING, A_POWER = 1e-3, .4

@lru_cache(None)
def noise_multiplier():
    return get_noise_multiplier(target_epsilon=EPSILON, target_delta=DELTA,
        sample_rate=SAMPLE_RATE, steps=TOTAL_STEPS, accountant='rdp', epsilon_tolerance=1e-4)

def condition(name):
    if name not in CONDITIONS:
        raise ValueError(name)
    if name == CONDITIONS[0]:
        return 'identity', 'none'
    return ('a_only' if name.startswith('dp_kfc_a_') else 'full'), name.split('_')[-3]

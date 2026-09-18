"""Fixed protocol; only C and learning rate vary."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "exp27" / "results"
C_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0)
LEARNING_RATES = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3)
METHODS = (("dp_kfc", "DP-KFC", "full"), ("dp_kfc_a", "DP-KFC-A", "a_only"))
EPOCHS = 5
BATCH_SIZE = 256
EPSILON = 2.0
DELTA = 1e-5
SEED = 42
DAMPING = 1e-3
A_POWER = 0.4
PINK_ALPHA = 1.0
SOURCE = "pink"
ENGINE = "bk"
STEP_FIELDS = ("logical_steps", "optimizer_steps", "noise_events", "accountant_steps")


def grid():
    for c in C_VALUES:
        for lr in LEARNING_RATES:
            for slug, method, geometry in METHODS:
                yield c, lr, slug, method, geometry

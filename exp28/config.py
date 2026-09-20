"""Single-seed DP-KFC-A C × learning-rate search on DistilBERT/SST-2."""
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "exp28" / "results"
C_VALUES = (0.5, 1.0, 2.0, 4.0)
LEARNING_RATES = (1e-5, 1e-4, 5e-4, 1e-3)
EPSILON = 3.0
DELTA = 1e-5
SEED = 42
EPOCHS = 3
LOGICAL_BATCH_SIZE = 1024
PHYSICAL_BATCH_SIZE = 128
GEOMETRY_BATCH_SIZE = 256
GEOMETRY_PHYSICAL_BATCH_SIZE = 16
DAMPING = 1e-3
A_POWER = 0.4
MAX_LENGTH = 128
GEOMETRY = "a_only"
SOURCE = "synthetic"
ENGINE = "bk"
STEP_FIELDS = ("logical_steps", "optimizer_steps", "noise_events", "accountant_steps")
METRICS = ("accuracy", "test_loss", "train_loss", "clip_fraction", "mean_clip_factor",
           "norm_p50", "norm_p90", "norm_p99", "norm_max", "epsilon_spent",
           "noise_multiplier", "noise_std")


def grid():
    return product(C_VALUES, LEARNING_RATES)


def run_dir(c, lr):
    return RESULTS / "runs" / f"C{c:g}_LR{lr:g}"


def ranked(frame):
    result = frame.sort_values(["accuracy", "test_loss", "C", "learning_rate"],
                               ascending=[False, True, True, True]).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result

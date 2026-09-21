"""Single-seed DP-KFC-A damping sweep on DistilBERT/SST-2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "exp29" / "results"
C = 2.0
LEARNING_RATE = 5e-4
DAMPING_VALUES = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
EPSILON = 3.0
DELTA = 1e-5
SEED = 42
EPOCHS = 3
LOGICAL_BATCH_SIZE = 1024
PHYSICAL_BATCH_SIZE = 128
GEOMETRY_BATCH_SIZE = 256
GEOMETRY_PHYSICAL_BATCH_SIZE = 16
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
    return iter(DAMPING_VALUES)


def run_dir(damping):
    return RESULTS / "runs" / f"damping_{damping:g}"


def ranked(frame):
    result = frame.sort_values(["accuracy", "test_loss", "damping"],
                               ascending=[False, True, True]).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result

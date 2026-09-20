"""Single-variable, single-seed damping experiment."""

from pathlib import Path

EPSILON = 3.0
DELTA = 1e-5
EPOCHS = 3
SEED = 42
LEARNING_RATE = 5e-4
CLIP_NORM = 2.0
LOGICAL_BATCH_SIZE = 1024
PHYSICAL_BATCH_SIZE = 128
ENGINE = "bk"
GEOMETRY = "full"
SOURCE = "synthetic"
GEOMETRY_BATCH_SIZE = 256
GEOMETRY_PHYSICAL_BATCH_SIZE = 16
MAX_LENGTH = 128
DAMPINGS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
RESULTS = Path(__file__).resolve().parent / "results"

RESULT_FIELDS = [
    "method", "geometry", "source", "engine", "damping", "learning_rate", "C",
    "seed", "epoch", "accuracy", "test_loss", "train_loss",
    "clip_fraction", "mean_clip_factor", "norm_p50", "norm_p90", "norm_p99", "norm_max",
    "epsilon_target", "epsilon_spent", "delta", "noise_multiplier", "noise_std",
    "sample_rate", "logical_batch_size", "physical_batch_size",
    "logical_steps", "physical_steps", "optimizer_steps", "noise_events", "accountant_steps",
]

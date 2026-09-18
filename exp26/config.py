"""Fixed, single-seed coarse grid. No profiling comparison or second stage."""

from pathlib import Path

EPSILON = 3.0
DELTA = 1e-5
EPOCHS = 3
SEED = 42
LOGICAL_BATCH_SIZE = 1024
PHYSICAL_BATCH_SIZE = 128
ENGINE = "bk"
DAMPING = 1e-3
GEOMETRY_BATCH_SIZE = 256
GEOMETRY_PHYSICAL_BATCH_SIZE = 16
MAX_LENGTH = 128

METHODS = {"DP-Adam": ("base", "none"), "DP-KFC": ("full", "synthetic")}
CLIP_NORMS = [0.5, 1.0, 2.0, 4.0]
LEARNING_RATES = [1e-5, 1e-4, 5e-4, 1e-3]
RESULTS = Path(__file__).resolve().parent / "results"

# Shared schema for completed run CSVs, summary, and downstream rankings.
RESULT_FIELDS = [
    "method", "geometry", "source", "engine", "C", "learning_rate",
    "seed", "epoch", "accuracy", "test_loss", "train_loss",
    "clip_fraction", "mean_clip_factor", "norm_p50", "norm_p90", "norm_p99", "norm_max",
    "epsilon_target", "epsilon_spent", "delta", "noise_multiplier", "noise_std",
    "sample_rate", "logical_batch_size", "physical_batch_size",
    "logical_steps", "physical_steps", "optimizer_steps", "noise_events", "accountant_steps",
]

"""Delegate every complete training run to the unchanged paper reference."""
from contextlib import contextmanager
from math import ceil

import pandas as pd
import torch

from exp29 import config as cfg
from scripts.paper import exp_distilbert_sst2 as reference


@contextmanager
def reference_constants(damping):
    values = dict(MAX_GRAD_NORM=cfg.C, DELTA=cfg.DELTA, EPOCHS=cfg.EPOCHS,
                  LOGICAL_BATCH_SIZE=cfg.LOGICAL_BATCH_SIZE,
                  BK_PHYSICAL_BATCH_SIZE=cfg.PHYSICAL_BATCH_SIZE,
                  GEOMETRY_BATCH_SIZE=cfg.GEOMETRY_BATCH_SIZE,
                  GEOMETRY_PHYSICAL_BATCH_SIZE=cfg.GEOMETRY_PHYSICAL_BATCH_SIZE,
                  DAMPING=damping, A_POWER=cfg.A_POWER, MAX_LENGTH=cfg.MAX_LENGTH)
    previous = {key: getattr(reference, key) for key in values}
    try:
        for key, value in values.items():
            setattr(reference, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(reference, key, value)


def run_one(train, validation, tokenizer, damping, device):
    output = cfg.run_dir(damping)
    output.mkdir(parents=True, exist_ok=True)
    with reference_constants(damping):
        # Reference creates a fresh model, optimizer, loader RNG, noise RNG and
        # accountant. MAX_GRAD_NORM controls both BK clipping and sigma * C noise.
        row = reference.run_one(
            train=train, validation=validation, tokenizer=tokenizer,
            geometry=cfg.GEOMETRY, source=cfg.SOURCE, engine=cfg.ENGINE,
            epsilon=cfg.EPSILON, seed=cfg.SEED, epochs=cfg.EPOCHS, device=device,
            output_dir=output, physical_batch_size=cfg.PHYSICAL_BATCH_SIZE,
            profile=False, profile_mode="practical", lr=cfg.LEARNING_RATE, collect_diagnostics=True,
        )
    assert row["method"] == "DP-KFC-A" and row["epoch"] == cfg.EPOCHS
    steps = cfg.EPOCHS * ceil(len(train) / cfg.LOGICAL_BATCH_SIZE)
    assert all(row[key] == steps for key in cfg.STEP_FIELDS)
    row.update(C=cfg.C, learning_rate=cfg.LEARNING_RATE,
               noise_std=row["noise_multiplier"] * cfg.C,
               damping=damping, effective_damping=damping + 1e-5, a_power=cfg.A_POWER, collect_diagnostics=True)
    return row


def main():
    assert torch.cuda.is_available(), "Exp29 requires CUDA"
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    # Protect completed and partial experiment results; no automatic resume.
    assert not (cfg.RESULTS / "summary.csv").exists(), "Use a fresh Exp29 results directory"
    assert not (cfg.RESULTS / "runs").exists(), "Use a fresh Exp29 results directory"
    with reference_constants(cfg.DAMPING_VALUES[0]):
        tokenizer = reference.AutoTokenizer.from_pretrained(reference.MODEL_NAME)
        train, validation = reference.load_data(tokenizer)
    assert len(train) == 67349 and len(validation) == 872
    summaries = []
    for index, damping in enumerate(cfg.grid(), 1):
        print(f"[{index}/5] DP-KFC-A damping={damping:g} C={cfg.C:g} "
              f"LR={cfg.LEARNING_RATE:g} seed={cfg.SEED}", flush=True)
        summaries.append(run_one(train, validation, tokenizer, damping, torch.device("cuda")))
        frame = pd.DataFrame(summaries)
        frame.to_csv(cfg.RESULTS / "summary.csv", index=False)
        cfg.ranked(frame).to_csv(cfg.RESULTS / "ranking.csv", index=False)


if __name__ == "__main__":
    main()

"""Run five independent full DP-KFC runs using the current BK reference."""

import pandas as pd
import torch

from scripts.paper import exp_distilbert_sst2 as reference
from exp26c import config as cfg


def run_one(train, validation, tokenizer, damping, device):
    tag = f"damping{damping:g}"
    path = cfg.RESULTS / (
        f"{cfg.GEOMETRY}_{cfg.SOURCE}_{cfg.ENGINE}_{tag}_eps{cfg.EPSILON:g}_seed{cfg.SEED}.csv"
    )
    previous_damping = reference.DAMPING
    reference.DAMPING = damping
    try:
        # Reference resets seed, model, Adam, loader RNG, noise RNG and accountant.
        summary = reference.run_one(
            train, validation, tokenizer, cfg.GEOMETRY, cfg.SOURCE, cfg.ENGINE,
            cfg.EPSILON, cfg.SEED, cfg.EPOCHS, device, cfg.RESULTS,
            cfg.PHYSICAL_BATCH_SIZE, profile=False, profile_mode=tag,
            lr=cfg.LEARNING_RATE, collect_diagnostics=True,
        )
    finally:
        reference.DAMPING = previous_damping
    rows = pd.read_csv(path)
    assert rows["epoch"].tolist() == list(range(1, cfg.EPOCHS + 1))
    rows["damping"] = damping
    rows["C"] = cfg.CLIP_NORM
    rows["learning_rate"] = cfg.LEARNING_RATE
    rows["noise_std"] = rows["noise_multiplier"] * cfg.CLIP_NORM
    # Discard the reference's disabled (NaN) profiling columns.
    rows[cfg.RESULT_FIELDS].to_csv(path, index=False)
    summary.update(damping=damping, C=cfg.CLIP_NORM, learning_rate=cfg.LEARNING_RATE,
                   noise_std=summary["noise_multiplier"] * cfg.CLIP_NORM)
    return {key: summary[key] for key in cfg.RESULT_FIELDS}


def main():
    reference.DELTA = cfg.DELTA
    reference.MAX_GRAD_NORM = cfg.CLIP_NORM
    reference.LOGICAL_BATCH_SIZE = cfg.LOGICAL_BATCH_SIZE
    reference.GEOMETRY_BATCH_SIZE = cfg.GEOMETRY_BATCH_SIZE
    reference.GEOMETRY_PHYSICAL_BATCH_SIZE = cfg.GEOMETRY_PHYSICAL_BATCH_SIZE
    reference.MAX_LENGTH = cfg.MAX_LENGTH
    device = torch.device("cuda")
    tokenizer = reference.AutoTokenizer.from_pretrained(reference.MODEL_NAME)
    train, validation = reference.load_data(tokenizer)
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for damping in cfg.DAMPINGS:
        print(f"Exp26c DP-KFC: damping={damping:g}, C={cfg.CLIP_NORM:g}, "
              f"LR={cfg.LEARNING_RATE:g}, seed={cfg.SEED}", flush=True)
        summaries.append(run_one(train, validation, tokenizer, damping, device))
        summary = pd.DataFrame(summaries, columns=cfg.RESULT_FIELDS)
        # Damping must not change privacy calibration or batching/event counts.
        invariant_fields = [
            "sample_rate", "noise_multiplier", "noise_std", "epsilon_spent",
            "logical_batch_size", "physical_batch_size", "logical_steps",
            "physical_steps", "optimizer_steps", "noise_events", "accountant_steps",
        ]
        assert (summary[invariant_fields].nunique() == 1).all()
        summary.to_csv(cfg.RESULTS / "summary.csv", index=False)


if __name__ == "__main__":
    main()

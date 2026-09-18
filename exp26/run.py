"""Run all 32 complete runs using the current paper implementation verbatim."""

from itertools import product

import pandas as pd
import torch

from scripts.paper import exp_distilbert_sst2 as reference
from exp26 import config as cfg


def run_one(train, validation, tokenizer, method, clip_norm, learning_rate, device):
    geometry, source = cfg.METHODS[method]
    # run_one uses this tag in its own per-epoch CSV filename, so even its
    # intermediate writes cannot collide with any other grid combination.
    tag = f"search_{method}_C{clip_norm:g}_LR{learning_rate:g}"
    path = cfg.RESULTS / (
        f"{geometry}_{source}_{cfg.ENGINE}_{tag}_eps{cfg.EPSILON:g}_seed{cfg.SEED}.csv"
    )
    previous_clip_norm = reference.MAX_GRAD_NORM
    reference.MAX_GRAD_NORM = clip_norm
    try:
        # The same C is read by bk_clip and by Gaussian noise addition.
        # The reference resets RNGs, model, Adam, loader and accountant per run.
        # Keep epoch-wide diagnostics without timing/memory profiling.
        summary = reference.run_one(
            train, validation, tokenizer, geometry, source, cfg.ENGINE,
            cfg.EPSILON, cfg.SEED, cfg.EPOCHS, device, cfg.RESULTS,
            cfg.PHYSICAL_BATCH_SIZE, profile=False, profile_mode=tag,
            lr=learning_rate, collect_diagnostics=True,
        )
    finally:
        reference.MAX_GRAD_NORM = previous_clip_norm
    rows = pd.read_csv(path)
    rows["C"] = clip_norm
    rows["learning_rate"] = learning_rate
    rows["noise_std"] = rows["noise_multiplier"] * clip_norm
    rows[cfg.RESULT_FIELDS].to_csv(path, index=False)
    summary.update(C=clip_norm, learning_rate=learning_rate,
                   noise_std=summary["noise_multiplier"] * clip_norm)
    return {key: summary[key] for key in cfg.RESULT_FIELDS}


def main():
    reference.DELTA = cfg.DELTA
    reference.LOGICAL_BATCH_SIZE = cfg.LOGICAL_BATCH_SIZE
    reference.DAMPING = cfg.DAMPING
    reference.GEOMETRY_BATCH_SIZE = cfg.GEOMETRY_BATCH_SIZE
    reference.GEOMETRY_PHYSICAL_BATCH_SIZE = cfg.GEOMETRY_PHYSICAL_BATCH_SIZE
    reference.MAX_LENGTH = cfg.MAX_LENGTH
    # Use CUDA directly: no smaller-batch or CPU fallback on failure.
    device = torch.device("cuda")
    tokenizer = reference.AutoTokenizer.from_pretrained(reference.MODEL_NAME)
    train, validation = reference.load_data(tokenizer)
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method, clip_norm, learning_rate in product(
        cfg.METHODS, cfg.CLIP_NORMS, cfg.LEARNING_RATES
    ):
        print(f"Exp26 {method}: C={clip_norm:g}, LR={learning_rate:g}, "
              f"seed={cfg.SEED}, physical_batch={cfg.PHYSICAL_BATCH_SIZE}", flush=True)
        summaries.append(run_one(
            train, validation, tokenizer, method, clip_norm, learning_rate, device
        ))
        pd.DataFrame(summaries).to_csv(cfg.RESULTS / "summary.csv", index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only formal run audit; no torch, datasets, or worker imports."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1 import config as cfg


# Snapshot supplied for b731c71. Later launches may complete a subset of these.
INITIAL_INCOMPLETE = frozenset("""
mnist_dp_sgd_none_seed42
mnist_dp_kfm_pink_beta0.25_seed42
mnist_dp_kfm_pink_beta1_seed42
mnist_dp_kfm_public_beta0.75_seed42
mnist_dp_kfm_a_pink_beta0.5_seed42
mnist_dp_kfm_a_public_beta0.25_seed42
mnist_dp_kfm_a_public_beta1_seed42
vit_dp_sgd_none_seed42
vit_dp_kfc_public_seed42
vit_dp_kfm_pink_beta0.25_seed42
vit_dp_kfm_pink_beta0.75_seed42
vit_dp_kfm_pink_beta1_seed42
vit_dp_kfm_public_beta0.5_seed42
vit_dp_kfm_public_beta0.75_seed42
vit_dp_kfm_a_pink_beta0.25_seed42
vit_dp_kfm_a_pink_beta0.5_seed42
vit_dp_kfm_a_pink_beta1_seed42
vit_dp_kfm_a_public_beta0.25_seed42
vit_dp_kfm_a_public_beta0.75_seed42
vit_dp_kfm_a_public_beta1_seed42
""".split())


@dataclass(frozen=True)
class RunStatus:
    spec: cfg.RunSpec
    reasons: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.reasons


def inspect_run(spec: cfg.RunSpec, runs_root: Path | None = None) -> RunStatus:
    root = cfg.RESULTS_ROOT / "runs" if runs_root is None else runs_root
    directory = root / spec.name
    reasons = []
    if not directory.is_dir():
        return RunStatus(spec, ("missing directory",))
    # Never follow per-run symlinks for subsequent cleanup.
    if directory.is_symlink():
        return RunStatus(spec, ("symlink run directory",))
    for filename in ("config.json", "complete.json"):
        path = directory / filename
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("expected JSON object")
            if filename == "complete.json":
                steps = cfg.task_config(spec.task).accountant_steps
                for key, expected in dict(epochs=5, accountant_steps=steps,
                                          optimizer_steps=steps, noise_events=steps).items():
                    if payload.get(key) != expected:
                        reasons.append(f"complete.json {key}={payload.get(key)!r}, expected {expected}")
        except FileNotFoundError:
            reasons.append(f"missing {filename}")
        except (OSError, ValueError) as error:
            reasons.append(f"invalid {filename}: {error}")
    expected_epochs = set(range(1, 6))
    for filename in ("metrics.csv", "geometry.csv", "layer_groups.csv"):
        try:
            with (directory / filename).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            epochs = {int(row["epoch"]) for row in rows}
            if epochs != expected_epochs:
                reasons.append(f"{filename} epochs={sorted(epochs)}, expected 1..5")
            if filename == "metrics.csv" and len(rows) != 5:
                reasons.append(f"metrics epochs={len(rows)}/5 rows")
        except FileNotFoundError:
            reasons.append(f"missing {filename}")
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            reasons.append(f"invalid {filename}: {error}")
    return RunStatus(spec, tuple(reasons))


def scan(runs_root: Path | None = None) -> tuple[RunStatus, ...]:
    assert len(cfg.FORMAL_GRID) == 38
    return tuple(inspect_run(spec, runs_root) for spec in cfg.FORMAL_GRID)


def validate_remaining(statuses: tuple[RunStatus, ...], *, initial: bool = False) -> None:
    actual = {row.spec.name for row in statuses if not row.complete}
    unexpected = actual - INITIAL_INCOMPLETE
    completed_since = INITIAL_INCOMPLETE - actual
    if unexpected or (initial and completed_since):
        raise RuntimeError(
            "remaining-set regression mismatch\n"
            f"unexpected incomplete: {sorted(unexpected)}\n"
            f"expected incomplete but now complete: {sorted(completed_since)}"
        )


def require_complete(statuses: tuple[RunStatus, ...]) -> None:
    complete_count = sum(row.complete for row in statuses)
    incomplete_count = len(statuses) - complete_count
    assert len(statuses) == 38 and complete_count == 38 and incomplete_count == 0, (
        f"analysis refused: {complete_count}/38 complete, {incomplete_count} incomplete"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-remaining", action="store_true")
    parser.add_argument("--assert-initial", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    statuses = scan()
    count = sum(row.complete for row in statuses)
    print(f"formal: {len(statuses)}\ncomplete: {count}\nincomplete: {len(statuses) - count}")
    for complete, label in ((True, "COMPLETE"), (False, "INCOMPLETE/MISSING")):
        print(f"\n{label}:")
        for row in statuses:
            if row.complete == complete:
                suffix = "" if complete else ": " + "; ".join(row.reasons)
                print(row.spec.name + suffix)
    if args.validate_remaining or args.assert_initial:
        validate_remaining(statuses, initial=args.assert_initial)
    if args.require_complete:
        require_complete(statuses)


if __name__ == "__main__":
    main()

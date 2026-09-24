"""Restart only incomplete runs assigned to one original physical GPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1 import config as cfg
from expm1.status import inspect_run, scan, validate_remaining


def cleanup_incomplete(spec, runs_root=None, logs_root=None) -> bool:
    """Recheck immediately before deletion; a complete run and its log are untouched."""
    runs_root = cfg.RESULTS_ROOT / "runs" if runs_root is None else runs_root
    logs_root = cfg.LOGS_ROOT if logs_root is None else logs_root
    if inspect_run(spec, runs_root).complete:
        return False
    directory = runs_root / spec.name
    log = logs_root / f"{spec.name}.log"
    if directory.is_symlink() or log.is_symlink():
        raise RuntimeError(f"refusing symlink cleanup: {spec.name}")
    if directory.exists():
        shutil.rmtree(directory)
    log.unlink(missing_ok=True)
    return True


def run_gpu(gpu: int) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    for spec in cfg.GPU_RUNS[gpu]:
        if not cleanup_incomplete(spec):
            print(f"COMPLETE (untouched): {spec.name}", flush=True)
            continue
        command = [sys.executable, "-B", str(cfg.ROOT / "worker.py"),
                   "--task", spec.task, "--method", spec.method,
                   "--source", spec.source, "--seed", str(spec.seed)]
        if spec.beta is not None:
            command += ["--beta", f"{spec.beta:g}"]
        print(f"RESTART epoch 1: {spec.name} (physical GPU {gpu})", flush=True)
        cfg.LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        with (cfg.LOGS_ROOT / f"{spec.name}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = inspect_run(spec)
        if not result.complete:
            raise RuntimeError(f"worker exited without completing {spec.name}: {result.reasons}")


def preflight() -> None:
    statuses = scan()
    validate_remaining(statuses)
    if all(row.complete for row in statuses):
        return
    from expm1 import data, vit
    mnist_train, mnist_test = data.mnist_datasets()
    fashion = data.fashion_mnist_public_dataset()
    cifar_train, cifar_test = data.cifar10_datasets()
    assert (len(mnist_train), len(mnist_test), len(fashion)) == (60_000, 10_000, 60_000)
    assert len(mnist_train.classes) == len(fashion.classes) == 10
    assert (len(cifar_train), len(cifar_test)) == (50_000, 10_000)
    assert len(data.cifar100_public_dataset().classes) == 100
    checkpoint = vit.prepare_checkpoint()
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--gpu", type=int, choices=cfg.PHYSICAL_GPUS)
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        validate_remaining(scan())
        run_gpu(args.gpu)

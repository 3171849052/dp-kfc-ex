#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TMPDIR="$PWD/exp36/.cache/tmp"
mkdir -p "$TMPDIR" exp36/logs
python -B exp36/train.py > exp36/logs/train.log 2>&1
python -B exp36/diagnose.py > exp36/logs/diagnose.log 2>&1
python -B exp36/analyze.py > exp36/logs/analyze.log 2>&1

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp35/.cache/tmp"
mkdir -p "$TMPDIR" exp35/logs
python -B exp35/worker.py
python -B exp35/analyze.py > exp35/logs/analyze.log 2>&1

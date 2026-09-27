#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
python -B expm1d/launch.py
python -B expm1d/analyze.py

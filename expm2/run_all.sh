#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
python -B expm2/launch.py
python -B expm2/analyze.py

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/exp17/results/matplotlib"
export XDG_CACHE_HOME="$PWD/exp17/results/cache"
export TMPDIR="$PWD/exp17/results/tmp"
mkdir -p "$TMPDIR"
python exp17/run_exp17.py
python exp17/analyze_exp17.py

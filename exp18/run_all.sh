#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/exp18/.cache"
export MPLCONFIGDIR="$PWD/exp18/.cache/matplotlib"
export TORCH_HOME="$PWD/exp18/.cache/torch"
export CUDA_CACHE_PATH="$PWD/exp18/.cache/cuda"
export TMPDIR="$PWD/exp18/.cache/tmp"
mkdir -p "$TMPDIR" exp18/logs
python -u exp18/run_exp18.py "$@" 2>&1 | tee "exp18/logs/run$(if [[ " $* " == *' --smoke '* ]]; then echo _smoke; fi).log"

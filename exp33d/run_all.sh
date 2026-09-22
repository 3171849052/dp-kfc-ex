#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp33d/.cache/tmp" TMP="$PWD/exp33d/.cache/tmp" TEMP="$PWD/exp33d/.cache/tmp"
mkdir -p exp33d/logs exp33d/results/runs "$TMPDIR"
python -B - <<'PY'
from exp33d.run import load_data
from exp33d import config as cfg
for method in cfg.RUNS:
    if (cfg.RESULTS / 'runs' / method).exists():
        raise FileExistsError(method)
load_data()
PY
for method in dp_adamw wiener_a_beta_1 wiener_a_beta_1e-2 wiener_a_beta_1e-3 wiener_a_beta_1e-4 wiener_a_beta_1e-5 wiener_a_beta_1e-6 wiener_a_beta_1e-7; do
    python -B exp33d/run.py --method "$method" >"exp33d/logs/${method}.log" 2>&1
done
python -B exp33d/analyze.py

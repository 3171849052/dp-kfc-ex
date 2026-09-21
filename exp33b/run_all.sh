#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp33b/.cache/tmp" TMP="$PWD/exp33b/.cache/tmp" TEMP="$PWD/exp33b/.cache/tmp"
mkdir -p exp33b/logs exp33b/results/runs "$TMPDIR"
python -B - <<'PY'
from exp33b.run import load_data
from exp33b import config as cfg
for method in cfg.RUNS:
    if (cfg.RESULTS / 'runs' / method).exists():
        raise FileExistsError(method)
load_data(download=False)
PY
pids=()
for gpu in 0 1 2 3; do
    python -B exp33b/worker.py --gpu "$gpu" >"exp33b/logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
if [[ "$status" -ne 0 ]]; then
    echo 'Exp33b worker failed; see exp33b/logs/gpu*.log' >&2
    exit "$status"
fi
python -B exp33b/analyze.py

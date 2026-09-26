#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
# Python package initialization routes all caches into expm1b/.cache.
python -B - <<'PY'
from expm1b import config as cfg, data, vit
for run in cfg.FORMAL_GRID:
    assert not (cfg.RESULTS_ROOT / 'runs' / run.name).exists(), run.name
for task in cfg.TASKS:
    data.private_datasets(task)
    data.public_dataset(task)
vit.prepare_checkpoint()
PY
mkdir -p expm1b/logs expm1b/results/runs
run_gpu() (
    export CUDA_VISIBLE_DEVICES="$1"
    while IFS=$'\t' read -r task method source beta seed name; do
        python -B expm1b/worker.py --task "$task" --method "$method" \
            --source "$source" --beta "$beta" --seed "$seed" \
            2>&1 | tee "expm1b/logs/$name.log"
    done < <(python -B - "$1" <<'PY'
import sys
from expm1b.config import GPU_RUNS
for run in GPU_RUNS[int(sys.argv[1])]:
    print(run.task, run.method, run.source, run.beta, run.seed, run.name, sep='\t')
PY
    )
)
pids=()
for gpu in 0 1 2 3; do
    run_gpu "$gpu" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if (( status )); then
    echo "A GPU worker failed; analysis was not started." >&2
    exit 1
fi
python -B expm1b/analyze.py 2>&1 | tee expm1b/logs/analysis.log

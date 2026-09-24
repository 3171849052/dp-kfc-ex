#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
export XDG_CACHE_HOME="$PWD/expm1/.cache"
export MPLCONFIGDIR="$PWD/expm1/.cache/matplotlib"
export HF_HOME="$PWD/expm1/.cache/huggingface"
export HF_HUB_CACHE="$PWD/expm1/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PWD/expm1/.cache/huggingface/hub"
export HF_XET_CACHE="$PWD/expm1/.cache/huggingface/xet"
export TORCH_HOME="$PWD/expm1/.cache/torch"
export CUDA_CACHE_PATH="$PWD/expm1/.cache/cuda"
export TRITON_CACHE_DIR="$PWD/expm1/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/expm1/.cache/torchinductor"
export NUMBA_CACHE_DIR="$PWD/expm1/.cache/numba"
export TMPDIR="$PWD/expm1/.cache/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Hold this lock through analysis so two remaining launchers cannot clean a live run.
mkdir -p expm1/.cache/tmp
exec 9>expm1/.cache/remaining.lock
flock -n 9 || { echo "Another remaining launcher is active." >&2; exit 1; }

python -B expm1/status.py --validate-remaining
python -B expm1/remaining.py --preflight

run_gpu() (
    export CUDA_VISIBLE_DEVICES="$1"
    python -B expm1/remaining.py --gpu "$1"
)

run_gpu 1 &
pid_1=$!
run_gpu 2 &
pid_2=$!
run_gpu 3 &
pid_3=$!

worker_status=0
for pid in "$pid_1" "$pid_2" "$pid_3"; do
    if ! wait "$pid"; then
        worker_status=1
    fi
done
if (( worker_status != 0 )); then
    echo "At least one remaining worker failed; analysis was not started." >&2
    exit 1
fi

# Re-scan after all workers succeed, before touching aggregate outputs or analysis.log.
python -B expm1/status.py --require-complete
mkdir -p expm1/logs
python -B expm1/analyze.py 2>&1 | tee expm1/logs/analysis.log

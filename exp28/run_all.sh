#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/exp28/.mplconfig"
export XDG_CACHE_HOME="$PWD/exp28/.cache"
export HF_HOME="$PWD/exp28/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$PWD/exp28/.cache/torch"
export TRITON_CACHE_DIR="$PWD/exp28/.cache/triton"
export CUDA_CACHE_PATH="$PWD/exp28/.cache/cuda"
export TMPDIR="$PWD/exp28/.tmp"
mkdir -p exp28/logs "$TMPDIR"
python -u -B -m exp28.run 2>&1 | tee exp28/logs/run.log
python -u -B -m exp28.analyze 2>&1 | tee exp28/logs/analyze.log

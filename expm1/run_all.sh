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

# Refuse the whole launch before starting any worker if even one formal artifact
# already exists.  Formal runs are intentionally neither resumed nor replaced.
python -B - <<'PY'
from expm1 import config as cfg
from expm1 import data
from expm1 import vit

runs = tuple(cfg.formal_runs())
assert len(runs) in (13, 12)
assert {gpu: len(cfg.GPU_RUNS[gpu]) for gpu in cfg.PHYSICAL_GPUS} == {1: 13, 2: 13, 3: 12}
assert {run.gpu for run in runs} == {1, 2, 3}
targets = []
for run in runs:
    targets.extend((cfg.RESULTS_ROOT / "runs" / run.name, cfg.LOGS_ROOT / f"{run.name}.log"))
targets.append(cfg.LOGS_ROOT / "analysis.log")
targets.extend(
    cfg.RESULTS_ROOT / name
    for name in (
        "all_metrics.csv",
        "final_summary.csv",
        "beta_summary.csv",
        "paired_summary.csv",
        "geometry_summary.csv",
        "layer_group_summary.csv",
        "mnist_accuracy_vs_beta.png",
        "vit_accuracy_vs_beta.png",
        "final_accuracy_curves.png",
        "clipping_vs_beta.png",
        "clip_distortion_vs_beta.png",
        "signal_distortion_vs_beta.png",
        "update_distortion_vs_beta.png",
        "layer_group_snr.png",
        "oracle_error_utility_gap.png",
        "geometry_anisotropy_vs_beta.png",
        "runtime_memory_comparison.png",
    )
)
collisions = [str(path) for path in targets if path.exists()]
assert not collisions, "formal output exists; refusing resume/overwrite:\n" + "\n".join(collisions)

# Construct every fixed dataset with download=False before creating formal
# output directories.  All fixed inputs must already exist; no download is allowed.
mnist_train, mnist_test = data.mnist_datasets()
fashion = data.fashion_mnist_public_dataset()
cifar_train, cifar_test = data.cifar10_datasets()
assert (len(mnist_train), len(mnist_test), len(fashion)) == (60_000, 10_000, 60_000)
assert (len(cifar_train), len(cifar_test)) == (50_000, 10_000)
assert len(mnist_train.classes) == len(fashion.classes) == 10
cifar100 = data.cifar100_public_dataset()
assert len(cifar100.classes) == 100

# Stage only the repository's existing local checkpoint into ExpM1's offline
# cache; this function asserts the revision and safetensors payload exist.
checkpoint = vit.prepare_checkpoint()
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
PY

mkdir -p expm1/logs expm1/results/runs expm1/.cache/tmp

run_gpu() (
    local physical_gpu="$1"
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    while IFS=$'\t' read -r task method source beta seed run_name; do
        command=(
            python -B expm1/worker.py
            --task "$task"
            --method "$method"
            --source "$source"
            --seed "$seed"
        )
        if [[ "$beta" != none ]]; then
            command+=(--beta "$beta")
        fi
        "${command[@]}" 2>&1 | tee "expm1/logs/${run_name}.log"
    done < <(
        python -B - "$physical_gpu" <<'PY'
import sys
from expm1 import config as cfg

gpu = int(sys.argv[1])
assert gpu in (1, 2, 3)
runs = cfg.GPU_RUNS[gpu]
assert len(runs) in (13, 12)
for run in runs:
    beta = "none" if run.beta is None else f"{run.beta:g}"
    print(run.task, run.method, run.source, beta, run.seed, run.name, sep="\t")
PY
    )
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
    echo "At least one formal worker failed; analysis was not started." >&2
    exit 1
fi

python -B expm1/analyze.py 2>&1 | tee expm1/logs/analysis.log

#!/usr/bin/env bash
set -euo pipefail
source /HOME/sysu_qling/sysu_qling_3/miniconda3/etc/profile.d/conda.sh
conda activate adamex
export PYTHONDONTWRITEBYTECODE=1
trap 'code=$?; printf "%s\n" "$code" > exp10b/results/exit_code.txt' EXIT
python -m unittest exp10b.test_exp10b exp10b.test_standalone -v > exp10b/results/tests.log 2>&1
python -u exp10b/run_exp10b.py --smoke > exp10b/results/smoke.log 2>&1
python -u exp10b/run_exp10b.py > exp10b/results/full_run.log 2>&1

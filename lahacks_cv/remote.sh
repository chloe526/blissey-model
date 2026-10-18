#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VERI_OUTPUT_DIR:?Veri must provide its artifact output directory}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=42
TASK_WORK=$(mktemp -d /tmp/blissey-lahacks-cv-XXXXXX)
trap 'rm -rf "$TASK_WORK"' EXIT
python -B -m lahacks_cv.data --fetch --raw-root "$TASK_WORK/raw" --output "$TASK_WORK/dataset"
python -B -m lahacks_cv.train --data-dir "$TASK_WORK/dataset" --device cuda

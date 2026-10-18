#!/usr/bin/env bash
set -euo pipefail
: "${VERI_OUTPUT_DIR:?Veri artifact directory required}"
# Public train/validation arrays only; no test download or repository clone.
python fetch_dataset.py --source configs/dataset_source.json --output data/remote --splits train val
mkdir -p "$VERI_OUTPUT_DIR"
cp configs/dataset_source.json "$VERI_OUTPUT_DIR/dataset_source.json"
python train.py --data-dir data/remote "$@"

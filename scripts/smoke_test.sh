#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
"$PYTHON" inspect_dataset.py
"$PYTHON" preprocess.py
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" smoke_test_moment.py --import-only
"$PYTHON" train.py --help >/dev/null
"$PYTHON" evaluate.py --help >/dev/null
"$PYTHON" infer.py --help >/dev/null
bash -n scripts/check_environment.sh scripts/submit_veri.sh scripts/remote_smoke.sh
printf 'Local checks passed. No pretrained weights downloaded. Remote job requires explicit approval.\n'

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import platform,shutil,sys
import torch
from momentfm import MOMENTPipeline
print('Python:',sys.version.split()[0],platform.platform())
for name in ('numpy','torch','transformers','huggingface-hub','momentfm'):
    print(name,metadata.version(name))
print('CUDA available:',torch.cuda.is_available(),'CUDA runtime:',torch.version.cuda)
print('Free disk GB:',round(shutil.disk_usage('.').free/1e9,2))
if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
PY
if command -v veri >/dev/null 2>&1; then
  veri --version
  veri run-script --help
fi

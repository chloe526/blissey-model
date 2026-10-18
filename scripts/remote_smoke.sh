#!/usr/bin/env bash
set -euo pipefail
# Only called INSIDE a paid GPU job after explicit user approval.
: "${VERI_OUTPUT_DIR:?Veri must provide its artifact directory}"
python - <<'PY'
import importlib.metadata as m
import os
import torch
from pathlib import Path
from training_data import load_data
assert torch.cuda.is_available(), 'Expected CUDA GPU in veri/base'
p=Path(os.environ['VERI_OUTPUT_DIR']);p.mkdir(parents=True,exist_ok=True)
probe=p/'.write_probe';probe.write_text('ok');probe.unlink()
print('GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__,'CUDA:',torch.version.cuda,flush=True)
for package in ('momentfm','numpy','transformers','huggingface-hub','veri-sdk'):
    print(package,m.version(package),flush=True)
data,config=load_data('data/smoke',splits=('train','val'))
assert config.get('diagnostic_subset') is True
assert set(data)=={'train','val'}
assert all(len(d)==2 for d in data.values())
assert len(config['channel_indices'])==52 and config['sequence_length']==512
for split,dataset in data.items():
    print('Prepared data:',split,list(dataset.X.shape),flush=True)
print('Prepared data directory:',str(Path('data/smoke').resolve()),flush=True)
print('Artifacts:',str(p),flush=True)
PY
# Downloads pinned actual MOMENT-1-large on Veri, not the Mac.
# One training batch of ONE window -> exactly ONE optimizer update.
# One validation batch, then save/reload and compare forward logits.
python train.py --data-dir data/smoke --smoke-data --epochs 1 --max-batches 1 --batch-size 1
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['VERI_OUTPUT_DIR'])
history=json.loads((p/'history.json').read_text())
check=json.loads((p/'reload_check.json').read_text())
assert len(history)==1 and history[0]['train']['optimizer_steps']==1
assert history[0]['train']['windows']==1 and history[0]['validation']['windows']==1
assert check['passed'] and (p/'best.pt').stat().st_size>1_000_000_000
print('PASS: actual pretrained MOMENT; one optimizer step; checkpoint captured and reloaded.',flush=True)
PY

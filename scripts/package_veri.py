"""Mode-specific packaging: tiny real smoke fixtures, or code-only full runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_data import load_data

COMMON=('model_utils.py','metrics.py','training_data.py','train.py','labeling_v3.py',
        'requirements-veri.txt','requirements-veri.lock.txt','configs/training.json')


def copy_asset(source,target):
    """Retained for making independent baseline snapshots without duplicate disk use."""
    command=['cp','-c',str(source),str(target)] if sys.platform=='darwin' else ['cp','--reflink=always',str(source),str(target)]
    result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if result.returncode:
        if target.exists():target.unlink()
        if shutil.disk_usage(target.parent).free<source.stat().st_size+100_000_000:
            raise RuntimeError('Not enough disk for snapshot')
        shutil.copyfile(source,target)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_fixture(data_root,target):
    data,config=load_data(data_root,splits=('train','val'))
    if config.get('preprocessing_version') not in (2,3):raise ValueError('Smoke fixtures require corrected v2/v3 data')
    config=copy.deepcopy(config)
    config['diagnostic_subset']=True
    config['subset_provenance']=dict(source_config_sha256=sha(data_root/'preprocessing_config.json'),
        full_artifact_sha256=dict(config['artifact_sha256']),source_indices={})
    selected_ids=set();target.mkdir(parents=True)
    for split in ('train','val'):
        # One real fall and one real normal, in that order; no test windows.
        indices=np.asarray([int(np.flatnonzero(data[split].y==label)[0]) for label in (1,0)])
        with np.load(data_root/f'{split}.npz',allow_pickle=False) as source:
            arrays={name:source[name][indices].copy() for name in source.files}
        np.savez_compressed(target/f'{split}.npz',**arrays)
        config['subset_provenance']['source_indices'][split]=indices.tolist()
        selected_ids.update(map(str,arrays['recording_id']))
        config['subset_provenance'].setdefault('source_quality',{})[split]=config['quality'][split]
        config['quality'][split]=dict(windows=2,class_counts=[1,1],
            independent_recordings=len(set(arrays['recording_id'])),diagnostic_only=True)
    config['recordings']={rid:r for rid,r in config['recordings'].items() if rid in selected_ids}
    config['quality']={split:config['quality'][split] for split in ('train','val')}
    config['artifact_sha256']={s:sha(target/f'{s}.npz') for s in ('train','val')}
    # Mean/std, channels, quality-v2 policy and split assignments remain exactly
    # those fitted on the full training set. Never fit on the four samples.
    (target/'preprocessing_config.json').write_text(json.dumps(config,indent=2)+'\n')
    load_data(target,splits=('train','val'))
    return dict(train=2,val=2)


def build_package(source,data_root,mode='smoke',dataset_source=None):
    source=Path(source).resolve();data_root=Path(data_root).resolve()
    if mode not in ('smoke','epoch','full'):raise ValueError('Unsupported package mode')
    extras=('scripts/remote_smoke.sh',) if mode=='smoke' else ('fetch_dataset.py','evaluate.py','scripts/remote_train.sh')
    if mode!='smoke':
        if dataset_source is None:raise ValueError('Full/epoch mode requires --dataset-source with published, commit-pinned processed data')
        from fetch_dataset import validate_source
        reference=json.loads(Path(dataset_source).read_text());validate_source(reference)
        # Bind remote data to these exact local quality-v2 artifacts.
        for name,entry in reference['files'].items():
            if sha(data_root/name)!=entry['sha256'] or (data_root/name).stat().st_size!=entry['bytes']:
                raise ValueError(f'Remote reference does not match current corrected data: {name}')
    stage=Path(tempfile.mkdtemp(prefix='blissey-veri-'))
    try:
        for name in COMMON+extras:
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
        info=dict(mode=mode,preprocessing_version=json.loads((data_root/'preprocessing_config.json').read_text())['preprocessing_version'],test_arrays_uploaded=False,full_arrays_uploaded=False)
        if mode=='smoke':info['fixture_windows']=smoke_fixture(data_root,stage/'data/smoke')
        else:
            (stage/'configs/dataset_source.json').write_text(json.dumps(reference,indent=2)+'\n')
        (stage/'package_info.json').write_text(json.dumps(info,indent=2)+'\n')
        manifest={str(f.relative_to(stage)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(stage.rglob('*')) if f.is_file()}
        (stage/'upload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    except BaseException:
        shutil.rmtree(stage);raise
    return stage


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--mode',choices=('smoke','epoch','full'),default='smoke');p.add_argument('--dataset-source');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    print(build_package(root,root/a.data_dir,a.mode,a.dataset_source))

if __name__=='__main__':main()

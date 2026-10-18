"""Prepare an allowlisted, content-versioned train/val bundle; never publish it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fetch_dataset import DATA_FILES
from scripts.package_veri import copy_asset,sha
from training_data import load_data


def prepare(data_root,output_root):
    data_root=Path(data_root);output_root=Path(output_root)
    data,config=load_data(data_root,splits=('train','val'))
    if config.get('preprocessing_version')!=2 or config.get('diagnostic_subset'):
        raise ValueError('Only complete, corrected quality-v2 train/validation artifacts may be hosted')
    files={name:dict(bytes=(data_root/name).stat().st_size,sha256=sha(data_root/name)) for name in DATA_FILES}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    content_revision='quality-v2-trainval-'+digest[:12]
    target=output_root/content_revision;target.mkdir(parents=True,exist_ok=True)
    allowed=set(DATA_FILES)|{'artifact_manifest.json','SHA256SUMS'}
    if any(p.name not in allowed or not p.is_file() for p in target.iterdir()):
        raise ValueError('Unexpected content in hosting target; use a clean output directory')
    for name,entry in files.items():
        dest=target/name
        if not dest.exists():copy_asset(data_root/name,dest)
        if dest.stat().st_size!=entry['bytes'] or sha(dest)!=entry['sha256']:
            raise ValueError(f'Hosting copy differs from original: {name}')
    manifest=dict(schema_version=1,preprocessing_version=2,content_revision=content_revision,
        content_sha256=digest,files=files,total_artifact_bytes=sum(f['bytes'] for f in files.values()),
        splits={s:dict(shape=list(d.X.shape),class_counts=[int((d.y==i).sum()) for i in (0,1)],
                       collection_groups=config['group_splits'][s]) for s,d in data.items()},
        remote_commit_revision=None,
        note='Content revision identifies these bytes, not a Git commit. After publication, pin the actual 40-character remote commit with make_dataset_source.py. Original config is unchanged and retains all split provenance; no test tensors are bundled.')
    (target/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'SHA256SUMS').write_text(''.join(f"{entry['sha256']}  {name}\n" for name,entry in files.items()))
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/processed');p.add_argument('--output-root',default='hosting')
    a=p.parse_args();print(prepare(a.data_dir,a.output_root))


if __name__=='__main__':main()

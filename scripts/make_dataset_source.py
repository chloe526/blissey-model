"""Prepare a download manifest only; does not publish or commit any data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fetch_dataset import DATA_FILES,ALL_DATA_FILES,validate_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('github','huggingface'),required=True)
    p.add_argument('--repo-id',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--path-prefix',default='processed/quality-v2');p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--include-test',action='store_true',help='Explicit opt-in only; training needs no hosted test arrays')
    p.add_argument('--output',default='configs/dataset_source.json');a=p.parse_args()
    root=Path(a.data_dir)
    config=json.loads((root/'preprocessing_config.json').read_text())
    if config.get('preprocessing_version') not in (2,3) or config.get('diagnostic_subset') or config.get('preprocessing_version')==3 and not config.get('dataset_ready'):raise ValueError('Use full ready v2/v3 artifacts')
    reference=dict(schema_version=1,preprocessing_version=config['preprocessing_version'],backend=a.backend,repo_id=a.repo_id,revision=a.revision,path_prefix=a.path_prefix,
        files={name:dict(bytes=(root/name).stat().st_size,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest()) for name in (ALL_DATA_FILES if a.include_test else DATA_FILES)})
    validate_source(reference);target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(reference,indent=2)+'\n');print('Local manifest only; nothing published:',target)

if __name__=='__main__':main()

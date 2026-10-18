"""Fetch selected commit-pinned numerical artifacts; never clone raw recordings."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tempfile
from urllib.error import HTTPError,URLError
from urllib.parse import quote
from urllib.request import Request,urlopen

DATA_FILES=('preprocessing_config.json','train.npz','val.npz')
ALL_DATA_FILES=DATA_FILES+('test.npz',)


def validate_source(source):
    if source.get('schema_version')!=1 or source.get('preprocessing_version') not in (2,3):raise ValueError('Need a v2/v3 dataset manifest')
    if source.get('backend') not in ('github','huggingface'):raise ValueError('Backend must be github or huggingface')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',source.get('repo_id','')) or any(p in ('.','..') for p in source['repo_id'].split('/')):raise ValueError('Invalid repository ID')
    if not re.fullmatch(r'[0-9a-f]{40}',source.get('revision','')):raise ValueError('Dataset revision must be a full immutable 40-character commit SHA')
    prefix=source.get('path_prefix','')
    if not isinstance(prefix,str) or prefix.startswith('/') or '\\' in prefix or '..' in PurePosixPath(prefix).parts:raise ValueError('Unsafe dataset path prefix')
    if set(source.get('files',{})) not in (set(DATA_FILES),set(ALL_DATA_FILES)):
        raise ValueError('Manifest must identify config plus train/val, with optional test NPZ')
    for name,entry in source['files'].items():
        if not re.fullmatch(r'[0-9a-f]{64}',entry.get('sha256','')) or type(entry.get('bytes')) is not int or entry['bytes']<=0:raise ValueError(f'Invalid size/hash: {name}')
    return source


def file_url(source,name,media=False):
    path='/'.join(x for x in (source['path_prefix'].strip('/'),name) if x)
    if source['backend']=='github':
        if media:return f"https://media.githubusercontent.com/media/{source['repo_id']}/{source['revision']}/{quote(path,safe='/')}"
        return f"https://raw.githubusercontent.com/{source['repo_id']}/{source['revision']}/{quote(path,safe='/')}"
    from huggingface_hub import hf_hub_url
    return hf_hub_url(source['repo_id'],filename=path,repo_type='dataset',revision=source['revision'])


def open_public(url):
    return urlopen(Request(url,headers={'User-Agent':'blissey-beacon-quality-v2'}),timeout=60)


def copy_verified(source,name,output=None,opener=open_public):
    """Stream verification keeps large data off the Mac during preflight."""
    expected=source['files'][name];digest=hashlib.sha256();count=0
    response=opener(file_url(source,name))
    try:
        first=response.read(1024)
        if source['backend']=='github' and first.startswith(b'version https://git-lfs.github.com/spec/v1'):
            response.close();response=opener(file_url(source,name,media=True));first=response.read(1024)
        chunk=first
        while chunk:
            count+=len(chunk)
            if count>expected['bytes']:raise ValueError(f'{name}: remote file exceeds pinned size')
            digest.update(chunk)
            if output is not None:output.write(chunk)
            chunk=response.read(1024*1024)
    finally:response.close()
    if count!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:raise ValueError(f'{name}: remote size/SHA256 differs from corrected local artifact')
    return count


def selected_files(source,splits):
    validate_source(source)
    if not splits or len(set(splits))!=len(splits) or not set(splits)<={'train','val','test'}:
        raise ValueError('Invalid split selection')
    names=('preprocessing_config.json',)+tuple(f'{split}.npz' for split in splits)
    if not set(names)<=set(source['files']):raise ValueError('Requested split is absent from the pinned manifest')
    return names


def materialize(source,target,verify_only=False,opener=open_public,splits=('train','val')):
    validate_source(source)
    names=selected_files(source,splits)
    if verify_only:
        for name in names:copy_verified(source,name,opener=opener)
        return
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for name in names:
        path=target/name;entry=source['files'][name]
        if path.exists() and path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']:continue
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=target,prefix='download-',delete=False) as stream:
                temporary=Path(stream.name);copy_verified(source,name,stream,opener)
            temporary.replace(path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
    from training_data import load_data
    _,config=load_data(target,splits=splits)
    if config.get('preprocessing_version')!=source['preprocessing_version'] or config.get('diagnostic_subset'):raise ValueError('Remote dataset version differs from pinned full-data manifest')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    p.add_argument('--output',default='data/remote');p.add_argument('--verify-only',action='store_true')
    p.add_argument('--splits',nargs='+',choices=('train','val','test'),default=['train','val']);a=p.parse_args()
    try:
        source=validate_source(json.loads(Path(a.source).read_text()))
        materialize(source,a.output,a.verify_only,splits=tuple(a.splits))
    except HTTPError as exc:
        raise SystemExit(f'Pinned processed data unavailable (HTTP {exc.code}). Publish the selected processed artifacts, then update the immutable revision; no GPU job should start.') from None
    except (ValueError,URLError,OSError) as exc:raise SystemExit(f'Dataset preflight failed: {exc}') from None
    print('Pinned numerical dataset SHA256 verification passed.' if a.verify_only else f'Full numerical dataset verified: {a.output}')

if __name__=='__main__':main()

"""Measure the actual Veri 0.2.44 tar/gzip stream without writing a large archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

SDK_EXCLUDES={'.git','__pycache__','.venv','venv','env','.mypy_cache','.pytest_cache','.ruff_cache','node_modules','.DS_Store'}


class CountingSink(io.RawIOBase):
    def __init__(self):super().__init__();self.count=0
    def writable(self):return True
    def tell(self):return self.count
    def write(self,data):self.count+=len(data);return len(data)


def sdk_filter(info):
    return None if set(info.name.split('/')) & SDK_EXCLUDES else info


def write_archive(stage,buffer):
    # Exact SDK order, format, gzip defaults and filter; never create an
    # intermediate tar.gz on the nearly full local disk.
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for child in sorted(Path(stage).iterdir()):archive.add(child,arcname=child.name,filter=sdk_filter)


def measure(stage):
    stage=Path(stage)
    files=[]
    for p in sorted(stage.rglob('*')):
        if p.is_symlink():raise ValueError(f'Symlinks are not allowed: {p.relative_to(stage)}')
        if p.is_file():files.append(dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size))
    sink=CountingSink();write_archive(stage,sink)
    return dict(file_count=len(files),file_bytes=sum(f['bytes'] for f in files),
        tar_gz_bytes=sink.count,largest_files=sorted(files,key=lambda f:f['bytes'],reverse=True)[:8])


def preflight(stage,max_mib=8.):
    stage=Path(stage)
    info=json.loads((stage/'package_info.json').read_text())
    manifest=json.loads((stage/'upload_manifest.json').read_text())
    actual={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
    if actual!=set(manifest)|{'upload_manifest.json'}:raise ValueError('Unexpected file in upload directory')
    for name,entry in manifest.items():
        p=stage/name
        if p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Upload manifest mismatch: {name}')
    forbidden={'.git','.cache','__pycache__','node_modules','processed_legacy_v1','raw','recordings'}
    for name in actual:
        parts=Path(name).parts
        if any(p in forbidden or p.startswith('.venv') or p in ('.env','credentials') for p in parts):
            raise ValueError(f'Forbidden upload path: {name}')
        if Path(name).suffix in ('.pt','.pth','.safetensors','.bin'):raise ValueError(f'Weights/raw binary forbidden: {name}')
    npz={name for name in actual if name.endswith('.npz')}
    if info['mode']=='smoke':
        if npz!={'data/smoke/train.npz','data/smoke/val.npz'}:raise ValueError('Smoke upload must have only tiny train/val fixtures')
        if any(info['fixture_windows'][s]!=2 for s in ('train','val')):raise ValueError('Smoke fixture must contain exactly two real windows per split')
    elif npz:raise ValueError('Full/epoch code upload must contain no dataset arrays')
    stats=measure(stage)
    budget=int(max_mib*1024**2)
    if budget<=65536:raise ValueError('Upload budget must exceed 64 KiB')
    stats.update(mode=info['mode'],budget_bytes=budget,multipart_margin_bytes=65536,
        budget_is_local_policy=True,server_limit_verified=False,
        passed=stats['tar_gz_bytes']+65536<=budget)
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--max-mib',type=float,default=8.)
    p.add_argument('--measure-only',action='store_true');p.add_argument('--report');a=p.parse_args()
    r=measure(a.stage) if a.measure_only else preflight(a.stage,a.max_mib)
    if a.report:
        target=Path(a.report)
        if target.resolve().is_relative_to(Path(a.stage).resolve()):raise ValueError('Preflight report must stay outside the upload')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(r,indent=2)+'\n')
    print(f"Upload files: {r['file_bytes']:,} bytes ({r['file_bytes']/1024**2:.3f} MiB)")
    print(f"Compressed tar.gz: {r['tar_gz_bytes']:,} bytes ({r['tar_gz_bytes']/1024**2:.3f} MiB)")
    if not a.measure_only:
        print(f"Local upload budget: {r['budget_bytes']:,} bytes, including 64 KiB multipart margin (server limit unknown)")
        if not r['passed']:raise SystemExit('Upload exceeds local size budget; submission stopped before network upload')
    if a.measure_only:print(json.dumps(r['largest_files'],indent=2))

if __name__=='__main__':main()

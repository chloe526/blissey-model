"""Curated code-only Veri submission. Defaults to dry run; no all-data refit."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from .common import ROOT, sha, write_json
from .data import read_source, transfer, verify_code
from scripts.archive_preflight import preflight

CODE = ('csi.py', 'csi_quality.py', 'preprocess.py', 'inspect_dataset.py',
        'configs/packet_quality_v2.json', 'cnn_baseline/__init__.py', 'cnn_baseline/model.py')


def package(parent=None):
    source = read_source(); verify_code(source)
    stage = Path(tempfile.mkdtemp(prefix='blissey-lahacks-code-', dir=parent))
    files = list(CODE) + ['lahacks_cv/' + name for name in (
        '__init__.py', 'common.py', 'data.py', 'features.py', 'metrics.py', 'train.py',
        'infer.py', 'transport.py', 'source.json', 'config.json', 'requirements.txt', 'remote.sh')]
    for name in files:
        target = stage / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(stage / 'package_info.json', dict(mode='lahacks_cv', refit=False,
        source_revision=source['revision'], raw_files_in_upload=0, dataset_arrays_in_upload=0))
    manifest = {str(p.relative_to(stage)): dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(stage.rglob('*')) if p.is_file()}
    write_json(stage / 'upload_manifest.json', manifest)
    return stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    p.add_argument('--verify-remote', action='store_true', help='Verify pinned source hashes, reusing verified local copies; download missing files only')
    p.add_argument('--source-root', type=Path, help='Existing read-only LA source root to reuse')
    p.add_argument('--offline', action='store_true', help='Fail on missing local source files instead of using the network')
    p.add_argument('--verification-timeout', type=float, default=120., help='Hard overall source-verification deadline in seconds')
    p.add_argument('--max-upload-mib', type=float, default=8.)
    a = p.parse_args(); source = read_source(); verify_code(source)
    print('[preflight] Checking Veri CLI (15s timeout)', flush=True)
    help_text = subprocess.run(['veri', 'run-script', '--help'], check=True, capture_output=True, text=True, timeout=15).stdout
    for flag in ('--entrypoint', '--gpu-type', '--gpu-count', '--base-image', '--requirements'):
        if flag not in help_text:
            raise RuntimeError('Installed Veri CLI lacks ' + flag)
    verification = None
    if a.verify_remote or a.execute or a.offline or a.source_root:
        roots = [a.source_root] if a.source_root else []
        roots += [Path('/tmp') / ('blissey-stage1-' + source['revision'][:12]), ROOT.parent / 'blissey-data/recordings']
        roots = [p for p in roots if p.is_dir()]
        print('[preflight] Verifying source SHA-256; local roots: ' + ', '.join(map(str, roots)), flush=True)
        verification = transfer(source, cache=ROOT / 'data/lahacks_source_cache' / source['revision'],
                                reuse_roots=roots, offline=a.offline, timeout=a.verification_timeout)
    print('[preflight] Packaging code and measuring archive; no upload yet', flush=True)
    stage = package(); stats = preflight(stage, a.max_upload_mib)
    report = dict(stage=str(stage), **stats, source_revision=source['revision'],
                  source_download_bytes=sum(f['bytes'] for f in source['files']),
                  source_verification=verification,
                  source_integrity_verified=verification is not None,
                  remote_sha256_verified=bool(verification and verification['downloaded_files'] == len(source['files'])),
                  submitted=False)
    write_json(ROOT / 'reports/lahacks_cv_upload_preflight.json', report)
    print(f"Code upload: {stats['file_bytes']:,} bytes; tar.gz: {stats['tar_gz_bytes']:,} bytes", flush=True)
    if not stats['passed']:
        raise SystemExit('Archive exceeds local size policy; no upload or job submitted')
    command = ['veri', 'run-script', str(stage), '--entrypoint', 'bash lahacks_cv/remote.sh',
               '--gpu-type', 'L4-24GB', '--gpu-count', '1', '--base-image', 'veri/base',
               '--requirements', str(stage / 'lahacks_cv/requirements.txt')]
    print(shlex.join(command), flush=True)
    if a.execute:
        print('[submit] Source checks passed; starting Veri upload/submission now. Do not run a second submitter.', flush=True)
        subprocess.run(command, check=True)
    else:
        print('DRY RUN: no code upload, GPU submission, training or deployment refit.', flush=True)


if __name__ == '__main__':
    main()

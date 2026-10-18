"""Pinned LA-only source transfer and unchanged quality-v2 preprocessing."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from csi import load_recording
from csi_quality import window_quality
from preprocess import label_window, recording_features, transform_window
from .common import GROUPS, ROOT, canonical_hash, fit_normalization, fold_indices, sha, write_json

SOURCE = ROOT / 'lahacks_cv/source.json'
CONFIG = ROOT / 'lahacks_cv/config.json'


def read_source(path=SOURCE):
    source = json.loads(Path(path).read_text())
    if source['repository'] != 'chloe526/blissey-data' or not re.fullmatch('[0-9a-f]{40}', source['revision']):
        raise ValueError('Expected explicitly pinned public Blissey data repository')
    if source['groups'] != list(GROUPS):
        raise ValueError('LA-only groups required')
    paths = set()
    for f in source['files']:
        p = PurePosixPath(f['path'])
        if (not p.parts or p.parts[0] not in GROUPS or '..' in p.parts or p.is_absolute()
                or p.name not in ('metadata.json', 'events.jsonl', 'measurements.jsonl')):
            raise ValueError('Forbidden source path: ' + f['path'])
        if str(p) in paths or not re.fullmatch('[0-9a-f]{64}', f['sha256']) or f['bytes'] <= 0:
            raise ValueError('Invalid source manifest entry')
        paths.add(str(p))
    if len(paths) != 254:
        raise ValueError('Pinned source inventory changed')
    return source


def transfer(source, cache=None, **options):
    """Bounded transport only; dataset preprocessing and label semantics unchanged."""
    from .transport import transfer as bounded_transfer
    return bounded_transfer(source, cache=cache, **options)


def verify_cache(root, source):
    root = Path(root)
    for f in source['files']:
        p = root / f['path']
        if not p.is_file() or p.stat().st_size != f['bytes'] or sha(p) != f['sha256']:
            raise ValueError('Source SHA-256 mismatch: ' + f['path'])
    # Reject undeclared recording/annotation files instead of accidentally loading them.
    expected = {f['path'] for f in source['files']}
    for group in GROUPS:
        actual = {str(p.relative_to(root)) for p in (root / group).rglob('*')
                  if p.name in ('metadata.json', 'events.jsonl', 'measurements.jsonl')}
        if actual != {p for p in expected if p.startswith(group + '/')}:
            raise ValueError('Source cache inventory mismatch')


def geometry():
    # Constants and immutable acquisition rules only; never read old NPZ/scalers.
    quality = json.loads((ROOT / 'configs/packet_quality_v2.json').read_text())
    return dict(window_seconds=3., stride_seconds=.5, sequence_length=512,
                channel_indices=list(range(6, 32)) + list(range(33, 59)),
                max_gap_seconds=.25, packet_quality=quality, preprocessing_version=2,
                label_policy='unchanged label_window: normal=0; full fall containment with 0.25s inner margin=1; otherwise ignore')


def verify_code(source):
    for path, digest in source['implementation_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen quality-v2 implementation changed: ' + path)
    actual = geometry()
    if canonical_hash(actual) != source['geometry_sha256']:
        raise ValueError('Geometry or quality-v2 gates changed')


def prepare(raw_root, output, source_path=SOURCE):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Versioned dataset output already exists; refusing overwrite')
    source = read_source(source_path); verify_code(source); verify_cache(raw_root, source)
    cfg = geometry(); arrays = {k: [] for k in ('X', 'y', 'recording_id', 'group', 'event_id', 'start', 'end')}
    records = {}; counts = {}
    paths = [f['path'] for f in source['files'] if f['path'].endswith('/metadata.json')]
    for path in sorted(paths):
        group = PurePosixPath(path).parts[0]
        m, rows, status, intervals, issues, errors = load_recording(Path(raw_root) / Path(path).parent)
        if errors or status == 'ambiguous' or len({p['source_mac'] for p in rows}) != 1:
            raise ValueError('Unexpected invalid/ambiguous source recording: ' + path)
        rid = m['recording_id']
        if rid in records:
            raise ValueError('Recording identity repeated across blocks')
        times, features, duplicates = recording_features(rows, cfg['channel_indices'])
        records[rid] = dict(group=group, participant=m['participant'], activity=m['activity'],
                            status=status, intervals=intervals, duplicate_timestamps=duplicates,
                            source_path=str(Path(path).parent), issues=issues)
        c = counts.setdefault(group, dict(candidates=0, ignored=0, rejected=0, normal=0, fall=0))
        for start in np.arange(times[0], times[-1] - 3 + 1e-9, .5):
            c['candidates'] += 1
            label = label_window(start, start + 3, status, intervals)
            if label is None:
                c['ignored'] += 1; continue
            if window_quality(times, float(start), cfg)['reasons']:
                c['rejected'] += 1; continue
            x = transform_window(times, features, start, cfg, normalize=False)
            event_id = ''
            if label == 1:
                matches = [i for i, (a, b) in enumerate(intervals) if start >= a + .25 and start + 3 <= b - .25]
                if len(matches) != 1:
                    raise ValueError('Positive window must map to exactly one physical trial')
                event_id = f'{rid}:fall:{matches[0]}'
            for key, value in dict(X=x, y=label, recording_id=rid, group=group,
                                   event_id=event_id, start=float(start), end=float(start + 3)).items():
                arrays[key].append(value)
            c['fall' if label else 'normal'] += 1
    data = {k: np.stack(v) if k == 'X' else np.asarray(v) for k, v in arrays.items()}
    data['y'] = data['y'].astype(np.int64)
    expected = {GROUPS[0]: [1249, 21], GROUPS[1]: [301, 31]}
    covered = len(set(data['event_id']) - {''})
    if len(records) != 92 or sum(len(r['intervals']) for r in records.values()) != 34 or covered != 13:
        raise ValueError('Audited recording/event totals did not reproduce')
    for g in GROUPS:
        if np.bincount(data['y'][data['group'] == g], minlength=2).tolist() != expected[g]:
            raise ValueError('Audited class totals did not reproduce')
    scalers = {}
    for fold in ('A', 'B'):
        train, val = fold_indices(data, fold)
        scalers[fold] = dict(**fit_normalization(data['X'], train),
                             fit_recordings=sorted(set(data['recording_id'][train].tolist())),
                             train_group=str(data['group'][train[0]]),
                             validation_group=str(data['group'][val[0]]))
    output.mkdir(parents=True)
    np.savez_compressed(output / 'windows.npz', **data)
    metadata = dict(version='lahacks_quality_v2_cv_v1', normalized=False, geometry=cfg,
                    source=source, source_manifest_sha256=sha(source_path), records=records,
                    counts=counts, annotated_events=34, quality_covered_events=covered,
                    artifact_sha256=sha(output / 'windows.npz'), fold_normalization=scalers)
    write_json(output / 'dataset.json', metadata)
    print(json.dumps(dict(dataset=str(output), counts=counts, covered_events=covered,
                          annotated_events=34, bytes=(output / 'windows.npz').stat().st_size)), flush=True)
    return metadata


def load_dataset(path):
    path = Path(path); meta = json.loads((path / 'dataset.json').read_text())
    source = read_source(); verify_code(source)
    if meta['source_manifest_sha256'] != sha(SOURCE) or meta['source'] != source:
        raise ValueError('Dataset source provenance mismatch')
    if meta['normalized'] or meta['geometry'] != geometry():
        raise ValueError('Expected unnormalized, unchanged quality-v2 windows')
    if sha(path / 'windows.npz') != meta['artifact_sha256']:
        raise ValueError('Dataset SHA-256 mismatch')
    with np.load(path / 'windows.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if data['X'].shape != (1602, 52, 512) or data['X'].dtype != np.float32 or not np.isfinite(data['X']).all():
        raise ValueError('Invalid LA-only tensor dataset')
    for fold in ('A', 'B'):
        fold_indices(data, fold)
    return data, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--fetch', action='store_true'); p.add_argument('--verify-remote', action='store_true')
    p.add_argument('--check-dataset', type=Path)
    a = p.parse_args(); source = read_source(); verify_code(source)
    if a.verify_remote:
        transfer(source); print('All 254 pinned remote source files passed SHA-256 verification.'); return
    if a.check_dataset:
        data, meta = load_dataset(a.check_dataset)
        print(json.dumps(dict(shape=list(data['X'].shape), counts=meta['counts'], sha256=meta['artifact_sha256']))); return
    if a.raw_root is None or a.output is None:
        p.error('--raw-root and --output required')
    if a.fetch:
        transfer(source, a.raw_root)
    prepare(a.raw_root, a.output)


if __name__ == '__main__':
    main()

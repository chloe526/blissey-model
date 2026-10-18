import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('session_03_LAHACKS', 'session_04_LAHACKS')
FOLDS = {'A': (GROUPS[0], GROUPS[1]), 'B': (GROUPS[1], GROUPS[0])}
STATUS = 'development cross-validation only; no independent final test'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fold_indices(data, fold):
    train_group, val_group = FOLDS[fold]
    groups = set(data['group'].tolist())
    if groups != set(GROUPS):
        raise ValueError('Only the two explicit LA Hacks blocks are permitted')
    train = np.flatnonzero(data['group'] == train_group)
    val = np.flatnonzero(data['group'] == val_group)
    for key in ('recording_id', 'event_id'):
        a = set(data[key][train].tolist()) - {''}
        b = set(data[key][val].tolist()) - {''}
        if a & b:
            raise ValueError('Recording or physical event crosses folds')
    if not len(train) or not len(val):
        raise ValueError('Empty block')
    return train, val


def fit_normalization(x, indices):
    """Accumulate training-only moments without copying the full float64 dataset."""
    if len(indices) == 0:
        raise ValueError('Empty normalization fit')
    total = np.zeros(52, dtype=np.float64)
    squares = total.copy()
    count = 0
    for chunk in np.array_split(indices, max(1, int(np.ceil(len(indices) / 32)))):
        batch = x[chunk].astype(np.float64)
        total += batch.sum(axis=(0, 2))
        squares += np.square(batch).sum(axis=(0, 2))
        count += len(batch) * batch.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-12))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'fit_windows': len(indices)}


def normalize(x, scaler):
    return ((x - np.asarray(scaler['mean'], dtype=np.float32)[..., None]) /
            np.asarray(scaler['std'], dtype=np.float32)[..., None]).astype(np.float32)


def class_weights(y):
    counts = np.bincount(y, minlength=2)
    if len(counts) != 2 or np.any(counts == 0):
        raise ValueError('Both classes required in each training partition')
    # Predeclared, conservative weighting; inverse-frequency ratio can reach 59x.
    # Do not tune this cap on held-out block predictions.
    return np.array([1., min(4., float(np.sqrt(counts[0] / counts[1])))], dtype=np.float32)


def inner_recording_split(data, indices, seed):
    """Early-stop on whole training-block recordings, never the outer block."""
    rng = np.random.default_rng(seed)
    fit, dev = [], []
    for label in (0, 1):
        ids = np.unique(data['recording_id'][indices[data['y'][indices] == label]])
        if len(ids) < 2:
            raise ValueError('Need at least two contributing recordings per class for inner split')
        ids = rng.permutation(ids)
        n = min(len(ids) - 1, max(1, int(np.ceil(.2 * len(ids)))))
        dev.extend(indices[np.isin(data['recording_id'][indices], ids[:n])].tolist())
        fit.extend(indices[np.isin(data['recording_id'][indices], ids[n:])].tolist())
    fit, dev = np.asarray(sorted(set(fit))), np.asarray(sorted(set(dev)))
    if set(data['recording_id'][fit]) & set(data['recording_id'][dev]):
        raise ValueError('Mixed-label recording leaked across inner split')
    return fit, dev

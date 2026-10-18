"""Every quality-passing window of every LA recording, built in memory from the raw cache.

Reuses the unchanged quality-v2 gates, label rule and interpolation (`preprocess.py`).
Nothing is written to disk: the pinned raw cache is the single source of truth.
"""
from pathlib import Path

import numpy as np

from csi import load_recording
from csi_quality import window_quality
from preprocess import label_window, recording_features, transform_window
from lahacks_cv.common import GROUPS
from lahacks_cv.data import geometry, read_source, verify_cache

DEFAULT_RAW = Path('/private/tmp/blissey-stage1-06c97032c93a')
PRE_MARGIN = .5      # pre_action windows end at least this long before action_start
DETECT_AFTER = 2.    # an alert counts for an event if its window ends in [start, end + DETECT_AFTER]


def gain_normalize(x):
    """[52,512] raw amplitude -> common-mode (AGC) and per-channel level removed, centred on 1.

    1. Divide each time sample by its mean across subcarriers (exactly removes a common per-sample gain).
    2. Divide each channel by its own window median (removes each channel's window level; approximate
       under per-channel offsets because step 1's denominator also depends on channel levels).
    Window-local and stateless: identical offline and live.
    """
    x = np.asarray(x, dtype=np.float32)
    common = x / np.maximum(x.mean(axis=-2, keepdims=True), 1e-6)
    return (common / np.maximum(np.median(common, axis=-1, keepdims=True), 1e-6)).astype(np.float32)


def role(w0, w1, status, intervals):
    if status == 'verified_non_fall':
        return 'normal_recording', ''
    for k, (a, b) in enumerate(intervals):
        if w0 < b and w1 > a:
            return 'in_event', k
    if all(w1 <= a - PRE_MARGIN for a, _ in intervals):
        return 'pre_action', ''
    if any(w0 >= b for _, b in intervals):
        return 'post_event', ''
    return 'boundary', ''


def interpolate(times, amp, start, cfg):
    """transform_window(normalize=False) without its quality gate: same brackets, same linear grid."""
    q = window_quality(times, start, cfg)
    tt, xx = times[q['left']:q['right']], amp[q['left']:q['right']]
    grid = np.linspace(start, start + cfg['window_seconds'], cfg['sequence_length'])
    return np.stack([np.interp(grid, tt, xx[:, i]) for i in range(xx.shape[1])]).astype(np.float32)


def relaxed_pass(q, max_gap):
    """Gap-tolerant gate (capture review): longest gap <= max_gap, unsupported edge <= max_gap / 2."""
    return q['max_gap_seconds'] <= max_gap + 1e-9 and q['max_edge_distance_seconds'] <= max_gap / 2 + 1e-9


def build(raw_root=DEFAULT_RAW, verify=True, max_gap=None):
    """Return (data dict of aligned arrays, records dict). X is raw amplitude float32 [N,52,512].

    max_gap=None keeps the frozen quality-v2 gate. A number uses `relaxed_pass` instead; the
    `pass_v2` column still marks windows that also pass v2, so populations can be compared."""
    raw_root = Path(raw_root)
    source = read_source()
    if verify:
        verify_cache(raw_root, source)
    cfg = geometry(); idx = cfg['channel_indices']
    cols = {k: [] for k in ('X', 'label_v2', 'recording_id', 'group', 'activity', 'role',
                            'event_id', 'start', 'end', 'interp_fraction', 'pass_v2', 'max_gap')}
    records = {}
    for path in sorted(f['path'] for f in source['files'] if f['path'].endswith('/metadata.json')):
        folder = raw_root / Path(path).parent; group = Path(path).parts[0]
        m, rows, status, intervals, _, errors = load_recording(folder)
        if errors or status == 'ambiguous':
            raise ValueError('Unexpected invalid source recording: ' + path)
        rid = m['recording_id']
        times, amp, _ = recording_features(rows, idx)
        records[rid] = dict(group=group, activity=m['activity'], status=status, intervals=intervals,
                            started_at=m.get('started_at'), first=float(times[0]), last=float(times[-1]))
        for w0 in np.arange(times[0], times[-1] - 3 + 1e-9, .5):
            w0 = float(w0); w1 = w0 + 3
            q = window_quality(times, w0, cfg); pass_v2 = not q['reasons']
            if not (pass_v2 if max_gap is None else relaxed_pass(q, max_gap)):
                continue
            x = transform_window(times, amp, w0, cfg, normalize=False) if pass_v2 else interpolate(times, amp, w0, cfg)
            lab = label_window(w0, w1, status, intervals)
            r, k = role(w0, w1, status, intervals)
            inside = np.concatenate([[w0], times[(times > w0) & (times < w1)], [w1]]); gaps = np.diff(inside)
            for key, value in dict(X=x, label_v2=-1 if lab is None else lab, recording_id=rid, group=group,
                                   activity=m['activity'], role=r, event_id='' if k == '' else f'{rid}:fall:{k}',
                                   start=w0, end=w1, interp_fraction=float(gaps[gaps > .1].sum() / 3),
                                   pass_v2=pass_v2, max_gap=q['max_gap_seconds']).items():
                cols[key].append(value)
    data = {k: np.stack(v) if k == 'X' else np.asarray(v) for k, v in cols.items()}
    data['label_v2'] = data['label_v2'].astype(np.int64)
    # Must reproduce the audited v1 labeled population exactly.
    for g, expect in ((GROUPS[0], [1249, 21]), (GROUPS[1], [301, 31])):
        got = np.bincount(data['label_v2'][(data['group'] == g) & (data['label_v2'] >= 0) & data['pass_v2']],
                          minlength=2).tolist()
        if got != expect:
            raise ValueError(f'{g}: labeled population {got} != audited {expect}')
    return data, records


def onset_labels(data, records, lo, hi):
    """E3 labels: positive if window centre in [start+lo, start+hi] (offsets fit on the training block);
    pre_action windows of fall trials become negatives; everything else in fall trials ignored."""
    y = np.where(data['role'] == 'normal_recording', 0, -1)
    y[data['role'] == 'pre_action'] = 0
    centre = data['start'] + 1.5
    for i in np.flatnonzero(data['role'] == 'in_event'):
        a, _ = records[data['recording_id'][i]]['intervals'][int(data['event_id'][i].rsplit(':', 1)[1])]
        if a + lo <= centre[i] <= a + hi:
            y[i] = 1
    return y.astype(np.int64)

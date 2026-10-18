"""Read-only check: does removing ESP32 common-mode gain (AGC) expose motion/fall signal?

Unfitted, window-local transforms only. Descriptive AUCs on overlapping windows.
Writes gain_check.json next to this script. No training, no dataset writes.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from csi import load_recording  # noqa: E402
from preprocess import recording_features  # noqa: E402
from diagnose import auc, motion_profile  # noqa: E402

RAW = Path('/private/tmp/blissey-stage1-06c97032c93a')
IDX = list(range(6, 32)) + list(range(33, 59))
K = np.ones(25) / 25  # ~150 ms moving average on the 512-point / 3 s grid


def motion(x, common_mode):
    if common_mode:
        x = x / x.mean(axis=1, keepdims=True)  # divide each time sample by its across-subcarrier mean
    rel = x / np.median(x, axis=2, keepdims=True) - 1  # per-window, per-channel DC removal
    sm = np.apply_along_axis(lambda v: np.convolve(v, K, 'same'), 2, rel)[:, :, 20:-20]
    return dict(smooth_std=sm.std(2).mean(1), smooth_absdiff=np.abs(np.diff(sm, axis=2)).mean((1, 2)))


def main():
    z = np.load(ROOT / 'data/lahacks_quality_v2_cv_v1/windows.npz')
    X, y, g, rid = z['X'], z['y'], z['group'], z['recording_id']
    meta = json.loads((ROOT / 'data/lahacks_quality_v2_cv_v1/dataset.json').read_text())['records']
    act = np.array([meta[r]['activity'] for r in rid])
    S3, S4 = g == 'session_03_LAHACKS', g == 'session_04_LAHACKS'
    out = {'note': 'descriptive; ~8 feature variants were inspected on this same data, so treat as hypothesis-generating'}
    for cm in (False, True):
        for name, v in motion(X, cm).items():
            key = ('common_mode_removed:' if cm else 'raw_gain:') + name
            r = dict(s03_walking_vs_empty=auc(v[S3 & (act == 'walking')], v[S3 & (act == 'empty_room')]))
            for s, m in (('s03', S3), ('s04', S4)):
                r[f'{s}_fall_vs_normal'] = auc(v[m & (y == 1)], v[m & (y == 0)])
                r[f'{s}_fall_vs_lie_down'] = auc(v[m & (y == 1)], v[m & (act == 'lie_down')])
                r[f'{s}_fall_vs_sit_down'] = auc(v[m & (y == 1)], v[m & (act == 'sit_down')])
            r['s03_fall_vs_get_up'] = auc(v[S3 & (y == 1)], v[S3 & (act == 'get_up_from_floor')])
            for a in ('lie_down', 'sit_down'):
                r[f'{a}_s04_vs_s03'] = auc(v[S4 & (act == a)], v[S3 & (act == a)])
            # threshold = 95th percentile of TRAINING-block normals only; applied once to the other block
            for fold, tr, va in (('A_03to04', S3, S4), ('B_04to03', S4, S3)):
                t = float(np.quantile(v[tr & (y == 0)], .95))
                r[f'{fold}_val_normal_fp'] = float((v[va & (y == 0)] >= t).mean())
                r[f'{fold}_val_fall_recall'] = float((v[va & (y == 1)] >= t).mean())
            out[key] = r
    # marker-aligned packet-level motion (z within recording), raw vs common-mode removed
    rel_s = np.arange(-4, 10.01, .5); prof = {}
    for cm in (False, True):
        for grp in ('session_03_LAHACKS', 'session_04_LAHACKS'):
            rows_ = []
            for f in sorted((RAW / grp / 'staged_fall').glob('*')):
                _, rows, _, iv, _, _ = load_recording(f)
                t, a, _ = recording_features(rows, IDX)
                if cm:
                    a = a / a.mean(axis=1, keepdims=True)
                gr, mo = motion_profile(t, a)
                zz = (mo - np.nanmedian(mo)) / np.nanstd(mo); s = iv[0][0]
                rows_.append([float(np.nanmean(zz[(gr >= s + r - .25) & (gr < s + r + .25)]))
                              if ((gr >= s + r - .25) & (gr < s + r + .25)).any() else np.nan for r in rel_s])
            arr = np.array(rows_)
            prof[('cm:' if cm else 'raw:') + grp] = dict(seconds_from_action_start=rel_s.tolist(),
                mean_z=np.round(np.nanmean(arr, 0), 2).tolist(), n_events=len(arr))
    out['marker_aligned_motion'] = prof
    (HERE / 'gain_check.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        if isinstance(v, dict) and 's03_walking_vs_empty' in v:
            print(k, {a: round(b, 3) for a, b in v.items() if b is not None})
    for k, v in prof.items():
        print(k, v['mean_z'])


if __name__ == '__main__':
    import warnings; warnings.simplefilter('ignore')
    main()

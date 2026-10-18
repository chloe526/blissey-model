"""Read-only diagnostics for the ML strategy review. No training, no dataset writes.

Inputs: pinned LA raw cache, existing unnormalized windows.npz, saved Fold B artifacts.
Output: analysis.json next to this script (small).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csi import load_recording  # noqa: E402
from csi_quality import window_quality  # noqa: E402
from preprocess import label_window, recording_features  # noqa: E402
from lahacks_cv.data import geometry  # noqa: E402
from lahacks_cv.metrics import ALERT_POLICY  # noqa: E402

GROUPS = ('session_03_LAHACKS', 'session_04_LAHACKS')
DYNAMIC = {'walking', 'stand_up', 'sit_down', 'lie_down', 'get_up_from_floor'}


def auc(pos, neg):
    """Rank AUC P(pos > neg); descriptive only (windows overlap)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1.
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def motion_profile(times, amp, step=.1, half=.25):
    """Packet-only (no interpolation) relative-amplitude motion on a 0.1s grid."""
    rel = amp / np.maximum(np.median(amp, axis=0), 1e-6)
    grid = np.arange(times[0] + half, times[-1] - half, step)
    lo = np.searchsorted(times, grid - half); hi = np.searchsorted(times, grid + half)
    motion = np.full(len(grid), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 4:
            motion[i] = rel[a:b].std(axis=0).mean()
    return grid, motion


def coverage(times, a, b, bin_s=.1):
    if b - a < bin_s:
        return None
    inside = times[(times >= a) & (times < b)]
    n = int(np.floor((b - a) / bin_s))
    return len(np.unique(np.clip(((inside - a) / bin_s).astype(int), 0, n - 1))) / n if n else None


def max_gap(times, a, b):
    t = times[(times >= a) & (times <= b)]
    edges = np.concatenate([[a], t, [b]])
    return float(np.diff(edges).max())


def runs(ends):
    """Longest run of windows with no reset gap under the fixed alert policy."""
    if not len(ends):
        return 0
    best = cur = 1
    for d in np.diff(np.sort(ends)):
        cur = cur + 1 if d <= ALERT_POLICY['reset_after_gap_seconds'] + 1e-9 else 1
        best = max(best, cur)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, default=Path('/private/tmp/blissey-stage1-06c97032c93a'))
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/lahacks_quality_v2_cv_v1')
    p.add_argument('--artifacts', type=Path, default=ROOT / 'reports/lahacks_job_d4c10e9207e8/artifacts')
    a = p.parse_args()
    cfg = geometry(); idx = cfg['channel_indices']
    relaxed = json.loads(json.dumps(cfg)); relaxed['packet_quality']['max_gap_seconds'] = .5
    out = dict(recordings=[], events=[], notes='descriptive; overlapping windows are not independent')
    rec_cache = {}
    for g in GROUPS:
        for folder in sorted((a.raw_root / g).glob('*/*')):
            if not (folder / 'metadata.json').exists():
                continue
            m, rows, status, intervals, issues, errors = load_recording(folder)
            times, amp, _ = recording_features(rows, idx)
            rid = m['recording_id']; rec_cache[rid] = (times, amp)
            grid, motion = motion_profile(times, amp)
            gaps = np.diff(times)
            out['recordings'].append(dict(recording_id=rid, group=g, activity=m['activity'],
                started_at=m.get('started_at'), median_amplitude=float(np.median(amp)),
                rate_hz=len(times) / (times[-1] - times[0]), gaps_over_250ms=int((gaps > .25).sum()),
                gaps_over_100ms=int((gaps > .1).sum()), median_motion=float(np.nanmedian(motion)),
                p95_motion=float(np.nanpercentile(motion, 95))))
            for k, (s, e) in enumerate(intervals):
                ev = dict(event_id=f'{rid}:fall:{k}', group=g, interval=[s, e], duration=e - s)
                ok = np.isfinite(motion)
                inwin = ok & (grid >= s) & (grid <= e)
                ev['motion_coverage_in_interval'] = float(inwin.sum() / max(1, ((grid >= s) & (grid <= e)).sum()))
                if ok.any():
                    pk = grid[ok][np.nanargmax(motion[ok])]
                    ev['global_peak_rel_to_start'] = float(pk - s)
                    ev['global_peak_inside_interval'] = bool(s <= pk <= e)
                if inwin.any():
                    ipk = grid[inwin][np.nanargmax(motion[inwin])]
                    ev['interval_peak_rel_to_start'] = float(ipk - s)
                    ev['interval_peak_frac'] = float((ipk - s) / (e - s))
                    out_mask = ok & ((grid < s - .5) | (grid > e + .5))
                    ev['motion_ratio_in_vs_out'] = float(np.nanpercentile(motion[inwin], 90) /
                                                         np.nanpercentile(motion[out_mask], 90)) if out_mask.any() else None
                else:
                    ipk = None
                pre = (times[0], s); post = (e, times[-1])
                ev['coverage_in'] = coverage(times, s, e)
                ev['coverage_pre'] = coverage(times, *pre); ev['coverage_post'] = coverage(times, *post)
                ev['max_gap_in'] = max_gap(times, s, e)
                ev['max_gap_out'] = max(max_gap(times, *pre), max_gap(times, *post))
                mid = (s + e) / 2
                rules = {
                    'v2_contained': lambda w0, w1: w0 >= s + .25 and w1 <= e - .25,
                    # annotation-only: window holds the central 1.5s of the operator interval
                    'midpoint_centered': lambda w0, w1: w0 <= mid - .75 and w1 >= mid + .75,
                    # annotation-only: >=1.5s overlap with the interval
                    'overlap_1p5s': lambda w0, w1: min(w1, e) - max(w0, s) >= 1.5,
                }
                if ipk is not None:
                    rules['motion_peak_diag'] = lambda w0, w1: w0 <= ipk - .5 and w1 >= ipk + .5
                res = {}
                for name, rule in rules.items():
                    pass_ends, relax_ends, cand = [], [], 0
                    for w0 in np.arange(times[0], times[-1] - 3 + 1e-9, .5):
                        w1 = w0 + 3
                        if not rule(w0, w1):
                            continue
                        cand += 1
                        if not window_quality(times, float(w0), cfg)['reasons']:
                            pass_ends.append(w1)
                        if not window_quality(times, float(w0), relaxed)['reasons']:
                            relax_ends.append(w1)
                    res[name] = dict(candidates=cand, passing=len(pass_ends), longest_run=runs(np.array(pass_ends)),
                                     passing_gap_0p5=len(relax_ends), longest_run_gap_0p5=runs(np.array(relax_ends)))
                ev['rules'] = res
                # does each v2 positive window contain the in-interval motion peak?
                if ipk is not None:
                    v2 = [w0 for w0 in np.arange(times[0], times[-1] - 3 + 1e-9, .5)
                          if label_window(w0, w0 + 3, status, intervals) == 1
                          and not window_quality(times, float(w0), cfg)['reasons']]
                    ev['v2_positives_containing_peak'] = int(sum(w0 <= ipk <= w0 + 3 for w0 in v2))
                    ev['v2_positives'] = len(v2)
                out['events'].append(ev)

    # ---- accepted-window shortcut features (existing dataset; read-only) ----
    with np.load(a.dataset / 'windows.npz', allow_pickle=False) as z:
        X = z['X']; y = z['y']; grp = z['group']; rids = z['recording_id']; st = z['start']; eid = z['event_id']
    act = {r['recording_id']: r['activity'] for r in out['recordings']}
    level = np.median(X, axis=(1, 2))
    rel = X / np.maximum(np.median(X, axis=2, keepdims=True), 1e-6)
    relmotion = rel.std(axis=2).mean(axis=1)
    diffmotion = np.abs(np.diff(rel, axis=2)).mean(axis=(1, 2))
    interp = np.zeros(len(X))
    for i in range(len(X)):
        t = rec_cache[str(rids[i])][0]; w0 = st[i]
        tt = np.concatenate([[w0], t[(t > w0) & (t < w0 + 3)], [w0 + 3]]); d = np.diff(tt)
        interp[i] = d[d > .1].sum() / 3
    feats = dict(level=level, relmotion=relmotion, diffmotion=diffmotion, interp_fraction=interp)
    activities = np.array([act[str(r)] for r in rids])
    sc = {}
    for g in GROUPS:
        gm = grp == g
        for name, f in feats.items():
            sc[f'{g}:{name}'] = dict(
                fall_vs_all_normal=auc(f[gm & (y == 1)], f[gm & (y == 0)]),
                fall_vs_dynamic=auc(f[gm & (y == 1)], f[gm & (y == 0) & np.isin(activities, list(DYNAMIC))]),
                fall_median=float(np.median(f[gm & (y == 1)])), normal_median=float(np.median(f[gm & (y == 0)])))
    for name, f in feats.items():
        sc[f'session04_vs_03_normal:{name}'] = auc(f[(grp == GROUPS[1]) & (y == 0)], f[(grp == GROUPS[0]) & (y == 0)])
        sc[f'session04_vs_03_fall:{name}'] = auc(f[(grp == GROUPS[1]) & (y == 1)], f[(grp == GROUPS[0]) & (y == 1)])
    out['shortcut_features'] = sc
    # per-session per-activity amplitude level (median of window medians)
    out['level_by_activity'] = {f'{g}:{ac}': float(np.median(level[(grp == g) & (activities == ac)]))
                                for g in GROUPS for ac in sorted(set(activities[grp == g]))}
    out['level_by_activity'].update({f'{g}:FALL_WINDOWS': float(np.median(level[(grp == g) & (y == 1)])) for g in GROUPS})

    # ---- eligible-window structural ceiling under the fixed alert policy ----
    ceiling = []
    for e in sorted(set(eid.tolist()) - {''}):
        m = eid == e
        ceiling.append(dict(event_id=e, group=str(grp[m][0]), eligible=int(m.sum()),
                            longest_run=runs(st[m] + 3), detectable=runs(st[m] + 3) >= ALERT_POLICY['warmup_windows']))
    out['alert_ceiling'] = ceiling

    # ---- Fold B CNN counterfactual inference (saved weights; no training) ----
    import torch
    from cnn_baseline.model import build_model
    from lahacks_cv.common import normalize
    saved = torch.load(a.artifacts / 'cnn.pt', map_location='cpu', weights_only=True)
    pre = json.loads((a.artifacts / 'preprocessing.json').read_text())
    assert pre['train_group'] == GROUPS[1]
    model = build_model(saved['architecture']); model.load_state_dict(saved['state_dict']); model.eval()
    sc_mean = np.asarray(saved['normalization']['mean'], np.float32)

    def prob(x):
        with torch.inference_mode():
            return np.concatenate([model(torch.from_numpy(normalize(x[i:i + 64], saved['normalization']))).softmax(-1)[:, 1].numpy()
                                   for i in range(0, len(x), 64)])
    probe = {}
    s03 = grp == GROUPS[0]; s04 = grp == GROUPS[1]
    ratio = float(np.median(level[s04 & (y == 0)]) / np.median(level[s03 & (y == 0)]))
    variants = {
        'original': lambda x: x,
        'per_window_dc_replaced_by_train_mean': lambda x: (x - x.mean(axis=2, keepdims=True) + sc_mean[None, :, None]).astype(np.float32),
        'global_gain_to_s04_normal_level': lambda x: (x * ratio).astype(np.float32),
    }
    for vname, fn in variants.items():
        for gname, gm in (('s03_validation', s03), ('s04_training', s04)):
            pr = prob(fn(X[gm])); yy = y[gm]
            probe[f'{vname}:{gname}'] = dict(predicted_fall_fraction=float((pr >= .5).mean()),
                normal_fp_fraction=float((pr[yy == 0] >= .5).mean()), fall_recall=float((pr[yy == 1] >= .5).mean()),
                auc=auc(pr[yy == 1], pr[yy == 0]))
    probe['gain_ratio_used'] = ratio
    probe['caution'] = 'Diagnostic sensitivity probe of an existing checkpoint; not a model selection or reportable score.'
    probe['prob_vs_level_spearman_s03'] = float(np.corrcoef(prob(X[s03]).argsort().argsort(), level[s03].argsort().argsort())[0, 1])
    out['cnn_fold_b_probe'] = probe

    import joblib
    rf = joblib.load(a.artifacts / 'rf.joblib')['model']
    fam = ['var', 'absdiff', 'diffvar', 'var_t1', 'var_t2', 'var_t3', 'peak_end_ratio', 'energy', 'band_0_1', 'band_1_3', 'band_3_10']
    imp = rf.feature_importances_
    out['rf_fold_b_importance_by_family'] = {f: float(imp[i * 52:(i + 1) * 52].sum()) for i, f in enumerate(fam)}
    out['rf_fold_b_importance_by_family']['pca3'] = float(imp[-3:].sum())
    (Path(__file__).parent / 'analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('ok')


if __name__ == '__main__':
    main()

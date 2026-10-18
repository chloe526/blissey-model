"""Read-only realistic degradation: transplant the missing-time pattern of REAL rejected windows onto clean windows.
For each real rejected window R (reasons != []), mark its internal gaps >0.1 s (and unsupported edges) as missing,
delete the clean window's packets in those spans, re-interpolate, and measure model decision flips vs clean.
Reported by R's max gap. Also counts how much data/events each candidate gate would recover."""
import json, sys, warnings, collections; from pathlib import Path; import numpy as np
warnings.simplefilter('ignore'); sys.path.insert(0, '.')
import joblib
from csi import load_recording; from preprocess import recording_features, label_window
from csi_quality import window_quality; from lahacks_cv.data import geometry
from lahacks_cv.features import extract_one
from lahacks_v2.windows import DEFAULT_RAW, gain_normalize, role
from lahacks_v2.features import motion_score
cfg = geometry(); IDX = cfg['channel_indices']
B = joblib.load('reports/ml_v2/final/model.joblib'); rf, thr = B['model'], B['threshold']
rng = np.random.default_rng(1)
def interp(t, A, w0):
    g = np.linspace(w0, w0 + 3, 512); return np.stack([np.interp(g, t, A[:, i]) for i in range(A.shape[1])]).astype(np.float32)
clean, rejected, allwin = [], [], []
for f in sorted(DEFAULT_RAW.glob('session_0*_LAHACKS/*/*')):
    m, rows, status, iv, _, _ = load_recording(f); t, A, _ = recording_features(rows, IDX); g = f.parts[-3]
    for w0 in np.arange(t[0], t[-1] - 3 + 1e-9, .5):
        w0 = float(w0); q = window_quality(t, w0, cfg)
        r, k = role(w0, w0 + 3, status, iv)
        allwin.append(dict(group=g, rid=m['recording_id'], role=r, event=f"{m['recording_id']}:{k}" if k != '' else '',
                           label=label_window(w0, w0 + 3, status, iv), pass_v2=not q['reasons'], **{k_: q[k_] for k_ in
                           ('max_gap_seconds', 'occupied_bin_fraction', 'max_edge_distance_seconds', 'effective_rate_hz', 'sample_span_fraction')}))
        if not q['reasons'] and q['max_gap_seconds'] <= .15: clean.append((t, A, w0))
        elif q['reasons']:
            inside = t[(t >= w0) & (t <= w0 + 3)] - w0
            edges = np.concatenate([[0.], inside, [3.]]); d = np.diff(edges)
            spans = [(edges[j], edges[j + 1]) for j in np.flatnonzero(d > .1)]
            rejected.append(dict(spans=spans, max_gap=q['max_gap_seconds'], occ=q['occupied_bin_fraction']))
bins = [(.25, .35), (.35, .5), (.5, .75), (.75, 1.0), (1.0, 99)]
res = dict(clean=len(clean), rejected=len(rejected), threshold=thr, by_real_max_gap={})
for lo, hi in bins:
    R = [r for r in rejected if lo < r['max_gap'] <= hi]
    flips, dps, occ = [], [], []
    for r in [R[i] for i in rng.choice(len(R), min(300, len(R)), replace=False)] if R else []:
        t, A, w0 = clean[rng.integers(len(clean))]
        x0 = gain_normalize(interp(t, A, w0)); p0 = rf.predict_proba(extract_one(x0)[None])[0, 1]
        rel = t - w0; keep = np.ones(len(t), bool)
        for a, b in r['spans']: keep &= ~((rel > a) & (rel < b))
        # keep one bracketing packet on each side as transform_window does
        x1 = gain_normalize(interp(t[keep], A[keep], w0)); p1 = rf.predict_proba(extract_one(x1)[None])[0, 1]
        flips.append((p1 >= thr) != (p0 >= thr)); dps.append(abs(p1 - p0)); occ.append(r['occ'])
    res['by_real_max_gap'][f'{lo}-{hi}'] = dict(real_rejected_windows=len(R), tested=len(flips),
        decision_flip_rate=float(np.mean(flips)) if flips else None, p95_abs_dprob=float(np.quantile(dps, .95)) if dps else None,
        median_occupied=float(np.median(occ)) if occ else None)
    print(f'real max gap {lo}-{hi}s: n={len(R)} flips {np.mean(flips) if flips else float("nan"):.1%} p95|dp| {np.quantile(dps,.95) if dps else float("nan"):.3f} occ {np.median(occ) if occ else float("nan"):.2f}')
# what each gate recovers
def gate(w, G):
    return w['max_gap_seconds'] <= G and w['max_edge_distance_seconds'] <= G / 2 + 1e-9
summary = {}
for name, G in (('v2 (current)', None), ('gap<=0.35', .35), ('gap<=0.5', .5), ('gap<=0.75', .75)):
    ok = [w for w in allwin if (w['pass_v2'] if G is None else gate(w, G))]
    s = {}
    for g in ('session_03_LAHACKS', 'session_04_LAHACKS'):
        nr = [w for w in allwin if w['group'] == g and w['role'] == 'normal_recording']
        s[g[:10]] = dict(normal_pass_rate=round(sum(w in ok for w in nr) / len(nr), 3) if False else None)
    def frac(pred): 
        tot = [w for w in allwin if pred(w)]; return round(sum(1 for w in tot if (w['pass_v2'] if G is None else gate(w, G))) / len(tot), 3)
    summary[name] = dict(
        normal_pass_rate_s03=frac(lambda w: w['group'] == 'session_03_LAHACKS' and w['role'] == 'normal_recording'),
        normal_pass_rate_s04=frac(lambda w: w['group'] == 'session_04_LAHACKS' and w['role'] == 'normal_recording'),
        normal_windows=sum(1 for w in ok if w['role'] == 'normal_recording'),
        positive_v2_windows=sum(1 for w in ok if w['label'] == 1),
        events_with_v2_positive=len({w['event'] for w in ok if w['label'] == 1}),
        events_with_in_event_window=len({w['event'] for w in ok if w['role'] == 'in_event'}))
    print(name, summary[name])
res['gates'] = summary
json.dump(res, open('reports/capture_review/gap_transplant.json', 'w'), indent=1)

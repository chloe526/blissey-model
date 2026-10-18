"""Read-only synthetic degradation: how much do gaps of length L change the v2 model's output?
Take clean windows (all v2 gates pass AND max gap <= 0.15 s), delete packets in one interval of length L,
re-interpolate exactly as transform_window does, and compare model probability / E1 score with the clean version.
Label-free test of whether interpolated data is 'bad data' for this representation."""
import json, sys, warnings; from pathlib import Path; import numpy as np
warnings.simplefilter('ignore'); sys.path.insert(0, '.')
import joblib
from csi import load_recording; from preprocess import recording_features
from csi_quality import window_quality; from lahacks_cv.data import geometry
from lahacks_cv.features import extract_one
from lahacks_v2.windows import DEFAULT_RAW, gain_normalize
from lahacks_v2.features import motion_score
cfg = geometry(); IDX = cfg['channel_indices']
B = joblib.load('reports/ml_v2/final/model.joblib'); rf, thr = B['model'], B['threshold']
rng = np.random.default_rng(0)
def interp(t, A, w0):
    g = np.linspace(w0, w0 + 3, 512); return np.stack([np.interp(g, t, A[:, i]) for i in range(A.shape[1])]).astype(np.float32)
samples = []
for f in sorted(DEFAULT_RAW.glob('session_0*_LAHACKS/*/*')):
    m, rows, status, iv, _, _ = load_recording(f); t, A, _ = recording_features(rows, IDX)
    for w0 in np.arange(t[0], t[-1] - 3 + 1e-9, .5):
        q = window_quality(t, float(w0), cfg)
        if q['reasons'] or q['max_gap_seconds'] > .15: continue
        samples.append((t, A, float(w0), status))
idx = rng.choice(len(samples), min(500, len(samples)), replace=False)
Ls = (.25, .35, .5, .75, 1.0); out = {L: dict(dp=[], flip=[], de1=[]) for L in Ls}
e1_clean = []
for i in idx:
    t, A, w0, _ = samples[i]
    x0 = interp(t, A, w0); g0 = gain_normalize(x0); p0 = rf.predict_proba(extract_one(g0)[None])[0, 1]; s0 = np.log(motion_score(g0)); e1_clean.append(s0)
    for L in Ls:
        c = w0 + rng.uniform(.3 + L / 2, 2.7 - L / 2) if L < 2.4 else w0 + 1.5
        keep = (t <= c - L / 2) | (t >= c + L / 2)
        g1 = gain_normalize(interp(t[keep], A[keep], w0)); p1 = rf.predict_proba(extract_one(g1)[None])[0, 1]
        out[L]['dp'].append(abs(p1 - p0)); out[L]['flip'].append((p1 >= thr) != (p0 >= thr)); out[L]['de1'].append(np.log(motion_score(g1)) - s0)
sd = float(np.std(e1_clean)); res = dict(clean_windows_available=len(samples), tested=len(idx), threshold=thr, e1_clean_sd=sd)
for L in Ls:
    o = out[L]; res[str(L)] = dict(median_abs_dprob=float(np.median(o['dp'])), p95_abs_dprob=float(np.quantile(o['dp'], .95)),
        decision_flip_rate=float(np.mean(o['flip'])), median_e1_shift_in_sd=float(np.median(o['de1']) / sd), p95_abs_e1_shift_in_sd=float(np.quantile(np.abs(o['de1']), .95) / sd))
    print(f"gap {L:.2f}s  |dprob| med {res[str(L)]['median_abs_dprob']:.3f} p95 {res[str(L)]['p95_abs_dprob']:.3f}  flips {res[str(L)]['decision_flip_rate']:.1%}  E1 shift med {res[str(L)]['median_e1_shift_in_sd']:+.2f}sd p95|.| {res[str(L)]['p95_abs_e1_shift_in_sd']:.2f}sd")
json.dump(res, open('reports/capture_review/gap_tolerance.json', 'w'), indent=1)

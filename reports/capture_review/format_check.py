"""Read-only: (1) normal-recording window pass rates; (2) do legacy vs HT40 packets differ only by scale?"""
import json, sys; from pathlib import Path; import numpy as np
sys.path.insert(0, '.')
from csi_quality import window_quality
from lahacks_cv.data import geometry
RAW = Path('/private/tmp/blissey-stage1-06c97032c93a'); IDX = list(range(6, 32)) + list(range(33, 59)); cfg = geometry()
res = {}
for g in ('session_03_LAHACKS', 'session_04_LAHACKS'):
    tot = ok = ok_ht = tot_ht = 0; shape_corr = []; scale = []
    for f in sorted((RAW / g).glob('*/*/measurements.jsonl')):
        act = f.parent.parent.name
        P = [json.loads(l) for l in f.open()]
        t = np.array([p['elapsed_seconds'] for p in P]); A = np.array([np.hypot(*np.array(p['csi'], float).reshape(64, 2).T)[IDX] for p in P])
        ht = np.array([p['firmware_fields'][5] == '1' for p in P])
        _, u = np.unique(t, return_index=True); t, A, ht = t[u], A[u], ht[u]
        if ht.sum() > 20 and (~ht).sum() > 20:
            mh, ml = np.median(A[ht], 0), np.median(A[~ht], 0)
            shape_corr.append(np.corrcoef(mh / mh.mean(), ml / ml.mean())[0, 1]); scale.append(ml.mean() / mh.mean())
        if act == 'staged_fall':
            continue
        for sel, name in ((np.ones(len(t), bool), 'all'), (ht, 'ht')):
            tt = t[sel]
            for w0 in np.arange(tt[0], tt[-1] - 3 + 1e-9, .5):
                q = not window_quality(tt, float(w0), cfg)['reasons']
                if name == 'all': tot += 1; ok += q
                else: tot_ht += 1; ok_ht += q
    res[g] = dict(normal_window_pass_rate_all_packets=ok / tot, normal_window_pass_rate_ht_only=ok_ht / tot_ht,
                  legacy_vs_ht_subcarrier_shape_corr_median=float(np.median(shape_corr)),
                  legacy_vs_ht_shape_corr_min=float(np.min(shape_corr)), legacy_over_ht_scale_median=float(np.median(scale)))
    print(g, {k: round(v, 3) for k, v in res[g].items()})
json.dump(res, open('reports/capture_review/format_check.json', 'w'), indent=1)

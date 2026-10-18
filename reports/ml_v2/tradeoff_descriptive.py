"""DESCRIPTIVE ONLY: event detection vs false alerts across thresholds on each validation block.
Not used for model or threshold selection (that would tune on held-out blocks)."""
import json, warnings; from pathlib import Path; import numpy as np
warnings.simplefilter('ignore')
from lahacks_cv.common import FOLDS
from lahacks_v2.windows import build
from lahacks_v2.evaluate import evaluate
out = Path(__file__).parent; data, records = build()
res = {}
for name in ('E1_motion_gain', 'E2_rf_gain', 'E2_rf_raw'):
    s = np.load(out / f'scores_{name}.npz')['score'].astype(float); res[name] = {}
    for fold, (_, vb) in FOLDS.items():
        val = data['group'] == vb; rows = []
        for q in (.9, .95, .98, .99, .995, .999):
            t = float(np.quantile(s[val & (data['role'] == 'normal_recording')], q))
            r = evaluate(data, records, val, s, t)
            rows.append(dict(val_normal_quantile=q, detected=r['detected_events'], alertable=r['alertable_events'],
                             fa_per_hour=round(r['false_alerts_per_hour_combined'], 1)))
        res[name][fold] = rows
(out / 'tradeoff_descriptive.json').write_text(json.dumps(res, indent=1))
for n, v in res.items():
    for f, rows in v.items():
        print(n, f, ' '.join(f"q{r['val_normal_quantile']}:{r['detected']}/{r['alertable']}@{r['fa_per_hour']}" for r in rows))

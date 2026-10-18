"""E4: does the gap-tolerant gate (max gap 0.5 s) help or hurt, without changing anything else?

Same model (RF on gain-normalized windows, v1 config), same v2 label rule, same folds, same
pre-declared OOF 95th-percentile threshold. Only the training-window gate changes. Both trained
models are scored on BOTH fixed evaluation populations (v2-gate windows, and relaxed-gate windows).
Also checks the shortcut risk: does interpolation amount alone predict the label?
"""
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from lahacks_cv.common import FOLDS, ROOT, write_json
from lahacks_cv.features import extract as rf_extract
from .evaluate import evaluate
from .run import fit_predict, oof_threshold
from .windows import build, gain_normalize

MAX_GAP = .5


def main():
    warnings.simplefilter('ignore')
    data, records = build(max_gap=MAX_GAP)
    G = np.empty_like(data['X'])
    for i in range(0, len(G), 256):
        G[i:i + 256] = gain_normalize(data['X'][i:i + 256])
    F = rf_extract(G); del G
    y = data['label_v2']; rid = data['recording_id']; v2 = data['pass_v2']
    out = dict(max_gap=MAX_GAP, windows=len(y), windows_pass_v2=int(v2.sum()),
               labeled=dict(v2=np.bincount(y[(y >= 0) & v2]).tolist(), relaxed=np.bincount(y[y >= 0]).tolist()))
    # Shortcut check: interpolation amount / max gap alone vs label, per block, on the relaxed population
    out['shortcut_auc'] = {}
    for g in FOLDS['A']:
        m = (data['group'] == g) & (y >= 0)
        for name in ('interp_fraction', 'max_gap'):
            out['shortcut_auc'][f'{g}:{name}'] = float(roc_auc_score(y[m], data[name][m]))
    for train_gate in ('v2', 'relaxed'):
        for fold, (tb, vb) in FOLDS.items():
            block = (data['group'] == tb) & (y >= 0)
            train = np.flatnonzero(block & v2) if train_gate == 'v2' else np.flatnonzero(block)
            thr = oof_threshold('rf', F, y, rid, train, ROOT)
            val_all = data['group'] == vb
            score = np.zeros(len(y)); score[val_all] = fit_predict('rf', F, y, rid, train, np.flatnonzero(val_all), ROOT)
            for pop, mask in (('eval_v2', val_all & v2), ('eval_relaxed', val_all)):
                r = evaluate(data, records, mask, score, thr); r.pop('per_event')
                out[f'train_{train_gate}:{fold}:{pop}'] = dict(r, train_windows=np.bincount(y[train]).tolist())
                w = r['window']
                print(f"train {train_gate:7s} fold {fold} on {pop:12s} AUC {w['roc_auc']:.3f} "
                      f"det {r['detected_events']}/{r['alertable_events']} (of {r['annotated_events']}) "
                      f"FA/h {r['false_alerts_per_hour_combined']:.0f} normal-min {r['normal_exposure_min']:.1f}", flush=True)
    print('shortcut AUCs', {k: round(v, 2) for k, v in out['shortcut_auc'].items()})
    write_json(ROOT / 'reports/ml_v2/e4_gate.json', out)


if __name__ == '__main__':
    main()

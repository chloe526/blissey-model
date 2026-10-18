"""Run E0 (evaluation v2) and experiments E1/E2/E3/E5 on fixed folds A (03->04) and B (04->03).

Rules: every fitted quantity uses the training block only. Thresholds for trained models are
the 95th percentile of out-of-fold (3-fold, recording-grouped) training-block normal scores;
for the unfitted E1 score, the 95th percentile of training-block normal scores. The validation
block is scored once. Outputs are small JSON/NPZ files under reports/ml_v2/.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lahacks_cv.common import FOLDS, ROOT, class_weights, fit_normalization, write_json
from .evaluate import evaluate
from .features import extract, with_context
from .windows import DEFAULT_RAW, build, gain_normalize, onset_labels

RUN = json.loads((ROOT / 'lahacks_cv/config.json').read_text())
Q = .95
LR = dict(C=1.0, class_weight='balanced', max_iter=2000)


def rf_model(y):
    w = class_weights(y)
    return RandomForestClassifier(**RUN['rf'], class_weight={0: float(w[0]), 1: float(w[1])},
                                  random_state=RUN['seed'], n_jobs=RUN['threads'])


def fit_predict(kind, inputs, y, rid, train, predict, workdir):
    if kind == 'rf':
        m = rf_model(y[train]).fit(inputs[train], y[train]); return m.predict_proba(inputs[predict])[:, 1]
    if kind == 'lr':
        m = make_pipeline(StandardScaler(), LogisticRegression(**LR)).fit(inputs[train], y[train])
        return m.predict_proba(inputs[predict])[:, 1]
    if kind == 'cnn':
        from lahacks_cv.train import fit_cnn, predict_cnn
        d = dict(X=inputs, y=y, recording_id=rid); scaler = fit_normalization(inputs, train)
        with tempfile.TemporaryDirectory(dir=workdir) as tmp:
            model, _, _ = fit_cnn(d, train, scaler, dict(RUN, device='cpu'), 'cpu', Path(tmp))
        return predict_cnn(model, d, predict, scaler, RUN['cnn'], 'cpu')
    raise ValueError(kind)


def oof_threshold(kind, inputs, y, rid, train, workdir):
    """95th percentile of out-of-fold normal scores inside the training block only."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RUN['seed'])
    normals = []
    for a, b in cv.split(train, y[train], rid[train]):
        tr, te = train[a], train[b]
        if len(set(y[tr].tolist())) < 2:
            continue
        try:
            s = fit_predict(kind, inputs, y, rid, tr, te, workdir)
        except ValueError as exc:      # CNN early stopping needs >=2 fall recordings in the split
            if 'two contributing recordings' not in str(exc):
                raise
            continue
        normals.append(s[y[te] == 0])
    if not normals:
        raise ValueError('No feasible out-of-fold split for threshold')
    return float(np.quantile(np.concatenate(normals), Q))


def onset_offsets(data, records, score, train_block):
    """Fit E3 onset window [lo, hi] (s after action_start) from training-block fall events only."""
    centres, z = [], []
    for rid, rec in records.items():
        if rec['group'] != train_block or rec['status'] != 'annotated_fall':
            continue
        rows = np.flatnonzero(data['recording_id'] == rid)
        s = score[rows]; s = (s - np.median(s)) / (s.std() + 1e-9)
        a = rec['intervals'][0][0]
        centres.extend(data['start'][rows] + 1.5 - a); z.extend(s)
    centres, z = np.array(centres), np.array(z)
    bins = np.arange(-2, 8.01, .5)
    prof = np.array([z[np.abs(centres - c) <= .25].mean() if (np.abs(centres - c) <= .25).any() else -np.inf
                     for c in bins])
    peak = int(np.argmax(prof)); keep = prof >= .5 * prof[peak]
    lo = hi = peak
    while lo > 0 and keep[lo - 1]:
        lo -= 1
    while hi < len(bins) - 1 and keep[hi + 1]:
        hi += 1
    return float(bins[lo] - .25), float(bins[hi] + .25), dict(zip(bins.tolist(), np.round(prof, 2).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/ml_v2')
    p.add_argument('--only', nargs='*', default=None)
    a = p.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    work = out / '_tmp'; work.mkdir(exist_ok=True)
    import torch; torch.set_num_threads(RUN['threads'])
    t0 = time.time()
    data, records = build(a.raw_root)
    X = data['X']; G = np.empty_like(X)
    for i in range(0, len(X), 256):
        G[i:i + 256] = gain_normalize(X[i:i + 256])
    print(f'windows {len(X)} built in {time.time() - t0:.0f}s', flush=True)
    F = extract(G); Fctx = with_context(F, data); e1 = F[:, 0].astype(np.float64)
    rid = data['recording_id']; yv2 = data['label_v2']
    cache = out / 'rf_features.npz'
    if cache.exists():
        z = np.load(cache); RF_RAW, RF_GAIN = z['raw'], z['gain']
    else:
        from lahacks_cv.features import extract as rf_extract
        RF_RAW = rf_extract(X); RF_GAIN = rf_extract(G)
        np.savez_compressed(cache, raw=RF_RAW, gain=RF_GAIN)
    print(f'features ready {time.time() - t0:.0f}s', flush=True)

    meta = dict(windows=len(X), by_role={r: int((data['role'] == r).sum()) for r in np.unique(data['role'])},
                labeled_v2=np.bincount(yv2[yv2 >= 0]).tolist(), quantile=Q, folds=FOLDS,
                status='development cross-validation only; one participant; no frozen test block')
    results = json.loads((out / 'results.json').read_text()) if (out / 'results.json').exists() else {}
    results['_meta'] = meta

    def save():
        write_json(out / 'results.json', results)

    # ---- v1 saved Fold B checkpoints, inference only, raw input, v1 threshold 0.5 ----
    want = lambda n: a.only is None or n in a.only
    if want('v1_saved_foldB') and 'v1_saved_foldB_rf' not in results:
        import joblib, torch
        from cnn_baseline.model import build_model
        from lahacks_cv.common import normalize
        art = ROOT / 'reports/lahacks_job_d4c10e9207e8/artifacts'
        val = data['group'] == FOLDS['B'][1]
        rf = joblib.load(art / 'rf.joblib')['model']
        s = np.zeros(len(X)); s[val] = rf.predict_proba(RF_RAW[val])[:, 1]
        results['v1_saved_foldB_rf'] = {'B': evaluate(data, records, val, s, .5)}
        saved = torch.load(art / 'cnn.pt', map_location='cpu', weights_only=True)
        m = build_model(saved['architecture']); m.load_state_dict(saved['state_dict']); m.eval()
        idx = np.flatnonzero(val); s = np.zeros(len(X))
        with torch.inference_mode():
            for i in range(0, len(idx), 128):
                j = idx[i:i + 128]
                s[j] = m(torch.from_numpy(normalize(X[j], saved['normalization']))).softmax(-1)[:, 1].numpy()
        results['v1_saved_foldB_cnn'] = {'B': evaluate(data, records, val, s, .5)}
        save(); print('v1 saved re-scored', flush=True)

    # ---- experiments ----
    experiments = [
        ('E1_motion_gain', 'none', None, 'v2'),
        ('E2_rf_raw', 'rf', RF_RAW, 'v2'),            # v1 RF config reproduced locally (Fold A lost)
        ('E2_rf_gain', 'rf', RF_GAIN, 'v2'),
        ('E5_lr_context_gain', 'lr', Fctx, 'v2'),
        ('E3_rf_gain_onset', 'rf', RF_GAIN, 'onset'),
        ('E3_lr_context_onset', 'lr', Fctx, 'onset'),
        ('E2_cnn_raw', 'cnn', X, 'v2'),
        ('E2_cnn_gain', 'cnn', G, 'v2'),
    ]
    for name, kind, inputs, labels in experiments:
        if not want(name) or name in results:
            continue
        tick = time.time(); results[name] = {}; scores = np.zeros(len(X))
        for fold, (train_block, val_block) in FOLDS.items():
            block = data['group'] == train_block; val = data['group'] == val_block
            extra = {}
            if labels == 'onset':
                lo, hi, prof = onset_offsets(data, records, e1, train_block)
                y = onset_labels(data, records, lo, hi); extra = dict(onset_window=[lo, hi], onset_profile=prof)
            else:
                y = yv2
            train = np.flatnonzero(block & (y >= 0))
            extra['train_class_counts'] = np.bincount(y[train], minlength=2).tolist()
            if kind == 'none':
                s = e1; thr = float(np.quantile(e1[block & (data['role'] == 'normal_recording')], Q))
                scores[val] = s[val]
            else:
                thr = oof_threshold(kind, inputs, y, rid, train, work)
                scores[val] = fit_predict(kind, inputs, y, rid, train, np.flatnonzero(val), work)
            r = evaluate(data, records, val, scores, thr)
            if kind != 'none':
                r['at_0.5'] = {k: v for k, v in evaluate(data, records, val, scores, .5).items() if k != 'per_event'}
            results[name][fold] = dict(r, **extra)
            print(name, fold, f"AUC {r['window']['roc_auc']:.3f} det {r['detected_events']}/{r['alertable_events']}"
                  f" FA/h {r['false_alerts_per_hour_combined']:.1f}", flush=True)
        results[name]['seconds'] = time.time() - tick
        np.savez_compressed(out / f'scores_{name}.npz', score=scores.astype(np.float32))
        save()
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

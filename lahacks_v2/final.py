"""Refit the selected configuration (RF on gain-normalized windows) on Sessions 03+04 and save it.

UNVALIDATED: there is no frozen test block. The threshold is the pre-declared 95th percentile of
out-of-fold (3-fold, recording-grouped) normal-window scores over the development data.
Also: CPU latency, and a raw-packet streaming replay checked against offline scores.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np

from lahacks_cv.common import ROOT, write_json
from lahacks_cv.data import geometry
from lahacks_cv.features import FEATURE_CONFIG, extract as rf_extract, extract_one
from lahacks_cv.metrics import ALERT_POLICY, alert_decisions
from preprocess import transform_window
from csi_quality import window_quality
from .run import RUN, oof_threshold, rf_model
from .windows import DEFAULT_RAW, build, gain_normalize

VERSION = 'blissey_rf_gain_v1'


class Detector:
    """Streaming detector: feed packets' (elapsed_seconds, 52 amplitudes); every 0.5 s emits a decision."""

    def __init__(self, bundle):
        self.model, self.threshold = bundle['model'], bundle['threshold']
        self.geometry = bundle['geometry']; self.policy = dict(bundle['alert_policy'], threshold=self.threshold)
        self.times, self.amps = deque(), deque()
        self.next_start = None; self.history = deque(maxlen=self.policy['history'])
        self.previous_end = None; self.last_alert = -float('inf')

    def probability(self, raw_window):
        return float(self.model.predict_proba(extract_one(gain_normalize(raw_window))[None])[0, 1])

    def push(self, t, amplitudes):
        """Returns a list of (window_end, probability or None if quality-rejected, alert)."""
        if self.times and t <= self.times[-1]:
            return []                                   # duplicate/out-of-order packet dropped (as offline)
        self.times.append(t); self.amps.append(np.asarray(amplitudes, np.float64))
        if self.next_start is None:
            self.next_start = t
        out = []
        while self.next_start + 3 < t:                  # need a packet after the window end
            start = self.next_start; self.next_start += .5
            times = np.fromiter(self.times, float)
            q = window_quality(times, start, self.geometry)
            if q['reasons']:
                out.append((start + 3, None, False)); continue
            p = self.probability(transform_window(times, np.stack(self.amps), start, self.geometry, normalize=False))
            end = start + 3
            if self.previous_end is not None and end - self.previous_end > self.policy['reset_after_gap_seconds'] + 1e-9:
                self.history.clear()
            self.history.append(p >= self.threshold); self.previous_end = end
            alert = (len(self.history) >= self.policy['warmup_windows'] and sum(self.history) >= self.policy['votes']
                     and end - self.last_alert >= self.policy['cooldown_seconds'])
            if alert:
                self.last_alert = end
            out.append((end, p, alert))
        while len(self.times) > 2 and self.times[1] < self.next_start - 1:   # keep a bracket packet before the window
            self.times.popleft(); self.amps.popleft()
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/ml_v2/final')
    a = p.parse_args(); out = a.output
    if (out / 'model.joblib').exists():
        raise FileExistsError('Final model exists; refusing to overwrite')
    out.mkdir(parents=True, exist_ok=True)
    data, records = build(a.raw_root)
    G = np.stack([gain_normalize(x) for x in data['X']])
    cache = ROOT / 'reports/ml_v2/rf_features.npz'
    F = np.load(cache)['gain'] if cache.exists() else rf_extract(G)
    if not np.allclose(F[:5], rf_extract(G[:5]), rtol=1e-5, atol=1e-6):
        raise ValueError('Cached features do not match current windows')
    y = data['label_v2']; rid = data['recording_id']; train = np.flatnonzero(y >= 0)
    threshold = oof_threshold('rf', F, y, rid, train, out)
    model = rf_model(y[train]).fit(F[train], y[train])
    bundle = dict(format=VERSION, model=model, threshold=threshold, geometry=geometry(), alert_policy=ALERT_POLICY,
                  transform='lahacks_v2.windows.gain_normalize then lahacks_cv.features.extract_one',
                  feature_config=FEATURE_CONFIG, trained_on=['session_03_LAHACKS', 'session_04_LAHACKS'],
                  train_class_counts=np.bincount(y[train]).tolist(),
                  status='UNVALIDATED: development refit; no frozen test block; one participant')
    joblib.dump(bundle, out / 'model.joblib', compress=3)
    det = Detector(joblib.load(out / 'model.joblib'))
    x = data['X'][0]
    for _ in range(5):
        det.probability(x)
    lat = []
    for _ in range(50):
        tick = time.perf_counter(); det.probability(x); lat.append((time.perf_counter() - tick) * 1000)
    # Streaming replay of whole recordings from raw packets vs offline batch scores
    from csi import load_recording
    from preprocess import recording_features
    replay = {}
    offline = model.predict_proba(F)[:, 1]
    for rid_ in [r for r in sorted(records) if records[r]['status'] == 'annotated_fall'][:3] + \
               [r for r in sorted(records) if records[r]['activity'] == 'walking'][:1]:
        rec = records[rid_]
        folder = next((a.raw_root / rec['group']).glob(f'*/{rid_}'))
        _, rows, *_ = load_recording(folder)
        t, amp, _ = recording_features(rows, geometry()['channel_indices'])
        d = Detector(bundle); emitted = []
        for ti, ai in zip(t, amp):
            emitted += d.push(float(ti), ai)
        live = {round(e, 3): (pr, al) for e, pr, al in emitted if pr is not None}
        mask = rid == rid_; off = {round(float(e), 3): s for e, s in zip(data['end'][mask], offline[mask])}
        common = sorted(set(live) & set(off))
        ends = np.array(sorted(off)); offline_alerts = alert_decisions(ends, np.array([off[e] for e in ends]),
                                                                        dict(ALERT_POLICY, threshold=threshold))
        replay[rid_] = dict(offline_windows=len(off), live_windows=len(live), matched=len(common),
                            max_abs_probability_difference=float(max(abs(live[e][0] - off[e]) for e in common)) if common else None,
                            live_alerts=int(sum(v[1] for v in live.values())), offline_alerts=int(offline_alerts.sum()))
    report = dict(bundle, model=None, threshold=threshold,
                  model_sha256=hashlib.sha256((out / 'model.joblib').read_bytes()).hexdigest(),
                  model_bytes=(out / 'model.joblib').stat().st_size,
                  cpu_latency_ms=dict(median=float(np.median(lat)), p95=float(np.quantile(lat, .95)),
                                      includes='gain transform + 575 features + RF; excludes packet parsing/interpolation'),
                  streaming_replay=replay)
    report.pop('model')
    write_json(out / 'model_card.json', report)
    print(json.dumps(dict(threshold=threshold, latency=report['cpu_latency_ms'], replay=replay), indent=1))


if __name__ == '__main__':
    main()

"""Evaluation v2 (E0): every quality-passing window is scored, as a live stream would be.

- Window AUC/AP: on the fixed v1 labeled population (label_v2), per fold.
- Alerts: unchanged v1 ALERT_POLICY (2-of-3 votes, warmup 3, 0.75 s reset, 5 s cooldown),
  run over ALL passing windows of each recording in time order, at a given threshold.
- Detection: an alert whose window ends in [action_start, action_end + 2 s].
- False alerts: alerts in normal recordings, and alerts in fall recordings ending before action_start.
  Post-event alerts (lying/get-up after the fall) are reported separately, not as false alerts.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from lahacks_cv.metrics import ALERT_POLICY, alert_decisions, union_seconds
from .windows import DETECT_AFTER


def contiguous_ok(ends, i, n=ALERT_POLICY['warmup_windows']):
    """Could the alert policy fire at window i at all (n contiguous windows ending at i)?"""
    if i < n - 1:
        return False
    return bool(np.all(np.diff(ends[i - n + 1:i + 1]) <= ALERT_POLICY['reset_after_gap_seconds'] + 1e-9))


def evaluate(data, records, mask, score, threshold):
    """mask: windows of the validation block; score: same length as data, used where mask."""
    policy = dict(ALERT_POLICY, threshold=float(threshold))
    y = data['label_v2'][mask]; s = score[mask]
    labeled = y >= 0
    window = dict(roc_auc=float(roc_auc_score(y[labeled], s[labeled])),
                  average_precision=float(average_precision_score(y[labeled], s[labeled])),
                  fall_recall=float((s[labeled & (y == 1)] >= threshold).mean()),
                  normal_fp_fraction=float((s[labeled & (y == 0)] >= threshold).mean()),
                  positives=int((y == 1).sum()), negatives=int((y == 0).sum()))
    events = []; fa_normal = fa_pre = post = 0; exp_normal = exp_pre = 0.
    groups = set(data['group'][mask].tolist())
    for rid, rec in sorted(records.items()):
        if rec['group'] not in groups:
            continue
        rows = np.flatnonzero(mask & (data['recording_id'] == rid))
        rows = rows[np.argsort(data['end'][rows])]
        ends = data['end'][rows]; starts = data['start'][rows]
        alerts = alert_decisions(ends, score[rows], policy) if len(rows) else np.zeros(0, bool)
        if rec['status'] == 'verified_non_fall':
            fa_normal += int(alerts.sum()); exp_normal += union_seconds(list(zip(starts, ends)))
            continue
        first_a = min(a for a, _ in rec['intervals']); last_b = max(b for _, b in rec['intervals'])
        pre = ends < first_a
        fa_pre += int((alerts & pre).sum()); exp_pre += union_seconds(list(zip(starts[pre], ends[pre])))
        post += int((alerts & (ends > last_b + DETECT_AFTER)).sum())
        for k, (a, b) in enumerate(rec['intervals']):
            win = (ends >= a) & (ends <= b + DETECT_AFTER)
            ceiling = any(contiguous_ok(ends, i) for i in np.flatnonzero(win))
            hit = np.flatnonzero(alerts & win)
            events.append(dict(event_id=f'{rid}:fall:{k}', alertable=bool(ceiling), detected=bool(len(hit)),
                               latency_from_start=float(ends[hit[0]] - a) if len(hit) else None))
    alertable = sum(e['alertable'] for e in events); detected = sum(e['detected'] for e in events)
    hours_n, hours_p = exp_normal / 3600, exp_pre / 3600
    return dict(threshold=float(threshold), window=window,
                annotated_events=len(events), alertable_events=alertable, detected_events=detected,
                recall_alertable=detected / alertable if alertable else None,
                recall_all_annotated=detected / len(events) if events else None,
                false_alerts_normal=fa_normal, normal_exposure_min=exp_normal / 60,
                false_alerts_per_hour_normal=fa_normal / hours_n if hours_n else None,
                false_alerts_pre_action=fa_pre, pre_action_exposure_min=exp_pre / 60,
                false_alerts_per_hour_combined=(fa_normal + fa_pre) / (hours_n + hours_p) if hours_n + hours_p else None,
                post_event_alerts=post, per_event=events)

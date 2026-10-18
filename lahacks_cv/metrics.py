"""Identical, predeclared window and causal alert evaluation for both baselines."""
from collections import deque
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

ALERT_POLICY = dict(threshold=.5, votes=2, history=3, cooldown_seconds=5.,
                    reset_after_gap_seconds=.75, warmup_windows=3,
                    decision_time='window end', tuning='none; fixed before outer-fold predictions')


def verify_rf_reload(probability, reloaded_probability):
    """Parallel tree reductions may differ by floating-point summation order."""
    probability = np.asarray(probability)
    reloaded_probability = np.asarray(reloaded_probability)
    if (probability.ndim != 1 or probability.shape != reloaded_probability.shape
            or not len(probability) or not np.isfinite(probability).all()
            or not np.isfinite(reloaded_probability).all()
            or np.any((probability < 0) | (probability > 1))
            or np.any((reloaded_probability < 0) | (reloaded_probability > 1))):
        raise ValueError('Invalid RF reload probability vectors')
    np.testing.assert_allclose(probability, reloaded_probability, rtol=1e-12, atol=1e-15, equal_nan=False)
    # Preserve exact window decisions and therefore the causal alert sequence.
    np.testing.assert_array_equal(probability >= ALERT_POLICY['threshold'],
                                  reloaded_probability >= ALERT_POLICY['threshold'])
    return dict(probability_rtol=1e-12, probability_atol=1e-15,
                maximum_absolute_difference=float(np.max(np.abs(probability - reloaded_probability))),
                threshold_decisions_identical=True)


def classification(y, probability):
    y = np.asarray(y); probability = np.asarray(probability)
    if y.shape != probability.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid probability vector')
    prediction = probability >= ALERT_POLICY['threshold']
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average='binary', zero_division=0)
    both = set(y.tolist()) == {0, 1}
    return dict(confusion_matrix=confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
                fall_precision=float(precision), fall_recall=float(recall), fall_f1=float(f1),
                roc_auc=float(roc_auc_score(y, probability)) if both else None,
                pr_auc_average_precision=float(average_precision_score(y, probability)) if both else None,
                predicted_fall_fraction=float(prediction.mean()), windows=len(y))


def union_seconds(spans):
    total = 0.; right = -float('inf')
    for a, b in sorted(spans):
        total += max(0., b - max(a, right)); right = max(right, b)
    return total


def alert_decisions(end, probability, policy=ALERT_POLICY):
    history = deque(maxlen=policy['history']); previous = None; last_alert = -float('inf')
    alerts = np.zeros(len(end), dtype=bool)
    for i, (time, p) in enumerate(zip(end, probability)):
        if previous is not None:
            if time <= previous:
                raise ValueError('Alert windows must be strictly time-ordered')
            if time - previous > policy['reset_after_gap_seconds'] + 1e-9:
                history.clear()
        history.append(p >= policy['threshold'])
        if (len(history) >= policy['warmup_windows'] and sum(history) >= policy['votes']
                and time - last_alert >= policy['cooldown_seconds']):
            alerts[i] = True; last_alert = time
        previous = time
    return alerts


def events(data, indices, probability, metadata):
    if len(indices) != len(probability):
        raise ValueError('Prediction alignment mismatch')
    per_event = []; false_alerts = 0; exposure = 0.
    groups = set(data['group'][indices].tolist())
    for rid, record in sorted(metadata['records'].items()):
        if record['group'] not in groups:
            continue
        local = np.flatnonzero(data['recording_id'][indices] == rid)
        local = local[np.argsort(data['end'][indices[local]])]
        rows = indices[local]
        alerts = alert_decisions(data['end'][rows], probability[local])
        if record['status'] == 'verified_non_fall':
            false_alerts += int(alerts.sum())
            exposure += union_seconds(list(zip(data['start'][rows], data['end'][rows])))
        for number, (a, b) in enumerate(record['intervals']):
            eligible = data['event_id'][rows] == f'{rid}:fall:{number}'
            hit = bool(np.any(alerts & eligible & (data['end'][rows] >= a) & (data['end'][rows] <= b)))
            per_event.append(dict(event_id=f'{rid}:fall:{number}', covered=bool(eligible.any()),
                                  eligible_windows=int(eligible.sum()), detected=hit))
    return event_summary(per_event, false_alerts, exposure)


def event_summary(per_event, false_alerts, exposure):
    covered = sum(e['covered'] for e in per_event); detected = sum(e['detected'] for e in per_event)
    return dict(annotated_events=len(per_event), quality_covered_events=covered,
                quality_uncovered_events=len(per_event) - covered, detected_events=detected,
                covered_event_recall=detected / covered if covered else None,
                all_annotated_event_recall=detected / len(per_event) if per_event else None,
                false_alert_episodes=int(false_alerts), accepted_normal_seconds=float(exposure),
                false_alerts_per_hour=false_alerts / (exposure / 3600) if exposure else None,
                per_event=per_event,
                limitations='Approximate operator action intervals; ignored windows absent; reset voting across missing windows. '
                            'Exposure is union of accepted verified-normal windows, including alert warmup. '
                            'Cooldown-defined episodes can repeat during a sustained positive run. No verified impact latency.')

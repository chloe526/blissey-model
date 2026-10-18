"""Local mirror of the backend contracts (blissey-beacon packages/contracts). Stdlib only.

InferenceEvent (exactly 7 fields) and DeviceHealth. Validating before sending means the Pi never puts
a payload in the outbox that the backend would reject as invalid_payload.
"""
from datetime import datetime

EVENT_FIELDS = ('event_id', 'room_id', 'device_id', 'event_type', 'confidence_scores', 'detected_at', 'sensor_status')
HEALTH_FIELDS = ('type', 'device_id', 'room_id', 'sent_at', 'sensor_status', 'model_status', 'last_valid_frame_at',
                 'outbox_depth', 'dropped_event_count', 'clock_synced')
SCORE_SUM_TOLERANCE = 1e-6


def _id(v):
    return isinstance(v, str) and 1 <= len(v) <= 128


def _datetime(v):
    if not isinstance(v, str) or 'T' not in v:
        return False
    try:
        d = datetime.fromisoformat(v.replace('Z', '+00:00'))
    except ValueError:
        return False
    return d.tzinfo is not None                      # timezone offset or Z required


def _score(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1


def event_errors(e):
    errs = []
    if not isinstance(e, dict) or set(e) != set(EVENT_FIELDS):
        return ['fields must be exactly %s' % (EVENT_FIELDS,)]
    for k in ('event_id', 'room_id', 'device_id'):
        if not _id(e[k]):
            errs.append('%s must be a 1-128 character string' % k)
    if e['event_type'] != 'inference_result':
        errs.append('event_type must be inference_result')
    cs = e['confidence_scores']
    if not isinstance(cs, dict) or set(cs) != {'fall', 'non_fall'} or not all(_score(cs[k]) for k in ('fall', 'non_fall')):
        errs.append('confidence_scores must be {fall, non_fall} in [0, 1]')
    elif abs(cs['fall'] + cs['non_fall'] - 1) > SCORE_SUM_TOLERANCE:
        errs.append('confidence_scores must sum to 1')
    if not _datetime(e['detected_at']):
        errs.append('detected_at must be ISO 8601 with timezone')
    if e['sensor_status'] != 'connected':
        errs.append('sensor_status must be connected for an inference event')
    return errs


def health_errors(h):
    errs = []
    if not isinstance(h, dict) or set(h) != set(HEALTH_FIELDS):
        return ['fields must be exactly %s' % (HEALTH_FIELDS,)]
    if h['type'] != 'device_health':
        errs.append('type must be device_health')
    for k in ('device_id', 'room_id'):
        if not _id(h[k]):
            errs.append('%s must be a 1-128 character string' % k)
    if not _datetime(h['sent_at']):
        errs.append('sent_at must be ISO 8601 with timezone')
    if h['sensor_status'] not in ('connected', 'degraded', 'disconnected'):
        errs.append('bad sensor_status')
    if h['model_status'] not in ('loading', 'ready', 'error'):
        errs.append('bad model_status')
    if h['last_valid_frame_at'] is not None and not _datetime(h['last_valid_frame_at']):
        errs.append('last_valid_frame_at must be ISO 8601 or null')
    for k in ('outbox_depth', 'dropped_event_count'):
        if not isinstance(h[k], int) or isinstance(h[k], bool) or h[k] < 0:
            errs.append('%s must be a non-negative integer' % k)
    if not isinstance(h['clock_synced'], bool):
        errs.append('clock_synced must be boolean')
    return errs

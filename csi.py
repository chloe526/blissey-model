"""Read-only parser for this project's legacy ESP32 LLTF recordings."""
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median

NORMAL_ACTIVITIES = frozenset(('empty_room', 'standing_still', 'sitting_still',
    'lying_still', 'walking', 'stand_up', 'sit_down', 'lie_down', 'get_up_from_floor'))


def seconds(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone required')
    return result.timestamp()


def packets(path):
    """Yield (line, packet, error); never silently lose malformed lines."""
    with Path(path).open() as stream:
        for line, text in enumerate(stream, 1):
            try:
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError('Expected object')
                validate_packet(value)
                yield line, value, None
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                yield line, None, str(exc)


def validate_packet(packet):
    csi = packet['csi']
    fields = packet['firmware_fields']
    # Source: firmware/_components/csi_component.h: 25 fields then raw LLTF.
    # Header len describes full CSI buffer (often 384), NOT the printed LLTF slice.
    if len(fields) != 25 or fields[0] != 'CSI_DATA' or len(csi) != 128:
        raise ValueError('Unsupported layout: require legacy 25-column / 128-byte LLTF')
    if any(type(x) is not int or not -128 <= x <= 127 for x in csi):
        raise ValueError('CSI must contain signed int8 I/Q pairs')
    if not math.isfinite(float(packet['elapsed_seconds'])) or packet['elapsed_seconds'] < 0:
        raise ValueError('Invalid elapsed timestamp')
    seconds(packet['host_time'])
    if int(fields[16]) <= 0 or int(fields[21]) != 0:
        raise ValueError('Invalid radio channel or RX state')
    return packet


def amplitudes(packet, indices):
    import numpy as np
    validate_packet(packet)
    pairs = np.asarray(packet['csi'], dtype=np.float64).reshape(64, 2)
    # ESP32 order: imaginary, real; hypot is invariant to the pair order.
    result = np.hypot(pairs[:, 0], pairs[:, 1])[indices]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite amplitudes')
    return result


def annotations(folder, metadata, valid_packets):
    """Use explicit category labels; align UI markers using shared UTC host time."""
    issues = []
    if metadata.get('status') != 'complete' or metadata.get('category_is_recording_label') is not True:
        return 'ambiguous', [], ['Recording category not explicitly verified']
    activity = metadata.get('activity')
    if activity in NORMAL_ACTIVITIES:
        return 'verified_non_fall', [], issues
    if activity != 'staged_fall':
        return 'ambiguous', [], ['Unknown activity']
    path = Path(folder) / 'events.jsonl'
    if not path.exists():
        return 'ambiguous', [], ['Fall trial has no events.jsonl; never infer negative']
    try:
        events = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if not valid_packets:
            raise ValueError('No valid measurements')
        origin = median(seconds(p['host_time']) - float(p['elapsed_seconds']) for p in valid_packets)
        pending, intervals, offsets = None, [], []
        for event in events:
            t = seconds(event['host_time']) - origin
            offsets.append(t - float(event['elapsed_seconds']))
            if event['event'] == 'action_start' and pending is None:
                pending = t
            elif event['event'] == 'action_end' and pending is not None and t > pending:
                intervals.append([pending, t]); pending = None
            else:
                raise ValueError('Unpaired, reversed, or unknown marker')
        if pending is not None or not intervals:
            raise ValueError('Missing start/end interval')
        lo, hi = min(p['elapsed_seconds'] for p in valid_packets), max(p['elapsed_seconds'] for p in valid_packets)
        if any(a < lo or b > hi for a, b in intervals):
            raise ValueError('Markers outside measurement exposure')
        if max(abs(x) for x in offsets) > 0.25:
            issues.append('UI elapsed differs from measurement clock; used host UTC alignment')
        issues.append('Operator action intervals are approximate staged-fall labels, not impact times')
        return 'annotated_fall', intervals, issues
    except (ValueError, KeyError, TypeError) as exc:
        return 'ambiguous', [], [str(exc)]


def load_recording(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'metadata.json').read_text())
    rows, errors = [], []
    for line, packet, error in packets(folder / 'measurements.jsonl'):
        if error: errors.append({'line': line, 'error': error})
        else: rows.append(packet)
    status, intervals, issues = annotations(folder, metadata, rows)
    return metadata, rows, status, intervals, issues, errors

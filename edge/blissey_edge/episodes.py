"""Group detector alerts into fall episodes and build schema-v2 inference events. Stdlib only.

One event per episode: after an event is emitted, further alerts in the next `refractory_seconds`
(default 60 s) are suppressed. This stops a single fall (plus lying still and getting up) from
creating several incidents. Event fields are exactly the 7 approved v2 fields.
"""
import os
import time
from datetime import datetime, timezone


def iso_utc(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat(timespec='milliseconds')


def boot_id():
    """Unique per service start: UTC start time + random suffix (event IDs never repeat across restarts)."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + os.urandom(3).hex()


class EpisodeGrouper:
    def __init__(self, device_id, room_id, refractory_seconds=60.0, run_id=None):
        if not isinstance(device_id, str) or not isinstance(room_id, str) or not device_id or not room_id:
            raise ValueError('device_id and room_id must be non-empty strings')
        self.device_id, self.room_id = device_id, room_id
        self.refractory = refractory_seconds
        self.run_id = run_id or boot_id()
        self.seq = 0
        self.last_emit = float('-inf')
        self.suppressed = 0

    def offer(self, decision, monotonic_now, wall_now=None):
        """Detector Decision -> v2 event dict if it starts a new episode, else None."""
        if not decision.alert or decision.fall_score is None:
            return None
        if monotonic_now - self.last_emit < self.refractory:
            self.suppressed += 1
            return None
        self.last_emit = monotonic_now
        self.seq += 1
        score = getattr(decision, 'episode_score', None)
        p = min(max(float(decision.fall_score if score is None else score), 0.0), 1.0)
        return {
            'event_id': 'pi%s-%s-%06d' % (self.device_id, self.run_id, self.seq),
            'room_id': self.room_id,
            'device_id': self.device_id,
            'event_type': 'inference_result',
            'confidence_scores': {'fall': p, 'non_fall': 1.0 - p},
            'detected_at': iso_utc(wall_now),
            'sensor_status': 'connected',
        }

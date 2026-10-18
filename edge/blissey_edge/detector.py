"""Streaming fall detector: push packets, receive one Decision per 0.5 s window. Stdlib only.

    det = Detector(Model.load('model.json'))
    for t, amps in packets:                 # t = monotonic seconds at receipt, amps = 52 amplitudes
        for d in det.push(t, amps):
            if d.alert: ...                 # one alert per episode -> backend event (schema v2)

Alert rule (from model.json): a window is positive if fall_score >= threshold; alert when `votes` of the
last `history` contiguous windows are positive (after `warmup` windows), reset after a gap longer than
`reset_after_gap_seconds`, at most one alert per `cooldown_seconds`. Windows that fail the gap-tolerant
quality gate produce a Decision with fall_score None and sensor_status 'degraded' (never a fake score).
"""
from collections import deque
from datetime import datetime, timezone

from . import window as W
from .csi import PacketError


class Decision:
    __slots__ = ('window_start', 'window_end', 'fall_score', 'positive', 'alert', 'sensor_status', 'quality',
                 'episode_score')

    def __init__(self, window_start, window_end, fall_score, positive, alert, sensor_status, quality, episode_score=None):
        self.window_start, self.window_end = window_start, window_end
        self.fall_score, self.positive, self.alert = fall_score, positive, alert
        self.sensor_status, self.quality = sensor_status, quality
        # On an alert: the strongest score among the voting windows (the evidence for the episode).
        self.episode_score = fall_score if episode_score is None else episode_score

    def confidence_scores(self):
        """Schema-v2 score pair {'fall': p, 'non_fall': 1 - p}, or None without a valid score."""
        if self.fall_score is None:
            return None
        p = min(max(float(self.episode_score if self.alert else self.fall_score), 0.0), 1.0)
        return {'fall': p, 'non_fall': 1.0 - p}

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != 'quality'}


class Detector:
    LOCK_PACKETS = 200            # ~2 s at ~100 Hz: majority vote over source MACs before inference starts

    def __init__(self, model, threshold=None, policy=None, stride=0.5, buffer_seconds=6.0, source_mac=None):
        # threshold: None = model.json value; a number = on-site calibrated override (config/threshold.json)
        # source_mac: pin the transmitter; None = majority of the first LOCK_PACKETS packets (never "first seen")
        self.model = model
        self.threshold = model.bundle['threshold'] if threshold is None else threshold
        self.policy = dict(model.bundle['alert_policy'], **(policy or {}))
        self.stride, self.buffer_seconds = stride, buffer_seconds
        self.times, self.amps = [], []
        self.next_start = None
        self.history = deque(maxlen=self.policy['history'])
        self.scores = deque(maxlen=self.policy['history'])
        self.previous_end = None
        self.last_alert = float('-inf')
        self.mac = source_mac.upper() if source_mac else None
        self.pending, self.mac_counts, self.foreign = [], {}, 0

    def push(self, t, amps, mac=None):
        """Add one packet; returns the Decisions for every window that became complete."""
        if mac is not None:
            mac = mac.upper()
            if self.mac is None:               # still choosing: buffer, then lock to the majority source
                self.pending.append((t, amps, mac)); self.mac_counts[mac] = self.mac_counts.get(mac, 0) + 1
                if len(self.pending) < self.LOCK_PACKETS:
                    return []
                self.mac = max(self.mac_counts, key=self.mac_counts.get)
                out = []
                for pt, pa, pm in self.pending:
                    out += self.push(pt, pa, pm)
                self.pending = []
                return out
            if mac != self.mac:
                self.foreign += 1
                return []                       # stray packet from another device would corrupt the stream
        if len(amps) != 52:
            raise PacketError('Expected 52 amplitudes')
        if self.times and t <= self.times[-1]:
            return []                           # duplicate / out-of-order timestamp dropped
        self.times.append(float(t)); self.amps.append(list(amps))
        if self.next_start is None:
            self.next_start = self.times[0]
        out = []
        while self.next_start + W.WINDOW_SECONDS < t:  # a packet exists after the window end
            start = self.next_start
            self.next_start += self.stride
            out.append(self._decide(start))
        self._trim()
        return out

    def _decide(self, start):
        end = start + W.WINDOW_SECONDS
        x, q = W.window_features(self.times, self.amps, start, gate='tolerant')
        if x is None:
            self.history.clear(); self.scores.clear()   # missing decisions break vote continuity
            return Decision(start, end, None, False, False, 'degraded', q)
        p = self.model.fall_score(x)
        positive = p >= self.threshold
        if self.previous_end is not None and end - self.previous_end > self.policy['reset_after_gap_seconds'] + 1e-9:
            self.history.clear(); self.scores.clear()
        self.history.append(positive); self.scores.append(p)
        self.previous_end = end
        alert = (len(self.history) >= self.policy['warmup_windows'] and sum(self.history) >= self.policy['votes']
                 and end - self.last_alert >= self.policy['cooldown_seconds'])
        if alert:
            self.last_alert = end
        return Decision(start, end, p, positive, alert, 'connected', q, episode_score=max(self.scores) if alert else p)

    def _trim(self):
        keep_from = self.next_start - 1.0         # keep a bracketing packet before the next window
        i = 0
        while i + 1 < len(self.times) and self.times[i + 1] < keep_from:
            i += 1
        if i:
            del self.times[:i]; del self.amps[:i]

    def idle_status(self, now, timeout=2.0):
        """'disconnected' if no packet for `timeout` seconds (call from the I/O loop)."""
        return 'disconnected' if not self.times or now - self.times[-1] > timeout else 'connected'


def iso_now():
    return datetime.now(timezone.utc).isoformat()

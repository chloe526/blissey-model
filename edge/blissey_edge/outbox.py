"""Durable event outbox (one JSON file per event). Standard library only, thread-safe.

An event is written and fsynced BEFORE any send attempt, and removed only after the backend's
event_ack. Retries resend the byte-identical payload (the backend deduplicates on event_id + payload).
Non-retryable rejections move the file to dead/ and increment dropped_event_count.
"""
import json
import os
import threading


def _atomic_write(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    try:                                   # persist the rename itself
        fd = os.open(os.path.dirname(path) or '.', os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class Outbox:
    def __init__(self, state_dir):
        self.dir = os.path.join(state_dir, 'outbox')
        self.dead_dir = os.path.join(state_dir, 'dead')
        self.counter_path = os.path.join(state_dir, 'dropped_event_count')
        os.makedirs(self.dir, exist_ok=True); os.makedirs(self.dead_dir, exist_ok=True)
        self.lock = threading.Lock()
        for name in os.listdir(self.dir):  # half-written temp files from a crash are never sent
            if name.endswith('.tmp'):
                os.remove(os.path.join(self.dir, name))

    def _path(self, event_id):
        if not event_id or '/' in event_id or event_id.startswith('.'):
            raise ValueError('Unsafe event_id')
        return os.path.join(self.dir, event_id + '.json')

    def add(self, event):
        with self.lock:
            path = self._path(event['event_id'])
            if os.path.exists(path):
                raise ValueError('Duplicate event_id in outbox: %s' % event['event_id'])
            _atomic_write(path, json.dumps(event, separators=(',', ':'), sort_keys=True))

    def pending(self):
        """[(event_id, payload_text)] oldest first; payload text is sent verbatim on every retry."""
        with self.lock:
            names = sorted(os.listdir(self.dir), key=lambda n: os.path.getmtime(os.path.join(self.dir, n)))
            out = []
            for n in names:
                if n.endswith('.json'):
                    with open(os.path.join(self.dir, n)) as f:
                        out.append((n[:-5], f.read()))
            return out

    def ack(self, event_id):
        with self.lock:
            try:
                os.remove(self._path(event_id))
            except FileNotFoundError:
                pass

    def dead(self, event_id, reason):
        with self.lock:
            src = self._path(event_id)
            if os.path.exists(src):
                os.replace(src, os.path.join(self.dead_dir, event_id + '.json'))
                with open(os.path.join(self.dead_dir, event_id + '.reason'), 'w') as f:
                    f.write(reason)
                _atomic_write(self.counter_path, str(self._dropped() + 1))

    def _dropped(self):
        try:
            with open(self.counter_path) as f:
                return int(f.read().strip() or 0)
        except FileNotFoundError:
            return 0

    def depth(self):
        with self.lock:
            return sum(1 for n in os.listdir(self.dir) if n.endswith('.json'))

    def dropped(self):
        with self.lock:
            return self._dropped()

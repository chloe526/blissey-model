"""Background WebSocket uplink to the Blissey backend. Standard library only.

- Connects with `Authorization: Bearer <token>` (token never logged or put in a URL/payload).
- On connect: sends health, then every pending outbox event (byte-identical payloads on retry).
- Health every `health_interval` seconds (backend marks the device stale after 30 s).
- event_ack (accepted/duplicate) -> event removed from the outbox.
- error retryable=true  -> kept, resent after backoff; retryable=false -> moved to dead/, counted.
- Credential rejected (HTTP 401/403, error unauthorized_device, close 1008) -> stops sending and
  retries only every `auth_retry_seconds`; detection and the outbox keep working meanwhile.
- Any other failure -> reconnect with jittered exponential backoff (1 s .. 60 s).
- clock_synced = |local UTC - server Date header| < 2 s, measured at each connect.
"""
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import json
import os
import threading
import time

from .contracts import health_errors
from .episodes import iso_utc
from .ws import AuthError, WebSocket, WSError

ACK_TIMEOUT = 30.0


class Uplink(threading.Thread):
    def __init__(self, url, token, device_id, room_id, outbox, health_state, ca_file=None,
                 health_interval=10.0, auth_retry_seconds=600.0, log=print):
        super().__init__(name='uplink', daemon=True)
        self.url, self._token, self.device_id, self.room_id = url, token, device_id, room_id
        self.outbox, self.health_state, self.ca_file = outbox, health_state, ca_file
        self.health_interval, self.auth_retry = health_interval, auth_retry_seconds
        self.log = log
        self._halt = threading.Event(); self._wake = threading.Event()
        self.clock_synced = False
        self.connected = False
        self.stats = dict(connects=0, acks=0, duplicates=0, retryable_errors=0, rejected=0, auth_failures=0)

    # ---- control
    def wake(self):
        self._wake.set()

    def stop(self):
        self._halt.set(); self._wake.set()

    # ---- messages
    def health_message(self):
        s = self.health_state()
        msg = {'type': 'device_health', 'device_id': self.device_id, 'room_id': self.room_id, 'sent_at': iso_utc(),
               'sensor_status': s['sensor_status'], 'model_status': s['model_status'],
               'last_valid_frame_at': s['last_valid_frame_at'], 'outbox_depth': self.outbox.depth(),
               'dropped_event_count': self.outbox.dropped(), 'clock_synced': self.clock_synced}
        errs = health_errors(msg)
        if errs:
            raise ValueError('Invalid health message: %s' % errs)
        return json.dumps(msg, separators=(',', ':'))

    def _check_clock(self, date_header):
        try:
            offset = (datetime.now(timezone.utc) - parsedate_to_datetime(date_header)).total_seconds()
            self.clock_synced = abs(offset) < 2.0
            if not self.clock_synced:
                self.log('WARNING clock offset vs server %+.1f s; fix NTP (detected_at would be wrong)' % offset)
        except Exception:
            self.clock_synced = False

    # ---- main loop
    def run(self):
        backoff = 1.0
        while not self._halt.is_set():
            try:
                self._session()
                backoff = 1.0
            except AuthError as exc:
                self.stats['auth_failures'] += 1
                self.log('CREDENTIAL REJECTED (%s): not sending; next attempt in %d s. '
                         'Check the token file and that its SHA-256 is registered for device %s.'
                         % (exc, self.auth_retry, self.device_id))
                self._halt.wait(self.auth_retry)
            except (WSError, OSError, ValueError) as exc:
                delay = min(60.0, backoff) * (0.5 + int.from_bytes(os.urandom(1), 'big') / 255)
                self.log('uplink: %s; reconnecting in %.1f s' % (exc, delay))
                backoff = min(60.0, backoff * 2)
                self._halt.wait(delay)
            finally:
                self.connected = False

    def _session(self):
        ws = WebSocket(self.url, headers={'Authorization': 'Bearer ' + self._token}, ca_file=self.ca_file).connect()
        self.connected = True; self.stats['connects'] += 1
        self._check_clock(ws.server_date)
        self.log('uplink: connected to %s (clock_synced=%s, outbox=%d)' % (self.url, self.clock_synced, self.outbox.depth()))
        in_flight = {}                    # event_id -> sent_at (monotonic)
        retry_at = {}                     # event_id -> monotonic time to resend after a retryable error
        next_health = 0.0
        try:
            while not self._halt.is_set():
                now = time.monotonic()
                if now >= next_health:
                    ws.send(self.health_message()); next_health = now + self.health_interval
                for event_id, payload in self.outbox.pending():
                    if event_id in in_flight or retry_at.get(event_id, 0) > now:
                        continue
                    ws.send(payload); in_flight[event_id] = now
                    self.log('uplink: sent %s' % event_id)
                if any(now - t > ACK_TIMEOUT for t in in_flight.values()):
                    raise WSError('No acknowledgment within %d s; reconnecting to resend' % ACK_TIMEOUT)
                msg = ws.recv(timeout=0.5)
                if msg is not None:
                    self._handle(msg, in_flight, retry_at)
                if self._wake.is_set():
                    self._wake.clear()
        finally:
            ws.close()

    def _handle(self, text, in_flight, retry_at):
        try:
            msg = json.loads(text)
        except ValueError:
            raise WSError('Malformed server message')
        kind, event_id = msg.get('type'), msg.get('event_id')
        if kind == 'event_ack' and msg.get('status') in ('accepted', 'duplicate'):
            self.outbox.ack(event_id); in_flight.pop(event_id, None); retry_at.pop(event_id, None)
            self.stats['acks' if msg['status'] == 'accepted' else 'duplicates'] += 1
            self.log('uplink: %s %s -> incident %s' % (event_id, msg['status'], msg.get('incident_id')))
        elif kind == 'error':
            code = msg.get('code')
            in_flight.pop(event_id, None)
            if code == 'unauthorized_device':
                raise AuthError('server error unauthorized_device')
            if msg.get('retryable'):
                self.stats['retryable_errors'] += 1
                retry_at[event_id] = time.monotonic() + 5.0
                self.log('uplink: %s retryable error %s; resending in 5 s' % (event_id, code))
            elif event_id != 'unknown':
                self.stats['rejected'] += 1
                self.outbox.dead(event_id, str(code))
                self.log('uplink: %s REJECTED (%s); moved to dead/ (not retried)' % (event_id, code))
            else:
                self.log('uplink: server rejected an uncorrelated message (%s)' % code)
        else:
            self.log('uplink: ignoring unexpected message type %r' % kind)

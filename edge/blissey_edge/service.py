"""Blissey Pi service: ESP32 serial CSI -> fall detector -> episodes -> durable outbox -> backend WSS.

Run from the bundle folder (model.json next to blissey_edge/). Standard library only.

    # once: create the device credential ON THE PI; give only the printed SHA-256 to the backend admin
    python3 -m blissey_edge.service --make-token

    # live (device "1" in room "101" are the registered string IDs)
    python3 -m blissey_edge.service --serial /dev/serusb1

    # no network: print the events that would be sent
    python3 -m blissey_edge.service --serial /dev/serusb1 --no-network

    # test without hardware: replay a recorded measurements.jsonl
    python3 -m blissey_edge.service --replay path/to/measurements.jsonl --no-network

    # PRODUCER for the verified QNX transport (blissey-beacon edge/qnx, branch codex/qnx-websocket-client):
    # enqueue grouped events into its SQLite outbox + write its health snapshot; it does the sending.
    python3 -m blissey_edge.service --serial /dev/serusb1 \
        --qnx-config /data/home/qnxuser/blissey-beacon/config/device.json

Each fall episode produces ONE schema-v2 event (7 fields); alerts in the following --refractory
seconds (default 60) are suppressed. Health goes out every 10 s. Ctrl-C / SIGTERM stops cleanly;
unsent events stay in <state-dir>/outbox and are resent on the next start.
"""
import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import sys
import threading
import time

from .contracts import event_errors
from .csi import PacketError, parse_record, parse_serial_line
from .detector import Detector
from .episodes import EpisodeGrouper, iso_utc
from .model import Model
from .outbox import Outbox
from .selftest import check_sums

DEFAULT_URL = 'wss://api.blissey-beacon.health/api/devices/v1/stream'
HOME = os.path.expanduser('~/.blissey')
EXIT_SERIAL_GARBLED = 4       # supervise.py restarts the USB-serial driver, then this service
GARBLED_SECONDS = 8.0         # bytes arriving but no valid CSI line for this long = garbled stream
GARBLED_MIN_BYTES = 4096
TOKEN_RE = re.compile(r'^[\x21-\x7e]{32,256}$')         # same rule as the backend's Bearer parser


def log(msg):
    print('%s %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def make_token(path):
    if os.path.exists(path):
        raise SystemExit('%s already exists; refusing to overwrite an existing credential' % path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    token = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(token)
    print('Token written to %s (mode 0600). It is never printed.' % path)
    print('Give ONLY this SHA-256 to the backend admin to register for device "1":')
    print(hashlib.sha256(token.encode('utf-8')).hexdigest())


def read_token(path):
    with open(path) as f:
        token = f.read().strip()
    if not TOKEN_RE.match(token):
        raise SystemExit('Token in %s is not 32-256 printable ASCII characters' % path)
    return token


def atomic_write_json(path, value):
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(value, f, separators=(',', ':')); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class QnxProducer:
    """Hands grouped events and health to the QNX transport (blissey_qnx) instead of sending ourselves.

    Reads the transport's device.json (device_id, room_id, outbox_path, health_file; paths relative to it),
    enqueues into its SQLite Outbox (committed before enqueue returns) and atomically replaces its
    5-field health snapshot (updated_at, sensor_status, model_status, last_valid_frame_at, clock_synced)."""
    def __init__(self, config_path, transport_dir=None, clock_synced=False):
        config_path = os.path.abspath(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        base = os.path.dirname(config_path)
        transport_dir = os.path.abspath(transport_dir or os.path.join(base, '..', 'transport'))
        if not os.path.isdir(os.path.join(transport_dir, 'blissey_qnx')):
            raise SystemExit('blissey_qnx not found in %s; pass --qnx-transport DIR' % transport_dir)
        sys.path.insert(0, transport_dir)
        from blissey_qnx.outbox import Outbox as QnxOutbox
        self.device_id, self.room_id = cfg['device_id'], cfg['room_id']
        self.health_file = os.path.normpath(os.path.join(base, cfg['health_file']))
        self.outbox = QnxOutbox(os.path.normpath(os.path.join(base, cfg['outbox_path'])), self.device_id, self.room_id)
        self.clock_synced = bool(clock_synced)
        self.last_health = 0.0

    def enqueue(self, event):
        return self.outbox.enqueue(event)        # raises on validation/persistence failure

    def write_health(self, snapshot, force=False):
        if not force and time.monotonic() - self.last_health < 5.0:
            return
        atomic_write_json(self.health_file, {'updated_at': iso_utc(), 'sensor_status': snapshot['sensor_status'],
                                             'model_status': snapshot['model_status'],
                                             'last_valid_frame_at': snapshot['last_valid_frame_at'],
                                             'clock_synced': self.clock_synced})
        self.last_health = time.monotonic()

    def close(self):
        self.outbox.close()


class State:
    """Shared sensor/model state for health messages (updated by the detection loop)."""
    def __init__(self):
        self.lock = threading.Lock()
        self.model_status = 'loading'
        self.last_packet = None           # monotonic
        self.last_valid_frame_at = None   # ISO wall time of the last window with a valid score
        self.last_window_valid = False

    def snapshot(self):
        with self.lock:
            if self.last_packet is None or time.monotonic() - self.last_packet > 2.0:
                sensor = 'disconnected'
            else:
                sensor = 'connected' if self.last_window_valid else 'degraded'
            return dict(sensor_status=sensor, model_status=self.model_status, last_valid_frame_at=self.last_valid_frame_at)


def packets_from_serial(device, baud, stop):
    from .serial_source import SerialSource
    while not stop.is_set():
        src = SerialSource(device, baud)
        try:
            src.open(); log('serial: opened %s (%s)' % (device, src.baud_note))
            last_ok, bytes_at_ok = time.monotonic(), 0
            while not stop.is_set():
                # QNX devc-serusb can deliver scrambled bytes after a reopen even at the right baud; only a
                # driver restart fixes it, so report it instead of silently reading garbage forever.
                if (time.monotonic() - last_ok > GARBLED_SECONDS
                        and src.bytes_read - bytes_at_ok > GARBLED_MIN_BYTES):
                    log('serial: GARBLED, %d bytes in %.0f s but no valid CSI line; exiting %d so the supervisor '
                        'restarts the USB-serial driver' % (src.bytes_read - bytes_at_ok, time.monotonic() - last_ok,
                                                           EXIT_SERIAL_GARBLED))
                    yield 'garbled'
                    return
                line = src.readline(timeout=1.0)
                if line is None or not line.startswith('CSI_DATA'):
                    yield None if line is None else 'skip'
                    continue
                try:
                    mac, role, amps = parse_serial_line(line)
                except PacketError:
                    continue
                last_ok, bytes_at_ok = time.monotonic(), src.bytes_read
                yield time.monotonic(), mac, role, amps
        except OSError as exc:
            log('serial: %s; retrying in 2 s' % exc)
            yield 'reset'
            stop.wait(2.0)
        finally:
            src.close()


def packets_from_replay(path, speed, stop):
    start_wall = time.monotonic(); first = None
    with open(path) as f:
        for text in f:
            if stop.is_set():
                return
            try:
                t, mac, role, amps = parse_record(text)
            except (PacketError, KeyError, ValueError):
                continue
            first = t if first is None else first
            if speed > 0:
                delay = (t - first) / speed - (time.monotonic() - start_wall)
                if delay > 0:
                    time.sleep(delay)
            yield t, mac, role, amps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument('--serial', help='ESP32 AP receiver serial device, e.g. /dev/serusb1')
    src.add_argument('--replay', help='recorded measurements.jsonl (testing without hardware)')
    src.add_argument('--make-token', action='store_true', help='create the device token on this Pi and print its SHA-256')
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--replay-speed', type=float, default=1.0, help='1 = real time, 0 = as fast as possible')
    p.add_argument('--model', default='model.json')
    p.add_argument('--backend', choices=('tflite', 'python'), default=None,
                   help='default: tflite if model.json declares model.tflite, else the stdlib scorer')
    p.add_argument('--url', default=DEFAULT_URL)
    p.add_argument('--token-file', default=os.path.join(HOME, 'device_token'))
    p.add_argument('--device-id', default='1')
    p.add_argument('--room-id', default='101')
    p.add_argument('--state-dir', default=os.path.join(HOME, 'state'))
    p.add_argument('--refractory', type=float, default=60.0, help='seconds of suppression after each event')
    p.add_argument('--ca-file', default=None, help='CA bundle if the system default is missing')
    p.add_argument('--no-network', action='store_true', help='log events instead of sending (no token needed)')
    p.add_argument('--qnx-config', default=None, help='hand events/health to the QNX transport using its device.json')
    p.add_argument('--qnx-transport', default=None, help='blissey_qnx transport dir (default: <config dir>/../transport)')
    p.add_argument('--threshold-file', default=None,
                   help='on-site calibrated threshold (calibrate.py output); ignored if missing or for another model')
    p.add_argument('--source-mac', default=None, help='pin the ESP32 transmitter MAC (default: majority of first packets)')
    p.add_argument('--clock-synced', action='store_true',
                   help='report clock_synced=true (only if the Pi clock service is verified synchronized)')
    a = p.parse_args(argv)
    if a.make_token:
        return make_token(a.token_file)
    if not (a.serial or a.replay):
        p.error('choose --serial DEVICE or --replay FILE (or --make-token)')

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    state = State()
    outbox = Outbox(a.state_dir)
    producer = None
    if a.qnx_config:
        if a.no_network:
            p.error('--qnx-config and --no-network are exclusive')
        producer = QnxProducer(a.qnx_config, a.qnx_transport, a.clock_synced)
        a.device_id, a.room_id = producer.device_id, producer.room_id      # identity comes from the transport config
        log('producer: events -> %s, health -> %s (sending is done by the QNX transport process)'
            % (producer.outbox.path, producer.health_file))
    grouper = EpisodeGrouper(a.device_id, a.room_id, a.refractory)
    uplink = None
    if not a.no_network and producer is None:
        from .uplink import Uplink
        uplink = Uplink(a.url, read_token(a.token_file), a.device_id, a.room_id, outbox, state.snapshot,
                        ca_file=a.ca_file, log=log)
        uplink.start()
    try:
        log('checksums: %s' % check_sums(os.path.dirname(os.path.abspath(a.model))))
        model = Model.load(a.model, a.backend)
        with state.lock:
            state.model_status = 'ready'
        log('model: %s via %s (%s), threshold %.4f, trained on %s; device %s room %s; refractory %.0f s; outbox %d pending'
            % (model.bundle['model_kind'], model.backend, model.bundle.get('created', '?'), model.bundle['threshold'],
               model.bundle.get('trained_on'), a.device_id, a.room_id, a.refractory, outbox.depth()))
    except (OSError, ValueError, KeyError, ImportError, RuntimeError, SystemExit) as exc:
        with state.lock:
            state.model_status = 'error'
        log('MODEL ERROR: %s. Sending health only (model_status=error); no inference events.' % exc)
        while not stop.is_set():
            if producer:
                producer.write_health(state.snapshot())
            stop.wait(1.0)
        return 1

    threshold = None
    if a.threshold_file and os.path.exists(a.threshold_file):
        try:
            with open(a.threshold_file) as f:
                cal = json.load(f)
            with open(a.model, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            if cal.get('model_sha256') != sha:
                log('calibration %s is for a different model.json; ignoring it (recalibrate)' % a.threshold_file)
            else:
                threshold = float(cal['threshold'])
                log('using on-site calibrated threshold %.4f (model %.4f; calibrated %s over %s windows)'
                    % (threshold, model.bundle['threshold'], cal.get('calibrated_at'), cal.get('windows')))
        except (OSError, ValueError, KeyError) as exc:
            log('could not read %s (%s); using model threshold' % (a.threshold_file, exc))
    make_detector = lambda: Detector(model, threshold=threshold, source_mac=a.source_mac)
    det = make_detector(); packets = windows = valid = 0; last_report = time.monotonic(); last_score = None
    exit_code = 0
    source = packets_from_serial(a.serial, a.baud, stop) if a.serial else packets_from_replay(a.replay, a.replay_speed, stop)
    for item in source:
        if producer:
            producer.write_health(state.snapshot())
        if stop.is_set():
            break
        if item == 'garbled':
            exit_code = EXIT_SERIAL_GARBLED
            break
        if item == 'reset':
            det = make_detector()          # clean stream start: require a full valid window again
            continue
        if item is None or item == 'skip':
            continue
        t, mac, role, amps = item
        if role != 'AP':
            continue                        # model trained on AP-received CSI only
        packets += 1
        with state.lock:
            state.last_packet = time.monotonic()
        for d in det.push(t, amps, mac):
            windows += 1
            with state.lock:
                state.last_window_valid = d.fall_score is not None
                if d.fall_score is not None:
                    state.last_valid_frame_at = iso_utc()
            if d.fall_score is not None:
                valid += 1; last_score = d.fall_score
            if d.alert:
                log('ALERT window %.1f-%.1f s score %.3f' % (d.window_start, d.window_end, d.fall_score))
            event = grouper.offer(d, d.window_end)
            if event is None:
                continue
            errs = event_errors(event)
            if errs:
                log('BUG: event failed contract check %s; not sent' % errs); continue
            if producer:
                try:
                    producer.enqueue(event)    # SQLite commit completes before this returns
                    how = 'enqueued for QNX transport'
                except Exception as exc:       # never claim delivery on a failed commit
                    how = 'ENQUEUE FAILED (%s: %s)' % (type(exc).__name__, exc)
            elif uplink:
                outbox.add(event)              # durable before any send attempt
                uplink.wake()
                how = 'queued'
            else:
                how = 'no-network, not queued: ' + json.dumps(event)
            log('EVENT %s fall=%.3f (%s)' % (event['event_id'], event['confidence_scores']['fall'], how))
        if time.monotonic() - last_report > 10:
            log('status: %d packets, %d windows (%d valid), last score %s, suppressed alerts %d, outbox %d%s'
                % (packets, windows, valid, 'n/a' if last_score is None else '%.3f' % last_score, grouper.suppressed,
                   outbox.depth(), '' if not uplink else ', uplink %s' % ('connected' if uplink.connected else 'down')))
            last_report = time.monotonic()
    log('stopping: %d packets, %d windows, %d events this run, %d alerts suppressed, %d unsent in outbox'
        % (packets, windows, grouper.seq, grouper.suppressed, outbox.depth()))
    if uplink:
        uplink.stop(); uplink.join(timeout=5)
    if producer:
        with state.lock:
            state.last_packet = None              # report disconnected on clean shutdown
        producer.write_health(state.snapshot(), force=True); producer.close()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

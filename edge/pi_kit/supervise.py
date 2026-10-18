"""Start and supervise the Blissey Pi processes (run as qnxuser). Standard library only.

    python3 supervise.py --serial /dev/serusb1        # from <root>; or: sh start.sh /dev/serusb1

1. Serial: if the device is missing, starts the CP210x USB-serial driver (drivers/devc-serusb via sudo)
   and waits for it. Emilia's report: VID 10c4 / PID ea60 -> /dev/serusb1, 921600 baud 8N1.
2. Transport (blissey_qnx, sender) and detector (blissey_edge, producer) run as child processes with
   logs in <root>/logs/{transport,detector}.log (rotated at 5 MB, 3 kept).
3. A crashed child restarts with backoff (2 s .. 60 s). Transport exit 2/3 and detector exit 2 mean
   "operator repair needed" (credential/config/storage): NOT restarted; the reason is logged.
   Detector exit 4 = garbled serial stream (QNX devc-serusb after a reopen): the driver is restarted first.
4. SIGINT/SIGTERM stop both children cleanly (queued events stay in the outbox).
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPAIR = {'transport': (2, 3), 'detector': (2,)}
SERIAL_GARBLED = 4            # blissey_edge.service EXIT_SERIAL_GARBLED
MAX_LOG, KEEP = 5 * 1024 * 1024, 3


def log(msg):
    print('%s supervise: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


def ensure_serial(device, wait=15.0):
    if os.path.exists(device):
        return True
    driver = os.path.join(ROOT, 'drivers', 'devc-serusb')
    if not os.path.exists(driver):
        log('%s missing and %s not found; start the USB-serial driver manually' % (device, driver)); return False
    log('%s missing: starting USB-serial driver (sudo %s)' % (device, driver))
    r = subprocess.run(['sudo', driver])
    if r.returncode != 0:
        log('driver start exited %d; run `use %s` to see its options' % (r.returncode, driver))
    end = time.time() + wait
    while time.time() < end:
        if os.path.exists(device):
            log('%s ready' % device); return True
        time.sleep(0.5)
    log('%s did not appear; present: %s. Check the AP board cable/power.'
        % (device, ', '.join(sorted(glob.glob('/dev/serusb*'))) or 'none'))
    return False


def restart_driver(device):
    """slay devc-serusb and start it again (the only fix seen for a scrambled-but-open port)."""
    log('serial garbled: restarting USB-serial driver (sudo slay devc-serusb)')
    subprocess.run(['sudo', '-n', 'slay', 'devc-serusb'], stdin=subprocess.DEVNULL)
    end = time.time() + 5.0
    while os.path.exists(device) and time.time() < end:
        time.sleep(0.2)
    return ensure_serial(device)


class Child:
    def __init__(self, name, cmd, cwd, on_exit=None):
        self.name, self.cmd, self.cwd, self.on_exit = name, cmd, cwd, on_exit or {}
        self.proc, self.log_path = None, os.path.join(ROOT, 'logs', name + '.log')
        self.backoff, self.next_start, self.gave_up, self.started_at = 2.0, 0.0, False, 0.0

    def _rotate(self):
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > MAX_LOG:
            for i in range(KEEP - 1, 0, -1):
                if os.path.exists('%s.%d' % (self.log_path, i)):
                    os.replace('%s.%d' % (self.log_path, i), '%s.%d' % (self.log_path, i + 1))
            os.replace(self.log_path, self.log_path + '.1')

    def start(self):
        self._rotate()
        out = open(self.log_path, 'a')
        out.write('\n==== %s start: %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(self.cmd))); out.flush()
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdout=out, stderr=subprocess.STDOUT)
        out.close()
        self.started_at = time.time()
        log('%s started (pid %d), log %s' % (self.name, self.proc.pid, self.log_path))

    def poll(self, now):
        if self.gave_up:
            return
        if self.proc is None:
            if now >= self.next_start:
                self.start()
            return
        code = self.proc.poll()
        if code is None:
            if now - self.started_at > 300:
                self.backoff = 2.0          # healthy for 5 min: reset backoff
            return
        self.proc = None
        if code in REPAIR[self.name]:
            self.gave_up = True
            log('%s exited %d = operator repair needed (see %s); NOT restarting' % (self.name, code, self.log_path))
            return
        if code in self.on_exit:
            self.on_exit[code]()
        self.next_start = now + self.backoff
        log('%s exited %d; restarting in %.0f s' % (self.name, code, self.backoff))
        self.backoff = min(60.0, self.backoff * 2)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--serial', default='/dev/serusb1', help='AP receiver serial device')
    p.add_argument('--replay', default=None, help='testing: replay a measurements.jsonl instead of serial')
    p.add_argument('--no-transport', action='store_true', help='testing: run the detector only')
    a = p.parse_args(argv)
    config = os.path.join(ROOT, 'config', 'device.json')
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    if not a.replay and not ensure_serial(a.serial):
        return 1
    source = ['--replay', a.replay, '--replay-speed', '1'] if a.replay else ['--serial', a.serial]
    calibration = ['--threshold-file', os.path.join(ROOT, 'config', 'threshold.json')]
    children = [Child('detector', [sys.executable, '-B', '-m', 'blissey_edge.service'] + source + ['--qnx-config', config]
                      + calibration,
                      os.path.join(ROOT, 'detector'),
                      on_exit={} if a.replay else {SERIAL_GARBLED: lambda: restart_driver(a.serial)})]
    if not a.no_transport:
        children.insert(0, Child('transport', ['sh', os.path.join(ROOT, 'transport', 'run-client.sh'), config],
                                 os.path.join(ROOT, 'transport')))
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.append(1))
    log('running %s; Ctrl-C to stop' % ', '.join(c.name for c in children))
    while not stopping:
        now = time.time()
        for c in children:
            c.poll(now)
        if all(c.gave_up for c in children):
            log('all processes need operator repair; exiting'); return 2
        time.sleep(0.5)
    log('stopping')
    for c in children:
        c.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())

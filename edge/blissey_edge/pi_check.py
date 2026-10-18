"""Phase 1 platform check for the Pi / QNX. Standard library only. Run from the bundle folder:

    python3 -m blissey_edge.pi_check                              # Python, TLS, clock, serial devices, model
    python3 -m blissey_edge.pi_check --serial /dev/serusb1        # + read live CSI for 10 s
    python3 -m blissey_edge.pi_check --serial /dev/serusb1 --baud 921600 --seconds 20 --host api.blissey-beacon.health

Prints PASS / WARN / FAIL per check and exits 1 if anything FAILs. Never needs or prints a token.
"""
import argparse
import glob
import os
import socket
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

RESULTS = []


def report(status, name, detail):
    RESULTS.append(status)
    print('%-5s %-22s %s' % (status, name, detail), flush=True)


def check_python():
    ok = sys.version_info >= (3, 8)
    report('PASS' if ok else 'FAIL', 'python', '%s on %s' % (sys.version.split()[0], sys.platform))


def check_ssl():
    try:
        import ssl
    except ImportError as exc:
        report('FAIL', 'ssl module', 'missing (%s): WSS impossible; Python needs an OpenSSL-enabled build' % exc)
        return None
    paths = ssl.get_default_verify_paths()
    cafile = paths.cafile and os.path.exists(paths.cafile)
    capath = paths.capath and os.path.isdir(paths.capath) and os.listdir(paths.capath)
    report('PASS', 'ssl module', ssl.OPENSSL_VERSION)
    report('PASS' if (cafile or capath) else 'WARN', 'CA certificates',
           'cafile=%s capath=%s' % (paths.cafile if cafile else None, paths.capath if capath else None) +
           ('' if (cafile or capath) else ' -> none found; pass --ca-file to the service'))
    return ssl


def check_tls_and_clock(ssl, host, ca_file):
    if ssl is None:
        return
    try:
        ctx = ssl.create_default_context(cafile=ca_file)
        with socket.create_connection((host, 443), timeout=10) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as s:
                s.sendall(('HEAD /healthz HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n' % host).encode())
                head = s.recv(4096).decode('latin-1')
                report('PASS', 'TLS to backend', '%s verified (%s)' % (host, s.version()))
    except Exception as exc:
        report('FAIL', 'TLS to backend', '%s: %s' % (host, exc))
        return
    date = next((l.split(':', 1)[1].strip() for l in head.split('\r\n') if l.lower().startswith('date:')), None)
    if not date:
        report('WARN', 'clock', 'server sent no Date header; check NTP manually')
        return
    offset = (datetime.now(timezone.utc) - parsedate_to_datetime(date)).total_seconds()
    report('PASS' if abs(offset) < 2 else 'FAIL', 'clock',
           'local UTC %s, offset vs server %+.1f s%s' % (datetime.now(timezone.utc).isoformat(timespec='seconds'), offset,
                                                        '' if abs(offset) < 2 else ' -> fix NTP before sending events'))


def candidates():
    found = []
    for pattern in ('/dev/serusb*', '/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ser[0-9]*', '/dev/cu.usbserial*'):
        found += sorted(glob.glob(pattern))
    return found


def check_serial(device, baud, seconds):
    from .serial_source import SerialSource
    from .csi import PacketError, parse_serial_line
    devs = candidates()
    report('PASS' if devs else 'WARN', 'serial devices', ', '.join(devs) or 'none found (is the ESP32 plugged in / driver loaded?)')
    if not device:
        return
    try:
        src = SerialSource(device, baud)
        src.open()
    except Exception as exc:
        report('FAIL', 'serial open', '%s: %s' % (device, exc))
        return
    report('PASS', 'serial open', '%s (%s)' % (device, src.baud_note))
    lines = csi = bad = 0; macs, roles = {}, {}
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        line = src.readline(timeout=1.0)
        if line is None:
            continue
        lines += 1
        if not line.startswith('CSI_DATA'):
            continue
        try:
            mac, role, _ = parse_serial_line(line)
            csi += 1; macs[mac] = macs.get(mac, 0) + 1; roles[role] = roles.get(role, 0) + 1
        except PacketError:
            bad += 1
    src.close()
    rate = csi / seconds
    top = max(macs.values()) if macs else 0
    status = 'PASS' if rate >= 80 and roles.keys() == {'AP'} and top >= 0.95 * csi else ('WARN' if csi else 'FAIL')
    report(status, 'live CSI', '%d lines, %d CSI packets (%.1f Hz), %d unparseable; roles %s; source MACs %s'
           % (lines, csi, rate, bad, roles, macs))
    if csi and roles.keys() != {'AP'}:
        print('      expected only AP-received CSI (role AP); the model was trained on AP-side data')


def check_model():
    if not (os.path.exists('model.json') and os.path.exists('golden.json')):
        report('WARN', 'model selftest', 'run from the bundle folder containing model.json and golden.json')
        return
    from .selftest import main as selftest
    print('----- model selftest -----')
    try:
        code = selftest(['model.json', 'golden.json'])
    except (ImportError, OSError, ValueError, RuntimeError) as exc:     # e.g. TFLite runtime not installed
        print('--------------------------')
        report('FAIL', 'model selftest', str(exc))
        return
    print('--------------------------')
    report('PASS' if code == 0 else 'FAIL', 'model selftest', 'golden replay identical' if code == 0 else 'mismatch')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--serial', default=None, help='serial device of the ESP32 AP receiver')
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--host', default='api.blissey-beacon.health')
    p.add_argument('--ca-file', default=None)
    a = p.parse_args(argv)
    check_python()
    ssl = check_ssl()
    check_tls_and_clock(ssl, a.host, a.ca_file)
    check_serial(a.serial, a.baud, a.seconds)
    check_model()
    verdict = 'FAIL' if 'FAIL' in RESULTS else 'WARN' if 'WARN' in RESULTS else 'PASS'
    print('\nPHASE 1 RESULT: %s' % verdict)
    return 1 if verdict == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())

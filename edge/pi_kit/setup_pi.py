"""Install / repair the Blissey Pi setup so it matches the current architecture. Run ON THE PI from the kit:

    python3 setup_pi.py --check        # inspect only, change nothing
    python3 setup_pi.py                # install/repair (backs up anything it replaces)

Target layout (Emilia's verified QNX layout + the detector):
    <root>/transport   blissey_qnx WSS sender (blissey-beacon codex/qnx-websocket-client) + vendor/websockets
    <root>/config      device.json (device "1", room "101"), device-token (EXISTING registered token), health.json
    <root>/outbox      SQLite outbox shared by detector (producer) and transport (sender)
    <root>/logs
    <root>/detector    model.json, golden.json, blissey_edge (serial -> features -> model -> episodes)
    <root>/run-transport.sh, <root>/run-detector.sh
Default root: /data/home/qnxuser/blissey-beacon. Standard library only (Python 3.8+; the Pi has 3.14).
The token is never read into this script's output, generated or rewritten; only its file mode and its
SHA-256 (which the backend stores and may be compared with the registered digest) are reported.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile

KIT = os.path.dirname(os.path.abspath(__file__))
DEVICE_ID, ROOM_ID = '1', '101'
URL = 'wss://api.blissey-beacon.health/api/devices/v1/stream'
CONFIG = dict(url=URL, device_id=DEVICE_ID, room_id=ROOM_ID, token_file='device-token',
              outbox_path='../outbox/events.sqlite3', health_file='health.json')
PROBLEMS, ACTIONS = [], []


def say(status, what, detail=''):
    print('%-5s %-24s %s' % (status, what, detail), flush=True)
    if status in ('FAIL', 'WARN'):
        PROBLEMS.append('%s: %s %s' % (status, what, detail))


def tree_hash(path):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(path, '**', '*'), recursive=True)):
        if os.path.isfile(f) and '__pycache__' not in f and '/vendor/' not in f.replace(os.sep, '/'):
            h.update(os.path.relpath(f, path).encode()); h.update(open(f, 'rb').read())
    return h.hexdigest()


def backup(path, fix):
    dst = '%s.bak-%s' % (path, time.strftime('%Y%m%d-%H%M%S'))
    if fix:
        shutil.move(path, dst)
    ACTIONS.append('%s %s -> %s' % ('backed up' if fix else 'would back up', path, dst))
    return dst


def install_tree(src, dst, name, fix, keep=()):
    """Install kit folder `src` at `dst`; if a different version exists, back it up (carrying over `keep`)."""
    if os.path.isdir(dst) and tree_hash(dst) == tree_hash(src):
        say('PASS', name, 'up to date (%s)' % dst); return
    if os.path.isdir(dst):
        say('FIX' if fix else 'WARN', name, 'differs from kit version -> will replace (backup kept)')
        old = backup(dst, fix)
    else:
        say('FIX' if fix else 'WARN', name, 'missing -> will install at %s' % dst); old = None
    if fix:
        shutil.copytree(src, dst)
        for k in keep:
            if old and os.path.exists(os.path.join(old, k)) and not os.path.exists(os.path.join(dst, k)):
                shutil.copytree(os.path.join(old, k), os.path.join(dst, k))
    ACTIONS.append('%s %s at %s' % ('installed' if fix else 'would install', name, dst))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='/data/home/qnxuser/blissey-beacon')
    p.add_argument('--check', action='store_true', help='inspect only; change nothing')
    a = p.parse_args(); fix = not a.check; root = os.path.abspath(a.root)
    kit = json.load(open(os.path.join(KIT, 'KIT_MANIFEST.json')))
    print('Blissey Pi kit %s | transport %s | model %s' % (kit['created'], kit['transport_commit'][:12], kit['model_kind']))
    print('Python %s on %s; root %s; mode %s\n' % (sys.version.split()[0], sys.platform, root, 'CHECK ONLY' if a.check else 'INSTALL/REPAIR'))

    # 0. What is there now
    if os.path.isdir(root):
        items = sorted(os.listdir(root))
        say('INFO', 'current contents', ', '.join(items) or '(empty)')
        stray = [i for i in items if i in ('apps', 'packages', 'tools', 'supabase', 'node_modules', 'package.json')]
        if stray:
            say('INFO', 'repo copy found', '%s are blissey-beacon backend/web files; the Pi does not run them (left untouched)' % stray)
    else:
        say('FIX' if fix else 'WARN', 'root', 'missing -> will create %s' % root)
        if fix:
            os.makedirs(root, mode=0o700)
    for d in ('config', 'outbox', 'logs'):
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            say('FIX' if fix else 'WARN', d + '/', 'missing -> will create (mode 700)')
            if fix:
                os.makedirs(path, mode=0o700)

    # 1. Transport (Emilia's verified sender) + vendored websockets
    install_tree(os.path.join(KIT, 'transport'), os.path.join(root, 'transport'), 'transport code', fix, keep=('vendor',))
    vendor = os.path.join(root, 'transport', 'vendor')
    if os.path.isdir(os.path.join(vendor, 'websockets')):
        say('PASS', 'websockets library', 'present in transport/vendor')
    else:
        wheel = glob.glob(os.path.join(KIT, 'wheels', 'websockets-*.whl'))[0]
        say('FIX' if fix else 'WARN', 'websockets library', 'missing -> will unpack %s (pure Python, no pip needed)' % os.path.basename(wheel))
        if fix:
            os.makedirs(vendor, exist_ok=True); zipfile.ZipFile(wheel).extractall(vendor)

    # 2. Detector bundle (model + runtime)
    install_tree(os.path.join(KIT, 'detector'), os.path.join(root, 'detector'), 'detector bundle', fix)

    # 3. Config: identity must be device "1" / room "101"
    cfg_path = os.path.join(root, 'config', 'device.json')
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
        except ValueError:
            cfg = None
        wrong = {k: (cfg or {}).get(k) for k, v in CONFIG.items() if (cfg or {}).get(k) != v and k != 'ca_file'}
        if not wrong:
            say('PASS', 'config/device.json', 'device "1", room "101", %s' % URL)
        else:
            say('FIX' if fix else 'WARN', 'config/device.json', 'wrong/missing %s -> will rewrite (backup kept)' % wrong)
            if fix:
                shutil.copyfile(cfg_path, cfg_path + '.bak-' + time.strftime('%Y%m%d-%H%M%S'))
                keep_ca = {'ca_file': cfg['ca_file']} if cfg and cfg.get('ca_file') else {}
                json.dump(dict(CONFIG, **keep_ca), open(cfg_path, 'w'), indent=2)
    else:
        say('FIX' if fix else 'WARN', 'config/device.json', 'missing -> will create with device "1", room "101"')
        if fix:
            json.dump(CONFIG, open(cfg_path, 'w'), indent=2)

    # 4. Token: must be the EXISTING registered one; never created or printed here
    tok = os.path.join(root, 'config', 'device-token')
    if not os.path.exists(tok):
        say('FAIL', 'config/device-token', 'MISSING. Restore the token that was registered on 2026-10-18 (its SHA-256 is '
            'in the backend). Do NOT generate a new one; the backend would reject it.')
    else:
        st = os.stat(tok)
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
            say('FIX' if fix else 'WARN', 'config/device-token', 'mode %o is too open -> chmod 600' % stat.S_IMODE(st.st_mode))
            if fix:
                os.chmod(tok, 0o600)
        with open(tok, 'rb') as f:
            digest = hashlib.sha256(f.read().strip()).hexdigest()
        say('PASS', 'config/device-token', 'present, mode %o; SHA-256 %s... (compare with the registered digest)'
            % (stat.S_IMODE(os.stat(tok).st_mode), digest[:16]))

    # 5. Supervisor + launchers
    sup_src, sup_dst = os.path.join(KIT, 'supervise.py'), os.path.join(root, 'supervise.py')
    if os.path.exists(sup_dst) and open(sup_dst, 'rb').read() == open(sup_src, 'rb').read():
        say('PASS', 'supervise.py', 'up to date')
    else:
        say('FIX' if fix else 'WARN', 'supervise.py', 'will install (starts driver if needed, runs + restarts both processes)')
        if fix:
            shutil.copyfile(sup_src, sup_dst)
    scripts = {
        'start.sh': '#!/bin/sh\n# Start everything: USB-serial driver (if needed) + transport + detector, supervised.\n'
                    '# Usage: sh start.sh [/dev/serusb1]   (Ctrl-C stops; logs in logs/)\n'
                    'exec python3 "$(cd "$(dirname "$0")" && pwd)/supervise.py" --serial "${1:-/dev/serusb1}"\n',
        'run-transport.sh': '#!/bin/sh\n# Sender: delivers queued events + health over WSS (exit 2/3 = needs operator repair; do not auto-restart)\n'
                            'exec sh "$(dirname "$0")/transport/run-client.sh" "$(cd "$(dirname "$0")" && pwd)/config/device.json"\n',
        'run-detector.sh': '#!/bin/sh\n# Producer: ESP32 AP serial -> model -> one event per episode + health. Usage: run-detector.sh /dev/serusbN\n'
                           '[ "$#" -ge 1 ] || { echo "usage: $0 /dev/serusbN [extra service args]" >&2; exit 2; }\n'
                           'ROOT="$(cd "$(dirname "$0")" && pwd)"; DEV="$1"; shift\n'
                           'cd "$ROOT/detector" && exec python3 -m blissey_edge.service --serial "$DEV" --qnx-config "$ROOT/config/device.json" '
                           '--threshold-file "$ROOT/config/threshold.json" "$@"\n'}
    for name, body in scripts.items():
        path = os.path.join(root, name)
        if os.path.exists(path) and open(path).read() == body:
            say('PASS', name, 'up to date')
            continue
        say('FIX' if fix else 'WARN', name, 'will write')
        if fix:
            with open(path, 'w') as f:
                f.write(body)
            os.chmod(path, 0o755)

    # 6. Verify what is installed
    if fix:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([vendor, os.path.join(root, 'transport')]), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([sys.executable, '-c', 'import blissey_qnx.client, websockets; print(websockets.__version__)'],
                           env=env, capture_output=True, text=True)
        say('PASS' if r.returncode == 0 else 'FAIL', 'transport imports', r.stdout.strip() or r.stderr.strip()[-300:])
        r = subprocess.run([sys.executable, '-B', '-m', 'blissey_edge.selftest', 'model.json', 'golden.json'],
                           cwd=os.path.join(root, 'detector'), capture_output=True, text=True)
        say('PASS' if r.returncode == 0 else 'FAIL', 'detector selftest',
            (r.stdout.strip().splitlines() or ['?'])[-1] + ('; ' + [l for l in r.stdout.splitlines() if 'latency' in l][0].strip()
                                                           if 'latency' in r.stdout else ''))
    devs = sorted(sum((glob.glob(g) for g in ('/dev/serusb*', '/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ser[0-9]*')), []))
    say('INFO', 'serial devices', ', '.join(devs) or 'none (plug the AP receiver ESP32 into the Pi)')

    print('\n' + ('Changes made:' if fix else 'Changes needed:'))
    for x in ACTIONS or ['none']:
        print('  - ' + x)
    cal = os.path.join(root, 'config', 'threshold.json')
    say('PASS' if os.path.exists(cal) else 'INFO', 'on-site calibration',
        ('config/threshold.json present (%.4f)' % json.load(open(cal))['threshold']) if os.path.exists(cal)
        else 'none yet: run blissey_edge.calibrate in the demo room to cut false alarms')
    print('\nNext:\n  1. cd %s/detector && python3 -m blissey_edge.pi_check --serial /dev/serusb1   (expect roles {AP})' % root
          + '\n  2. sh %s/start.sh /dev/serusb1        (starts driver if needed + both processes; logs in logs/)' % root
          + '\n     or separately: sh run-transport.sh   and   sh run-detector.sh /dev/serusb1')
    if PROBLEMS:
        print('\nAttention:'); [print('  - ' + x) for x in PROBLEMS]
    return 1 if any(x.startswith('FAIL') for x in PROBLEMS) else 0


if __name__ == '__main__':
    sys.exit(main())

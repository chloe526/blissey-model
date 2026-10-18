"""On-site threshold calibration (run on the Pi in the demo room, with NO falls, service stopped):

    cd ~/blissey-beacon/detector
    python3 -m blissey_edge.calibrate --serial /dev/serusb1 --minutes 5

While it runs, do normal activity where the demo happens: walk through, sit, stand, people nearby, but no
falls and no lying on the floor. It scores every window with the deployed model and finds the lowest
threshold at which the alert rule (2 of the last 3 windows >= threshold) would have fired ZERO times. It
then multiplies that by a safety margin and writes ../config/threshold.json. The detector service (and
start.sh) use that file automatically; delete it to go back to the model's trained threshold.

Trade-off: a higher threshold means fewer false alarms, but weak falls may be missed. Rehearse a few falls
after calibrating; if they are missed, recalibrate with calmer activity or lower --margin.
Standard library only.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
import sys
import time

from .csi import PacketError, parse_record, parse_serial_line
from .detector import Detector
from .model import Model


def zero_alert_threshold(windows, policy):
    """windows: [(window_end, score or None)] in time order. Smallest t with no 2-of-3 alert is just above
    the largest 'second-highest score among any 3 contiguous valid windows'."""
    worst, run, prev = 0.0, [], None
    for end, score in windows:
        if score is None or (prev is not None and end - prev > policy['reset_after_gap_seconds'] + 1e-9):
            run = []
        if score is not None:
            run = (run + [score])[-policy['history']:]
            if len(run) >= policy['warmup_windows']:
                worst = max(worst, sorted(run, reverse=True)[policy['votes'] - 1])
        prev = end
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--serial')
    src.add_argument('--replay', help='testing: a recorded measurements.jsonl of normal activity')
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--minutes', type=float, default=5.0)
    p.add_argument('--margin', type=float, default=1.15, help='multiply the zero-alert threshold by this')
    p.add_argument('--model', default='model.json')
    p.add_argument('--out', default=os.path.join('..', 'config', 'threshold.json'))
    p.add_argument('--source-mac', default=None)
    a = p.parse_args(argv)
    model = Model.load(a.model)
    policy = model.bundle['alert_policy']
    det = Detector(model, source_mac=a.source_mac)
    windows, packets, t_end = [], 0, time.monotonic() + 60 * a.minutes
    print('Calibrating for %.1f min: normal activity only, NO falls. Ctrl-C to finish early.' % a.minutes, flush=True)
    try:
        if a.serial:
            from .serial_source import SerialSource
            src = SerialSource(a.serial, a.baud).open()
            last = time.monotonic()
            while time.monotonic() < t_end:
                line = src.readline(timeout=1.0)
                if not line or not line.startswith('CSI_DATA'):
                    continue
                try:
                    mac, role, amps = parse_serial_line(line)
                except PacketError:
                    continue
                if role != 'AP':
                    continue
                packets += 1
                windows += [(d.window_end, d.fall_score) for d in det.push(time.monotonic(), amps, mac)]
                if time.monotonic() - last > 30:
                    last = time.monotonic()
                    print('  %3.0f s left: %d packets, %d windows' % (t_end - last, packets, len(windows)), flush=True)
            src.close()
        else:
            with open(a.replay) as f:
                for text in f:
                    try:
                        t, mac, role, amps = parse_record(text)
                    except (PacketError, KeyError, ValueError):
                        continue
                    packets += 1
                    windows += [(d.window_end, d.fall_score) for d in det.push(t, amps, mac)]
    except KeyboardInterrupt:
        print('stopped early')
    valid = [s for _, s in windows if s is not None]
    if len(valid) < 60:
        print('Only %d valid windows (need >= 60, about 30 s); check the AP board / driver. Nothing written.' % len(valid))
        return 1
    base = model.bundle['threshold']
    alerts_at_model = sum(1 for i in range(len(valid)) if i >= 2 and sum(v >= base for v in valid[i - 2:i + 1]) >= 2)
    t0 = zero_alert_threshold(windows, policy)
    thr = min(max(base, t0 * a.margin), 0.95)
    with open(a.model, 'rb') as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()
    result = dict(threshold=thr, model_threshold=base, zero_alert_threshold=t0, margin=a.margin,
                  model_sha256=model_sha, windows=len(valid), packets=packets, minutes=round(a.minutes, 2),
                  source_mac=det.mac, windows_over_model_threshold=sum(v >= base for v in valid),
                  calibrated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  note='Normal-activity calibration; delete this file to use the model threshold.')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, a.out)
    pct = sorted(valid)
    print('\n%d windows from %s. Score percentiles: 50%% %.3f  95%% %.3f  99%% %.3f  max %.3f'
          % (len(valid), det.mac, pct[len(pct) // 2], pct[int(.95 * (len(pct) - 1))], pct[int(.99 * (len(pct) - 1))], pct[-1]))
    print('At the trained threshold %.4f this period would have had ~%d raw alert windows.' % (base, alerts_at_model))
    print('Zero-alert threshold %.3f x margin %.2f -> NEW THRESHOLD %.3f written to %s' % (t0, a.margin, thr, a.out))
    if thr >= 0.5:
        print('WARNING: high threshold. Normal activity here already scores high, so weaker falls may be missed. '
              'Rehearse falls now; if they are missed, recalibrate with calmer activity or a smaller --margin.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

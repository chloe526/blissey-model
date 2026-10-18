"""On-device check (run on the Pi / QNX before going live):

    python3 -m blissey_edge.selftest model.json golden.json [--backend tflite|python]

Verifies SHA256SUMS if present, loads the model, replays the bundled raw packets through the exact
streaming path, compares every window's fall score and alert with the training machine's results,
and reports per-window latency on this device. Exit code 0 = identical, 1 = mismatch.
With a TFLite model (float32), scores may differ from the training machine by up to the tolerance recorded
in model.json; alerts must still match exactly.
"""
import hashlib
import json
import os
import sys
import time

from .csi import parse_record
from .detector import Detector
from .model import Model


def check_sums(folder):
    path = os.path.join(folder, 'SHA256SUMS')
    if not os.path.exists(path):
        return 'SHA256SUMS not found (skipped)'
    bad = []
    with open(path) as f:
        for line in f:
            digest, name = line.strip().split('  ', 1)
            with open(os.path.join(folder, name), 'rb') as g:
                if hashlib.sha256(g.read()).hexdigest() != digest:
                    bad.append(name)
    if bad:
        raise SystemExit('Checksum mismatch: ' + ', '.join(bad))
    return 'all checksums match'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    backend = None
    if '--backend' in argv:
        i = argv.index('--backend'); backend = argv[i + 1]; del argv[i:i + 2]
    if len(argv) != 2:
        raise SystemExit(__doc__)
    model_path, golden_path = argv
    print('checksums:', check_sums(os.path.dirname(os.path.abspath(model_path))))
    model = Model.load(model_path, backend)
    with open(golden_path) as f:
        gold = json.load(f)
    det = Detector(model); got = []; latencies = []
    for rec in gold['packets']:
        t, mac, _, amps = parse_record(rec)
        tick = time.perf_counter()
        decisions = det.push(t, amps, mac)
        if decisions:
            latencies.append((time.perf_counter() - tick) / len(decisions))
        got += [[d.window_end, d.fall_score, d.alert] for d in decisions]
    exp = gold['expected']; tol = max(gold.get('tolerance', 1e-9), model.score_tolerance); bad = 0
    if len(got) != len(exp):
        print('window count differs: got %d, expected %d' % (len(got), len(exp))); bad += 1
    for (e1, s1, a1), (e2, s2, a2) in zip(got, exp):
        same = abs(e1 - e2) < 1e-9 and a1 == a2 and ((s1 is None and s2 is None) or
                                                     (s1 is not None and s2 is not None and abs(s1 - s2) <= tol))
        if not same:
            bad += 1
            if bad <= 5:
                print('mismatch at window end %.3f: got %r/%r expected %r/%r' % (e1, s1, a1, s2, a2))
    latencies.sort()
    print('model: %s, %d features, threshold %.4f, backend %s'
          % (model.bundle['model_kind'], model.n_features, model.bundle['threshold'], model.backend))
    print('windows compared: %d, mismatches: %d' % (len(exp), bad))
    if latencies:
        print('per-window latency on this device: median %.1f ms, max %.1f ms'
              % (1000 * latencies[len(latencies) // 2], 1000 * latencies[-1]))
    print('PYTHON', sys.version.split()[0], sys.platform)
    print('RESULT:', 'PASS' if not bad else 'FAIL')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())

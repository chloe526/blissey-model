"""python -m unittest lahacks_v2.test_v2 -v   (parity test needs the pinned raw cache)."""
import unittest
from pathlib import Path

import numpy as np

from .evaluate import contiguous_ok, evaluate
from .features import extract, motion_score
from .windows import DEFAULT_RAW, gain_normalize


class GainTransform(unittest.TestCase):
    def test_exactly_invariant_to_common_gain_and_global_scale(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(5, 30, (52, 512)).astype(np.float32)
        common = rng.uniform(.5, 2, (1, 512)).astype(np.float32)       # AGC-like per-sample gain
        np.testing.assert_allclose(gain_normalize(x), gain_normalize(x * common * 1.7), rtol=1e-4, atol=1e-5)

    def test_output_centred_per_channel(self):
        x = np.random.default_rng(2).uniform(5, 30, (52, 512)).astype(np.float32)
        np.testing.assert_allclose(np.median(gain_normalize(x), axis=-1), 1, atol=1e-5)

    def test_batch_equals_single(self):
        x = np.random.default_rng(1).uniform(5, 30, (3, 52, 512)).astype(np.float32)
        np.testing.assert_allclose(gain_normalize(x)[1], gain_normalize(x[1]), rtol=1e-6)
        np.testing.assert_allclose(extract(gain_normalize(x))[1], extract(gain_normalize(x[1:2]))[0], rtol=1e-5)
        self.assertAlmostEqual(float(np.log(motion_score(gain_normalize(x[1])))), float(extract(gain_normalize(x[1:2]))[0, 0]), places=4)


@unittest.skipUnless((DEFAULT_RAW / 'session_04_LAHACKS').exists(), 'raw cache absent')
class LiveParity(unittest.TestCase):
    def test_live_window_matches_offline_features(self):
        from csi import load_recording
        from preprocess import recording_features
        from lahacks_cv.data import geometry
        from lahacks_cv.infer import live_window
        from csi_quality import window_quality
        cfg = geometry()
        folder = sorted((DEFAULT_RAW / 'session_04_LAHACKS/staged_fall').glob('*'))[3]
        _, rows, *_ = load_recording(folder)
        # live: packets arrive one by one; build features from only the packets seen so far
        times, amp, _ = recording_features(rows, cfg['channel_indices'])
        checked = 0
        for w0 in np.arange(times[0], times[-1] - 3 + 1e-9, .5):
            if window_quality(times, float(w0), cfg)['reasons']:
                continue
            seen = [p for p in rows if p['elapsed_seconds'] <= w0 + 3 + .3]   # packets up to just past window end
            t2, a2, _ = recording_features(seen, cfg['channel_indices'])
            live = extract(gain_normalize(live_window(t2, a2, float(w0), cfg))[None])
            offline = extract(gain_normalize(live_window(times, amp, float(w0), cfg))[None])
            np.testing.assert_array_equal(live, offline); checked += 1
        self.assertGreater(checked, 3)

    def test_gate_free_interpolation_equals_transform_window(self):
        from csi import load_recording
        from preprocess import recording_features, transform_window
        from lahacks_cv.data import geometry
        from csi_quality import window_quality
        from .windows import interpolate
        cfg = geometry()
        folder = sorted((DEFAULT_RAW / 'session_03_LAHACKS/walking').glob('*'))[0]
        _, rows, *_ = load_recording(folder)
        t, a, _ = recording_features(rows, cfg['channel_indices'])
        n = 0
        for w0 in np.arange(t[0], t[-1] - 3 + 1e-9, .5):
            if not window_quality(t, float(w0), cfg)['reasons']:
                np.testing.assert_array_equal(interpolate(t, a, float(w0), cfg),
                                              transform_window(t, a, float(w0), cfg, normalize=False)); n += 1
        self.assertGreater(n, 10)


class Evaluator(unittest.TestCase):
    def test_contiguity(self):
        e = np.array([3., 3.5, 4., 5.5, 6., 6.5])
        self.assertTrue(contiguous_ok(e, 2)); self.assertFalse(contiguous_ok(e, 3)); self.assertTrue(contiguous_ok(e, 5))

    def test_detection_false_alert_and_post_event(self):
        n = 30; start = np.arange(n) * .5
        data = dict(recording_id=np.array(['f'] * n), group=np.array(['g'] * n), start=start, end=start + 3,
                    label_v2=np.where((start >= 5) & (start <= 7), 1, np.where(start < 3, 0, -1)))
        records = {'f': dict(group='g', status='annotated_fall', intervals=[[8., 12.]])}
        score = np.zeros(n); score[[0, 1, 2]] = 1; score[[10, 11, 12]] = 1; score[[26, 27, 28]] = 1
        r = evaluate(data, records, np.ones(n, bool), score, .5)
        self.assertEqual((r['detected_events'], r['false_alerts_pre_action'], r['post_event_alerts']), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()

"""Focused safeguards plus a tiny SYNTHETIC CPU integration test, never real training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from cnn_baseline.model import build_model, weighted_cross_entropy
from preprocess import label_window, transform_window
from scripts.archive_preflight import preflight
from .common import (FOLDS, GROUPS, ROOT, canonical_hash, class_weights, fit_normalization,
                     fold_indices, inner_recording_split, normalize, sha)
from .data import geometry, load_dataset, read_source, verify_code
from .features import extract, extract_one
from .infer import live_window, load_fold
from .metrics import alert_decisions, classification, event_summary
from .submit import package
from .train import benchmark, seed_everything, state_digest


def fixture():
    rng = np.random.default_rng(17)
    values = {k: [] for k in ('X', 'y', 'group', 'recording_id', 'event_id', 'start', 'end')}
    records = {}
    for group in GROUPS:
        for y in (0, 1):
            for number in range(2):
                rid = f'{group}_{y}_{number}'
                records[rid] = dict(group=group, status='annotated_fall' if y else 'verified_non_fall',
                                    intervals=[[0., 4.]] if y else [])
                for i in range(3):
                    x = (2 + rng.normal(0, .1, (52, 512)) + y * .2 * np.sin(np.linspace(0, 15, 512))).astype(np.float32)
                    row = dict(X=x, y=y, group=group, recording_id=rid,
                               event_id=f'{rid}:fall:0' if y else '', start=.5 * i, end=3. + .5 * i)
                    for k, v in row.items():
                        values[k].append(v)
    data = {k: np.asarray(v) for k, v in values.items()}
    scalers = {f: fit_normalization(data['X'], fold_indices(data, f)[0]) for f in FOLDS}
    meta = dict(records=records, fold_normalization=scalers, geometry=geometry(),
                source_manifest_sha256='synthetic-unit-test-only', artifact_sha256='synthetic-unit-test-only')
    return data, meta


class Safeguards(unittest.TestCase):
    def test_source_is_pinned_lahacks_only(self):
        source = read_source(); verify_code(source)
        self.assertEqual({p['path'].split('/')[0] for p in source['files']}, set(GROUPS))
        bad = copy.deepcopy(source); bad['files'][0]['path'] = 'session_02/fall/metadata.json'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'bad.json'; p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_source(p)

    def test_folds_and_physical_events_do_not_overlap(self):
        data, _ = fixture()
        for fold in FOLDS:
            train, val = fold_indices(data, fold)
            self.assertFalse(set(data['recording_id'][train]) & set(data['recording_id'][val]))
            inner, dev = inner_recording_split(data, train, 42)
            self.assertFalse(set(data['recording_id'][inner]) & set(data['recording_id'][dev]))
            self.assertTrue(set(inner) | set(dev) == set(train))
        bad = {k: v.copy() for k, v in data.items()}
        bad['group'][0] = 'session_02'
        with self.assertRaises(ValueError):
            fold_indices(bad, 'A')
        train, val = fold_indices(data, 'A')
        data['event_id'][val[0]] = data['event_id'][train[-1]]
        with self.assertRaises(ValueError):
            fold_indices(data, 'A')

    def test_normalization_ignores_validation_and_is_fold_specific(self):
        data, _ = fixture(); train, val = fold_indices(data, 'A')
        a = fit_normalization(data['X'], train)
        data['X'][val] += 1000
        self.assertEqual(a, fit_normalization(data['X'], train))
        b = fit_normalization(data['X'], val)
        self.assertFalse(np.allclose(a['mean'], b['mean']))
        normalized = normalize(data['X'][train], a)
        np.testing.assert_allclose(normalized.mean(axis=(0, 2)), 0, atol=1e-5)
        np.testing.assert_allclose(normalized.std(axis=(0, 2)), 1, atol=1e-5)

    def test_original_quality_and_labels_are_frozen(self):
        source = read_source(); verify_code(source)
        self.assertEqual(canonical_hash(geometry()), source['geometry_sha256'])
        c = geometry(); q = c['packet_quality']
        self.assertEqual((c['window_seconds'], c['stride_seconds'], c['sequence_length']), (3., .5, 512))
        self.assertEqual((q['coverage_bin_seconds'], q['min_occupied_bin_fraction'],
                          q['min_sample_span_fraction'], q['max_edge_distance_seconds'],
                          q['max_gap_seconds'], q['min_effective_rate_hz']), (.1, .8, .9, .1, .25, 20.))
        self.assertEqual(label_window(.25, 3.25, 'annotated_fall', [[0., 3.5]]), 1)
        self.assertIsNone(label_window(0, 3, 'annotated_fall', [[0., 3.5]]))

    def test_live_and_offline_window_features_match(self):
        times = np.arange(0, 3.1, .025)
        amplitude = np.tile((2 + .1 * np.sin(times))[:, None], (1, 52))
        a = transform_window(times, amplitude, 0., geometry(), normalize=False)
        b = live_window(times, amplitude, 0., geometry())
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(extract_one(a), extract_one(b))
        keep = (times < 1.) | (times > 1.5)
        with self.assertRaises(ValueError):
            live_window(times[keep], amplitude[keep], 0., geometry())

    def test_rf_features_are_compact_finite_and_window_local(self):
        data, _ = fixture()
        with threadpool_limits(2):
            x = extract_one(data['X'][0])
            self.assertEqual(x.shape, (575,))
            self.assertTrue(np.isfinite(extract_one(np.zeros((52, 512)))).all())
            np.testing.assert_allclose(x, extract_one(data['X'][0] * 5), rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(x, extract(data['X'][:2])[0])

    def test_weighted_loss_keeps_positive_weight_at_batch_one(self):
        logits = torch.zeros((1, 2), requires_grad=True)
        target = torch.tensor([1])
        weighted = weighted_cross_entropy(logits, target, torch.tensor([1., 4.]))
        plain = weighted_cross_entropy(logits, target)
        self.assertAlmostEqual(float((weighted / plain).detach()), 4.)
        weighted.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(class_weights(np.r_[np.zeros(100, dtype=int), 1]).tolist(), [1., 4.])

    def test_seed_and_rf_artifact_reproducibility(self):
        cfg = json.loads((ROOT / 'lahacks_cv/config.json').read_text())['cnn']
        seed_everything(42, 2); a = state_digest(build_model(cfg))
        seed_everything(42, 2); self.assertEqual(a, state_digest(build_model(cfg)))
        data, _ = fixture()
        with threadpool_limits(2):
            x = extract(data['X'][:12]); y = data['y'][:12]
        a = RandomForestClassifier(n_estimators=3, random_state=42, n_jobs=1).fit(x, y)
        b = RandomForestClassifier(n_estimators=3, random_state=42, n_jobs=1).fit(x, y)
        np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))
        with tempfile.TemporaryDirectory() as td:
            pa, pb = Path(td) / 'a.joblib', Path(td) / 'b.joblib'
            joblib.dump(a, pa, compress=3); joblib.dump(b, pb, compress=3)
            self.assertEqual(sha(pa), sha(pb))

    def test_alerts_reset_gaps_and_obey_cooldown(self):
        np.testing.assert_array_equal(alert_decisions([3., 3.5, 4., 4.5, 5., 5.5], [.9] * 6),
                                      [False, False, True, False, False, False])
        np.testing.assert_array_equal(alert_decisions([3., 3.5, 5., 5.5, 6.], [.9] * 5),
                                      [False, False, False, False, True])
        m = classification([0, 0, 1, 1], [.1, .8, .2, .9])
        self.assertEqual(m['confusion_matrix'], [[1, 1], [1, 1]])
        e = event_summary([dict(covered=True, detected=True), dict(covered=False, detected=False)], 2, 120)
        self.assertEqual(e['all_annotated_event_recall'], .5)
        self.assertEqual(e['false_alerts_per_hour'], 60.)

    def test_packaging_excludes_datasets_weights_and_old_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            stage = package(td); stats = preflight(stage)
            self.assertTrue(stats['passed'])
            files = [str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()]
            self.assertFalse(any(p.endswith(('.npz', '.pt', '.joblib')) or 'moment' in p or '.venv' in p for p in files))
            (stage / 'extra.npz').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                preflight(stage)

    def test_synthetic_cpu_job_saves_reloads_both_models_and_equal_inputs(self):
        data, meta = fixture()
        cfg = json.loads((ROOT / 'lahacks_cv/config.json').read_text())
        cfg['cnn'].update(epochs=1, patience=1, batch_size=4)
        cfg['rf']['n_estimators'] = 3; cfg['threads'] = 2
        with tempfile.TemporaryDirectory() as td, threadpool_limits(2):
            folder = Path(td) / 'synthetic_run'
            result = benchmark(data, meta, cfg, folder, 'cpu')
            self.assertFalse(result['deployment_refit_performed'])
            for fold in FOLDS:
                f = result['folds'][fold]
                self.assertEqual(f['cnn']['equivalent_input_hashes'], f['rf']['equivalent_input_hashes'])
                self.assertEqual(f['cnn']['parameter_count'], 21666)
                self.assertTrue(f['cnn']['checkpoint_reload_verified'] and f['rf']['checkpoint_reload_verified'])
                _, val = fold_indices(data, fold)
                with np.load(folder / ('fold_' + fold) / 'predictions.npz') as z:
                    for name in ('cnn', 'rf'):
                        predict, _ = load_fold(folder / ('fold_' + fold), name)
                        self.assertAlmostEqual(predict(data['X'][val[0]]), float(z[name + '_probability'][0]), places=6)
            with self.assertRaises(FileExistsError):
                benchmark(data, meta, cfg, folder, 'cpu')


class RealDataVerification(unittest.TestCase):
    def test_audited_real_dataset_and_fold_scalers(self):
        path = ROOT / 'data/lahacks_quality_v2_cv_v1'
        if not path.exists():
            self.skipTest('Run the documented local preparation first')
        data, meta = load_dataset(path)
        self.assertEqual(data['X'].shape, (1602, 52, 512))
        self.assertEqual(np.bincount(data['y']).tolist(), [1550, 52])
        self.assertEqual((len(meta['records']), meta['annotated_events'], meta['quality_covered_events']), (92, 34, 13))
        for fold, counts in (('A', [1249, 21]), ('B', [301, 31])):
            train, val = fold_indices(data, fold)
            self.assertEqual(np.bincount(data['y'][train]).tolist(), counts)
            self.assertEqual(fit_normalization(data['X'], train)['mean'], meta['fold_normalization'][fold]['mean'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

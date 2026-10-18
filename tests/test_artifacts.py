"""Integration checks on actual processed data; no pretrained downloads."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from csi import load_recording
from evaluate import temporal_metrics,union_duration
from preprocess import recording_features,transform_window
from training_data import CSIDataset,load_data

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT/'data/processed/preprocessing_config.json').exists(),'Run preprocess.py first')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data,cls.config=load_data(ROOT/'data/processed')

    def test_actual_session_separation_and_classes(self):
        groups=[]
        for split,d in self.data.items():
            self.assertEqual(set(d.y),{0,1});self.assertEqual(d.X.shape[1:],(52,512))
            groups.append({self.config['recordings'][rid]['group'] for rid in d.recording_id})
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        train=self.data['train'].X
        np.testing.assert_allclose(train.mean(axis=(0,2)),0,atol=2e-4)
        np.testing.assert_allclose(train.std(axis=(0,2)),1,atol=2e-4)

    def test_unseen_raw_window_matches_saved_transform(self):
        d=self.data['test'];rid=str(d.recording_id[0])
        root=ROOT.parent/'blissey-data/recordings'
        candidates=list(root.rglob(f'{rid}/metadata.json'));self.assertEqual(len(candidates),1)
        m,rows,*_=load_recording(candidates[0].parent)
        times,features,dupes=recording_features(rows,self.config['channel_indices'])
        x=transform_window(times,features,float(d.start[0]),self.config)
        np.testing.assert_allclose(x,d.X[0],rtol=1e-5,atol=2e-6)

    def test_nonfinite_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.npz'
            np.savez(p,X=np.full((1,52,512),np.nan,dtype=np.float32),y=np.array([0],dtype=np.int64),
                recording_id=np.array(['x']),start_seconds=np.array([0]),end_seconds=np.array([3]))
            with self.assertRaisesRegex(ValueError,'Non-finite'):CSIDataset(p,52)

    def test_declared_split_group_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=json.loads(json.dumps(self.config))
            config['group_splits']['val']=['incorrect_group']
            (root/'preprocessing_config.json').write_text(json.dumps(config))
            for split in ('train','val','test'):
                (root/f'{split}.npz').symlink_to(ROOT/'data/processed'/f'{split}.npz')
            with self.assertRaisesRegex(ValueError,'assigned collection group'):
                load_data(root)

    def test_exposure_uses_union_not_window_sum(self):
        self.assertEqual(union_duration([(0,3),(.5,3.5),(10,13)]),6.5)
        # Synthetic predictions exercise metric code, not model performance.
        result=temporal_metrics(self.data['test'],np.zeros(len(self.data['test']),dtype=int),self.config)
        self.assertEqual(result['false_positive_alerts'],0)
        self.assertIsNone(result['detection_latency_seconds'])
        self.assertGreater(result['normal_monitored_seconds'],0)
        self.assertEqual(result['covered_annotated_events']+result['uncovered_events'],4)

if __name__=='__main__':unittest.main()

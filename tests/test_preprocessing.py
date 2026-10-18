import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from csi import annotations,amplitudes
from preprocess import label_window,recording_features,transform_window
from metrics import classification_metrics


def packet(t=0):
    fields=['0']*25;fields[0]='CSI_DATA';fields[16]='6';fields[21]='0'
    return dict(csi=[3,4]*64,firmware_fields=fields,elapsed_seconds=t,
        host_time=f'2026-10-17T00:00:{t:06.3f}+00:00')


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(window_seconds=3,sequence_length=512,channel_indices=[6,7],
            reference_rate_hz=30,max_gap_seconds=0.25,mean=[1,2],std=[2,4])

    def test_complex_amplitude_and_invalid_bytes_excluded(self):
        p=packet();p['csi'][:4]=[127,127,127,127]
        np.testing.assert_equal(amplitudes(p,[6,7]),[5,5])

    def test_no_events_does_not_mean_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=dict(activity='staged_fall',status='complete',category_is_recording_label=True)
            self.assertEqual(annotations(tmp,m,[packet()])[0],'ambiguous')
            m['activity']='walking'
            self.assertEqual(annotations(tmp,m,[packet()])[0],'verified_non_fall')
            m['category_is_recording_label']=False
            self.assertEqual(annotations(tmp,m,[packet()])[0],'ambiguous')

    def test_host_aligned_event_not_ui_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[dict(event='action_start',host_time=packet(2)['host_time'],elapsed_seconds=20),
                dict(event='action_end',host_time=packet(6)['host_time'],elapsed_seconds=24)]
            Path(tmp,'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
            status,intervals,issues=annotations(tmp,dict(activity='staged_fall',status='complete',category_is_recording_label=True),[packet(0),packet(10)])
            self.assertEqual(status,'annotated_fall');np.testing.assert_allclose(intervals,[[2,6]])
            self.assertTrue(any('clock' in x for x in issues))

    def test_partial_and_unverified_labels_rejected(self):
        self.assertEqual(label_window(2.5,5.5,'annotated_fall',[[2,6]]),1)
        for a,b in [(1,4),(2,5),(7,10)]:self.assertIsNone(label_window(a,b,'annotated_fall',[[2,6]]))
        self.assertIsNone(label_window(0,3,'ambiguous',[]))

    def test_timestamp_interpolation_and_saved_scaler(self):
        t=np.arange(0,3.1,.02);x=np.stack([t,2*t],axis=1)
        out=transform_window(t,x,0,self.config)
        self.assertEqual(out.shape,(2,512));self.assertEqual(out.dtype,np.float32)
        np.testing.assert_allclose(out[:,0],[-.5,-.5]);np.testing.assert_allclose(out[:,-1],[1,1])
        # No scaler fitting on future/inference values.
        old=json.dumps(self.config);transform_window(t,x+100,0,self.config)
        self.assertEqual(json.dumps(self.config),old)

    def test_gap_and_nonfinite_rejected(self):
        for t in [np.array([0,1,2,3]),np.array([0,.1,.1,3]),np.array([0,np.nan,3])]:
            with self.assertRaises(ValueError):transform_window(t,np.ones((len(t),2)),0,self.config)
        t=np.arange(0,3.1,.02);x=np.ones((len(t),2));x[2,0]=np.inf
        with self.assertRaises(ValueError):transform_window(t,x,0,self.config)

    def test_duplicate_and_sort(self):
        t,x,count=recording_features([packet(2),packet(0),packet(2),packet(1)],[6,7])
        np.testing.assert_equal(t,[0,1,2]);self.assertEqual(count,1)

    def test_metrics(self):
        m=classification_metrics([0,0,1,1],[0,1,0,1])
        self.assertEqual(m['confusion_matrix'],[[1,1],[1,1]]);self.assertEqual(m['f1'],.5)
        self.assertIsNone(classification_metrics([0],[0])['recall_fall'])

if __name__=='__main__':unittest.main()

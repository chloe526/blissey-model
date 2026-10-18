import copy
import json
from pathlib import Path
import unittest
import numpy as np
from csi_quality import window_quality
from preprocess import recording_features,transform_window
from test_preprocessing import packet


class AcquisitionQualityTests(unittest.TestCase):
    def setUp(self):
        policy=json.loads((Path(__file__).resolve().parents[1]/'configs/packet_quality_v2.json').read_text())
        self.config=dict(window_seconds=3.,sequence_length=512,channel_indices=[6,7],
            max_gap_seconds=.25,packet_quality=policy,mean=[0.,0.],std=[1.,1.])

    def test_regular_sampling_and_moment_shape(self):
        t=np.arange(0,3.05,.025)
        self.assertEqual(window_quality(t,0,self.config)['reasons'],[])
        x=transform_window(t,np.ones((len(t),2)),0,self.config)
        self.assertEqual(x.shape,(2,512));self.assertEqual(x.dtype,np.float32)

    def test_bursty_cadence_does_not_inflate_expected_rate(self):
        # Four acquisitions every 100ms, with 1ms intra-burst intervals: 40Hz
        # wall-clock rate, but reciprocal median spacing would claim 1000Hz.
        t=np.sort(np.concatenate([np.array([0.,.001,.002,.003])+b for b in np.arange(0,3.1,.1)]))
        d=window_quality(t,0,self.config)
        self.assertEqual(d['reasons'],[]);self.assertGreater(1/np.median(np.diff(t)),900)
        self.assertLess(d['effective_rate_hz'],42)

    def test_long_gap_rejected_despite_high_total_packet_count(self):
        t=np.arange(0,3.01,.005);t=t[(t<1.)|(t>1.4)]
        self.assertIn('excessive_packet_gap',window_quality(t,0,self.config)['reasons'])

    def test_sparse_uniform_stream_rejected(self):
        t=np.arange(0,3.01,.1)
        self.assertIn('insufficient_effective_rate',window_quality(t,0,self.config)['reasons'])

    def test_many_samples_in_few_bursts_do_not_fake_temporal_coverage(self):
        # No gap exceeds 250ms and rate exceeds 20Hz, but almost half of the
        # 100ms intervals are unsupported. A count-only fix would accept this.
        t=np.sort(np.concatenate([b+np.arange(0,.01,.001) for b in np.arange(0,3.01,.2)]))
        d=window_quality(t,0,self.config)
        self.assertGreater(d['effective_rate_hz'],20)
        self.assertLess(d['max_gap_seconds'],.25)
        self.assertIn('insufficient_temporal_coverage',d['reasons'])

    def test_duplicate_timestamps_cannot_inflate_density(self):
        rows=[packet(round(t,3)) for t in np.arange(0,3.01,.1)]
        times,features,duplicates=recording_features([p for p in rows for _ in range(10)],[6,7])
        self.assertEqual(duplicates,len(rows)*9)
        self.assertIn('insufficient_effective_rate',window_quality(times,0,self.config)['reasons'])
        with self.assertRaisesRegex(ValueError,'strictly increasing'):
            window_quality(np.repeat(times,10),0,self.config)

    def test_edges_cannot_be_supported_only_by_distant_external_packets(self):
        t=np.r_[0.,np.arange(.12,3.13,.02)]
        # Window has globally bracketing data, but first in-window sample is
        # too far from this boundary even though total counts are sufficient.
        self.assertIn('unsupported_window_edge',window_quality(t,.001,self.config)['reasons'])

    def test_corruption_and_missing_endpoint_rejected(self):
        t=np.arange(0,3.01,.025)
        with self.assertRaisesRegex(ValueError,'endpoint'):
            window_quality(t,.5,self.config)
        bad=t.copy();bad[10]=np.nan
        with self.assertRaises(ValueError):window_quality(bad,0,self.config)
        with self.assertRaisesRegex(ValueError,'Non-finite|non-finite'):
            transform_window(t,np.full((len(t),2),np.nan),0,self.config)
        with self.assertRaisesRegex(ValueError,'No CSI signal'):
            transform_window(t,np.zeros((len(t),2)),0,self.config)

    def test_policy_is_not_refit_from_inference_window(self):
        old=copy.deepcopy(self.config)
        t=np.arange(0,3.01,.025);window_quality(t,0,self.config)
        self.assertEqual(old,self.config)

if __name__=='__main__':unittest.main()

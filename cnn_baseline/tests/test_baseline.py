import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from cnn_baseline.model import CSIConvNet,MODEL_TYPE,weighted_cross_entropy
from cnn_baseline.runtime import load_cnn
from cnn_baseline.metrics import classification,event_metrics
from cnn_baseline.data import load_splits,sha256

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent


class BaselineTests(unittest.TestCase):
    def test_shape_and_real_backward_at_batch_size_one(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        model=CSIConvNet();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randn(1,52,512);out=model(x)
        self.assertEqual(out.shape,(1,2));self.assertEqual(sum(p.numel() for p in model.parameters()),21666)
        before=model.head[-1].weight.detach().clone()
        loss=weighted_cross_entropy(out,torch.tensor([1]),torch.tensor([.5,8.5]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step();self.assertFalse(torch.equal(before,model.head[-1].weight))
        with self.assertRaisesRegex(ValueError,'52,512'):model(torch.randn(1,51,512))

    def test_weighted_loss_does_not_cancel_with_one_example(self):
        for target in (0,1):
            logits=torch.tensor([[.2,.1]],requires_grad=True);y=torch.tensor([target]);weights=torch.tensor([.5,8.5])
            plain=weighted_cross_entropy(logits,y)
            grad=torch.autograd.grad(plain,logits,retain_graph=True)[0]
            weighted=weighted_cross_entropy(logits,y,weights);weighted.backward()
            torch.testing.assert_close(weighted,plain*weights[target])
            torch.testing.assert_close(logits.grad,grad*weights[target])

    def test_checkpoint_roundtrip_and_moment_rejection(self):
        config=json.loads((BASE/'config.json').read_text());model=CSIConvNet().eval();x=torch.randn(1,52,512)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cnn.pt'
            torch.save(dict(model_type=MODEL_TYPE,format_version=1,config=config,state_dict=model.state_dict()),path)
            restored,_=load_cnn(path,torch.device('cpu'))
            with torch.inference_mode():torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
            torch.save(dict(schema_version=1,state_dict={}),path)
            with self.assertRaisesRegex(ValueError,'only CNN'):load_cnn(path,torch.device('cpu'))

    def test_metrics_and_uncovered_events(self):
        self.assertEqual(classification([0,0,1,1],[0,1,0,1])['confusion_matrix'],[[1,1],[1,1]])
        class Data:
            recording_id=np.array(['normal','normal','fall']);start=np.array([0.,.5,5.]);end=np.array([3.,3.5,8.]);y=np.array([0,0,1])
        config=dict(stride_seconds=.5,recordings={
            'normal':dict(split='test',label_status='verified_non_fall',fall_intervals=[]),
            'fall':dict(split='test',label_status='annotated_fall',fall_intervals=[[4.,9.],[15.,19.]])})
        result=event_metrics(Data(),[1,1,1],config,'test')
        self.assertEqual((result['detected_events'],result['covered_events'],result['uncovered_events']),(1,1,1))
        self.assertEqual(result['all_annotated_event_detection_fraction'],.5)
        self.assertEqual(result['normal_monitored_seconds'],3.5);self.assertEqual(result['false_alert_episodes'],1)

    def test_training_loader_does_not_open_test_array(self):
        with patch('cnn_baseline.data.sha256',wraps=sha256) as digest:
            data,config=load_splits(ROOT/'data/processed',('train','val'))
        self.assertEqual(set(data),{'train','val'})
        self.assertEqual([Path(call.args[0]).name for call in digest.call_args_list],['train.npz','val.npz'])
        self.assertEqual(data['train'].X.shape,(835,52,512));self.assertEqual(data['val'].X.shape,(465,52,512))
        self.assertFalse(set(data['train'].recording_id)&set(data['val'].recording_id))

    def test_no_moment_or_veri_dependencies_imported(self):
        code='import sys; import cnn_baseline.train, cnn_baseline.evaluate; assert not any(k.split(".")[0] in ("momentfm","veri_sdk","huggingface_hub") for k in sys.modules)'
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()

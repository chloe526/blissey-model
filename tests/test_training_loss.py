"""Regression: class weighting must survive the default batch size of one."""
import unittest
from unittest.mock import Mock,patch
import torch
from train import SampleWeightedCrossEntropy,log_veri


class TrainingLossTests(unittest.TestCase):
    def test_single_example_class_weight_scales_loss_and_gradient(self):
        weights=torch.tensor([835/(2*786),835/(2*49)])
        for target in (0,1):
            x=torch.tensor([[.2,.1]],requires_grad=True)
            y=torch.tensor([target])
            plain=torch.nn.functional.cross_entropy(x,y)
            old=torch.nn.functional.cross_entropy(x,y,weight=weights)
            # Documents the original failure of weighted mean reduction.
            torch.testing.assert_close(old,plain)
            original_gradient=torch.autograd.grad(plain,x,retain_graph=True)[0]
            fixed=SampleWeightedCrossEntropy(weights)(x,y)
            fixed.backward()
            torch.testing.assert_close(fixed,plain*weights[target])
            torch.testing.assert_close(x.grad,original_gradient*weights[target])

    def test_unweighted_matches_cross_entropy_and_batch_grouping(self):
        x=torch.tensor([[.2,.1],[.7,.8],[1.,0.]],requires_grad=True);y=torch.tensor([0,1,1])
        torch.testing.assert_close(SampleWeightedCrossEntropy()(x,y),torch.nn.functional.cross_entropy(x,y))
        criterion=SampleWeightedCrossEntropy(torch.tensor([.5,2.]))
        singles=torch.stack([criterion(x[i:i+1],y[i:i+1]) for i in range(3)]).mean()
        torch.testing.assert_close(criterion(x,y),singles)

    def test_veri_reward_is_validation_f1_and_loss_is_validation_loss(self):
        sdk=Mock()
        with patch.dict('sys.modules',{'veri_sdk':sdk}):
            log_veri(3,dict(f1=.25,accuracy=.99,loss=.7))
        sdk.log_metrics.assert_called_once_with(step=3,reward=.25,loss=.7)

if __name__=='__main__':unittest.main()

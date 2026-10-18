"""Small Conv1D baseline; every parameter starts randomly, with no pretrained model."""
import torch
from torch import nn

MODEL_TYPE='blissey_cnn_1d_v1'


class CSIConvNet(nn.Module):
    def __init__(self,channels=(16,32,64),kernels=(9,7,5),dropout=0.2):
        super().__init__()
        if len(channels)!=3 or len(kernels)!=3:raise ValueError('Exactly three convolution blocks required')
        if any(c<=0 or c%4 for c in channels) or any(k<=0 or k%2!=1 for k in kernels):raise ValueError('Channels must be positive multiples of four; kernels positive odd sizes')
        if not 0<=dropout<1:raise ValueError('Invalid dropout')
        blocks=[];previous=52
        for width,kernel in zip(channels,kernels):
            blocks.append(nn.Sequential(
                nn.Conv1d(previous,width,kernel_size=kernel,stride=2,padding=kernel//2,bias=False),
                nn.GroupNorm(4,width),nn.ReLU()))
            previous=width
        # GroupNorm works in training at batch size one.
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(channels[-1],2))

    def forward(self,x):
        if x.ndim!=3 or tuple(x.shape[1:])!=(52,512):raise ValueError('CNN expects [B,52,512] input')
        return self.head(self.pool(self.features(x)).squeeze(-1))


def build_model(config):
    return CSIConvNet(config['channels'],config['kernels'],config['dropout'])


def weighted_cross_entropy(logits,targets,class_weights=None):
    # Mean of per-example weighted losses: no cancellation at B=1.
    return nn.functional.cross_entropy(logits,targets,weight=class_weights,reduction='none').mean()

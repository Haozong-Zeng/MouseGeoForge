"""Frozen-feature capacity diagnostic, not an instance detector or production head."""
import torch
from torch import nn
from torch.nn import functional as F


class ForegroundLinearProbe(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(1,channels,1,1))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self, features):
        if features.ndim!=4 or features.shape[1]!=self.weight.shape[1]:
            raise ValueError('Expected NCHW features')
        return F.conv2d(F.normalize(features,dim=1,eps=1e-6),self.weight,self.bias)[:,0]


def balanced_foreground_loss(logits, fractions):
    if logits.shape!=fractions.shape:raise ValueError('Mismatched targets')
    positive=(-F.logsigmoid(logits)*fractions).sum()/fractions.sum().clamp_min(1.)
    negative=(-F.logsigmoid(-logits)*(1-fractions)).sum()/(1-fractions).sum().clamp_min(1.)
    return .5*(positive+negative)


def mask_counts(probabilities, truth):
    pred=probabilities>=.5;truth=truth.bool()
    return dict(tp=int((pred&truth).sum()),fp=int((pred&~truth).sum()),fn=int((~pred&truth).sum()))


def ratios(counts):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return dict(iou=tp/max(1,tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))

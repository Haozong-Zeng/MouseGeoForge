"""Opt-in unmerged low-rank residual around an existing frozen linear layer."""
import math
import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self,base,rank=8,alpha=8.):
        super().__init__()
        if not isinstance(base,nn.Linear) or rank<1 or alpha<=0:raise ValueError('Expected linear base and positive rank/alpha')
        self.base=base.requires_grad_(False)
        self.in_features,self.out_features=base.in_features,base.out_features
        self.rank,self.scale=rank,float(alpha)/rank
        self.a=nn.Linear(self.in_features,rank,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.b=nn.Linear(rank,self.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5));nn.init.zeros_(self.b.weight)
        self.enabled=False

    def forward(self,x):
        original=self.base(x)  # Preserve custom K-bias masking on the actual base.
        if not self.enabled:return original
        return original+self.b(self.a(x))*self.scale


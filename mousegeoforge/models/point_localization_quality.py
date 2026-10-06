"""Frozen-prediction accuracy scoring only; this is not a visibility head."""
import torch
from torch import nn
from torch.nn import functional as F


class PointLocalizationQuality(nn.Module):
    def __init__(self,mode='semantic',scalar_dim=12):
        super().__init__()
        if mode not in ['scalar','semantic']:raise ValueError('Invalid quality mode')
        self.mode=mode
        self.semantic=nn.Sequential(nn.LayerNorm(256,elementwise_affine=False),nn.Linear(256,32),nn.GELU())
        self.score=nn.Sequential(nn.Linear(32+scalar_dim,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,tokens,scalars,prior_quality,*,enabled=False):
        if not enabled:return dict(probability=prior_quality,logits=None)
        if tokens.shape[:-1]!=scalars.shape[:-1] or tokens.shape[-1]!=256 or scalars.shape[-1]!=12 or prior_quality.shape!=scalars.shape[:-1]:raise ValueError('Invalid feature dimensions')
        if not torch.isfinite(tokens).all() or not torch.isfinite(scalars).all():raise ValueError('Invalid active features')
        semantic=self.semantic(tokens.float().detach()) if self.mode=='semantic' else scalars.new_zeros(*scalars.shape[:-1],32)
        logits=self.score(torch.cat([semantic,scalars.float().detach()],-1)).squeeze(-1)
        return dict(probability=logits.sigmoid(),logits=logits)


def masked_quality_loss(logits,target,valid):
    if logits.shape!=target.shape or logits.shape!=valid.shape or logits.ndim!=3 or valid.dtype!=torch.bool:raise ValueError('Expected [images,queries,points] and Boolean mask')
    if not torch.isfinite(target[valid]).all() or ((target[valid]<0)|(target[valid]>1)).any():raise ValueError('Invalid known targets')
    clean=torch.where(valid,target,torch.zeros_like(target))
    loss=F.binary_cross_entropy_with_logits(logits,clean,reduction='none')
    count=valid.sum((-1,-2));active=count>0
    per_image=(loss*valid).sum((-1,-2))/count.clamp_min(1)
    return per_image[active].mean() if active.any() else logits.sum()*0


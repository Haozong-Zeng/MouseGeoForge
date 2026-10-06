"""Experimental supervised selection among frozen spatial candidates.

Candidate zero is the original coordinate. Prediction eligibility is computed
outside this head without labels; unary/geometry arms use the same eligibility.
This is a hard readout for evaluation and logits for supervised ranking loss,
not a differentiable triangulation or a production model factory.
"""
import torch
from torch import nn


class LocalCandidateSelector(nn.Module):
    def __init__(self,mode='geometry',scalar_dim=9,geometry_dim=6):
        super().__init__()
        if mode not in ['unary','geometry']:raise ValueError('Unknown candidate mode')
        self.mode=mode;self.scalar_dim=scalar_dim;self.geometry_dim=geometry_dim
        self.semantic=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,32),nn.GELU())
        self.local=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,32),nn.GELU())
        self.score=nn.Sequential(nn.Linear(64+scalar_dim+geometry_dim,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,semantic,local,scalars,geometry,candidate_xy,baseline_xy,eligible,*,enabled=False):
        shape=baseline_xy.shape[:-1]
        if baseline_xy.shape[-1]!=2 or semantic.shape!=(*shape,256) or local.shape!=(*shape,6,256) or scalars.shape!=(*shape,6,self.scalar_dim) or geometry.shape!=(*shape,6,self.geometry_dim) or candidate_xy.shape!=(*shape,6,2) or eligible.shape!=shape:
            raise ValueError('Invalid selector shape')
        if eligible.dtype!=torch.bool:raise ValueError('Eligibility must be boolean')
        if not enabled:return dict(xy=baseline_xy,logits=None,choice=None,active=torch.zeros_like(eligible),eligible=torch.zeros_like(eligible))
        # Mask NaN-filled absent context before any neural arithmetic. A point
        # with any missing local/scalar evidence cannot override the baseline.
        valid=eligible & torch.isfinite(semantic).all(-1)&torch.isfinite(local).all((-1,-2))&torch.isfinite(scalars).all((-1,-2))&torch.isfinite(candidate_xy).all((-1,-2))
        if self.mode=='geometry':valid=valid&torch.isfinite(geometry).all((-1,-2))
        sem=torch.where(valid[...,None],semantic.detach(),0).float();loc=torch.where(valid[...,None,None],local.detach(),0).float()
        scalar=torch.where(valid[...,None,None],scalars.detach(),0).float()
        geo=torch.where(valid[...,None,None],geometry.detach(),0).float() if self.mode=='geometry' else torch.zeros_like(geometry,dtype=torch.float32)
        features=torch.cat([self.semantic(sem)[...,None,:].expand(*shape,6,32),self.local(loc),scalar,geo],-1)
        logits=self.score(features).squeeze(-1)
        # Initial residual is zero; a tiny deterministic baseline preference
        # avoids tie-based changes before any optimization.
        prior=torch.zeros(6,device=logits.device,dtype=logits.dtype);prior[0]=1e-3;logits=logits+prior
        choice=logits.argmax(-1);chosen=torch.gather(candidate_xy,-2,choice[...,None,None].expand(*shape,1,2)).squeeze(-2)
        xy=torch.where((valid&(choice!=0))[...,None],chosen,baseline_xy)
        return dict(xy=xy,logits=logits,choice=choice,active=valid&(choice!=0),eligible=valid)

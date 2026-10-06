"""Equal-capacity residual with selectable base or spatial-anchor output origin."""
import torch
from torch import nn
from torch.nn import functional as F


def reference_from_shape(shape):
    if shape.shape[-1]!=16 or not torch.isfinite(shape).all():raise ValueError('Expected finite mask shape features')
    s=shape.detach();box=s[...,4:8]
    fallback=box.new_tensor([.5,.5,1.,1.]).expand_as(box)
    box=torch.where((s[...,8:9]>0)&(box[...,2:].amin(-1,keepdim=True)>0),box,fallback)
    return torch.cat([box[...,:2],box[...,2:].clamp_min(1e-3)],-1)


class OriginCoordinateResidual(nn.Module):
    def __init__(self,token_dim=256,hidden_dim=32,*,relative=True,anchor_origin=False):
        super().__init__();self.relative=relative;self.token_dim=token_dim;self.anchor_origin=anchor_origin
        self.network=nn.Sequential(nn.Linear(token_dim+9,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,coarse,anchor,tokens,quality,reference):
        if coarse.shape!=anchor.shape or coarse.shape[-1]!=2:raise ValueError('Expected paired NQK2 coordinates')
        if tokens.shape!=(*coarse.shape[:-1],self.token_dim) or quality.shape!=(*coarse.shape[:-1],5):raise ValueError('Token/quality mismatch')
        if reference.shape!=(*coarse.shape[:-2],4):raise ValueError('Expected per-instance NQ4 reference')
        base=coarse.detach();spatial=anchor.detach();ref=reference.detach()
        if self.relative:
            center=ref[...,:2].unsqueeze(-2);scale=ref[...,2:].unsqueeze(-2)
        else:
            center=torch.full_like(ref[...,:2].unsqueeze(-2),.5);scale=torch.ones_like(center)
        if not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Positive finite extent required')
        features=torch.cat([F.normalize(tokens.detach(),dim=-1,eps=1e-6),(base-center)/scale,(spatial-center)/scale,quality.detach()],-1)
        delta=2*self.network(features).tanh()
        origin=spatial if self.anchor_origin else base
        return (origin+scale*delta).clamp(0,1)



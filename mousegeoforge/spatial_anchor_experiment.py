"""Opt-in coarse spatial-evidence experiment; original source remains untouched.

Install only after restoring the original checkpoint. No automatic production
integration; existing mask-box candidate hashes no longer pair with this model.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from .spatial_heatmap_probe import spatial_expectation


class SpatialAnchorResidual(nn.Module):
    def __init__(self,dim=256,keypoints=23,projection_dim=64):
        super().__init__()
        self.query=nn.Linear(dim,projection_dim)
        self.pixel=nn.Conv2d(dim,projection_dim,1)
        self.log_scale=nn.Parameter(torch.tensor(math.log(4.)))
        self.gate=nn.Parameter(torch.zeros(keypoints))

    def forward(self,raw_logits,tokens=None,pixels=None,mask_logits=None,*,enabled=False,use_mask_prior=True):
        if not enabled:return raw_logits
        if raw_logits.ndim!=4 or raw_logits.shape[-2:]!=(len(self.gate),2):raise ValueError('Expected N,Q,K,2 raw coordinates')
        if tokens is None or tokens.shape[:-1]!=raw_logits.shape[:-1]:raise ValueError('Token geometry mismatch')
        if pixels is None or pixels.ndim!=4 or pixels.shape[0]!=raw_logits.shape[0]:raise ValueError('Pixel geometry mismatch')
        if mask_logits is None or mask_logits.shape!=(*raw_logits.shape[:2],*pixels.shape[-2:]):raise ValueError('Mask geometry mismatch')
        # Frozen upstream evidence. Train only these projections/gates; no mask
        # loss may quietly turn the predicted mask into a convenient pseudo GT.
        query=F.normalize(self.query(tokens.detach()),dim=-1,eps=1e-6)
        pixel=F.normalize(self.pixel(pixels.detach()),dim=1,eps=1e-6)
        scores=torch.einsum('nqkd,ndhw->nqkhw',query,pixel)*self.log_scale.clamp(0.,math.log(32.)).exp()
        # A finite soft prior, not a hard foreground constraint: invisible points
        # may remain outside visible mask support. This is an experimental bias.
        if use_mask_prior:
            scores=scores+F.logsigmoid(mask_logits.detach()).clamp_min(-8.)[:,:,None]
        anchor=spatial_expectation(raw_logits,scores,enabled=True).clamp(1e-5,1-1e-5)
        target=torch.logit(anchor)
        # Zero initialization is exact and retains gate gradients. The residual
        # acts BEFORE local sampling, so reported coarse XY and heatmap centers agree.
        return raw_logits+self.gate.tanh()[None,None,:,None]*(target-raw_logits.detach())


def install_spatial_anchor(model):
    head=model.instance_pose_head
    if hasattr(head,'spatial_anchor'):raise ValueError('Already installed')
    with torch.random.fork_rng(devices=[]):
        branch=SpatialAnchorResidual(head.hidden_dim,head.num_keypoints)
    branch.to(next(head.parameters()))
    head.add_module('spatial_anchor',branch)
    return model


@contextmanager
def spatial_anchor_context(model,*,enabled=False):
    if not enabled:
        yield
        return
    head=model.instance_pose_head
    if not hasattr(head,'spatial_anchor'):raise ValueError('Install experimental branch first')
    if getattr(head,'_spatial_anchor_active',False):raise RuntimeError('Non-reentrant spatial context')
    head._spatial_anchor_active=True
    captured={};handles=[]
    def pixels(module,args,value):captured['pixels']=value
    def masks(module,args,value):captured['mask_embedding']=value
    def coarse(module,args,value):
        if not {'pixels','mask_embedding'}<=captured.keys():raise RuntimeError('Spatial evidence not captured')
        feature=captured['pixels']
        mask=torch.einsum('nqd,ndhw->nqhw',captured['mask_embedding'].detach(),feature.detach())
        return head.spatial_anchor(value,args[0],feature,mask,enabled=True)
    try:
        handles=[head.pyramid_refine[0].register_forward_hook(pixels),
                 head.mask_embedding.register_forward_hook(masks),
                 head.coarse_keypoint_xy.register_forward_hook(coarse)]
        yield
    finally:
        for handle in handles:handle.remove()
        captured.clear();head._spatial_anchor_active=False

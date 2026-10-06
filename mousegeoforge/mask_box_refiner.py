"""Isolated box residual conditioned on frozen instance tokens and mask shape.

No GT, instance count or camera role is accepted. Complete and largest-component
boxes are both features, never hard replacements. The feature extractor is a
CPU diagnostic prototype, not a latency-optimized production implementation.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.mask_shape_audit import mask_shape_statistics

MASK_FEATURE_DIM=16


def binary_mask_features(mask):
    mask=np.asarray(mask,dtype=bool);s=mask_shape_statistics(mask);yy,xx=np.nonzero(mask);h,w=mask.shape
    center=[float((xx+.5).mean()/w),float((yy+.5).mean()/h)] if len(xx) else [0.,0.]
    features=s['full_box_cxcywh']+s['largest_component_box_cxcywh']+[
        s['area']/(h*w),s['largest_component_fraction'],s['fill_fraction'],
        s['largest_to_full_box_area'],float(s['touches_boundary']),s['components']/(s['components']+1),*center]
    assert len(features)==MASK_FEATURE_DIM
    return np.asarray(features,dtype=np.float32)


@torch.no_grad()
def features_from_logits(logits,image_hw,*,corner_noise=False):
    if logits.ndim<3 or not torch.isfinite(logits).all():raise ValueError('Expected finite floating mask logits')
    flat=logits.detach().float().cpu().reshape(-1,1,*logits.shape[-2:])
    if len(flat)==0:return torch.empty((*logits.shape[:-2],MASK_FEATURE_DIM))
    masks=(F.interpolate(flat,size=tuple(image_hw),mode='bilinear',align_corners=False)[:,0]>0).numpy()
    features=[]
    for mask in masks:
        if corner_noise:
            # One fixed pixel in every corner; never inspect GT or tune by dev errors.
            mask=mask.copy();mask[0,0]=mask[0,-1]=mask[-1,0]=mask[-1,-1]=True
        features.append(binary_mask_features(mask))
    return torch.tensor(np.stack(features)).reshape(*logits.shape[:-2],MASK_FEATURE_DIM)


class MaskConditionedBoxRefiner(nn.Module):
    def __init__(self,token_dim=256,hidden_dim=128):
        super().__init__();self.token_dim=token_dim
        self.input=nn.Linear(token_dim+4+MASK_FEATURE_DIM,hidden_dim)
        self.output=nn.Linear(hidden_dim,4)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,boxes,tokens=None,mask_features=None,*,enabled=False,use_mask=False):
        if not enabled:return boxes
        if tokens is None or tokens.shape!=(*boxes.shape[:-1],self.token_dim):raise ValueError('Mismatched instance tokens')
        if boxes.shape[-1]!=4 or not torch.isfinite(boxes).all() or not torch.isfinite(tokens).all():raise ValueError('Invalid boxes/tokens')
        if use_mask:
            if mask_features is None or mask_features.shape!=(*boxes.shape[:-1],MASK_FEATURE_DIM) or not torch.isfinite(mask_features).all():raise ValueError('Invalid mask features')
            shape=mask_features.detach().to(boxes)
        else:shape=boxes.new_zeros((*boxes.shape[:-1],MASK_FEATURE_DIM))
        features=torch.cat((F.normalize(tokens.detach().to(boxes),dim=-1,eps=1e-6),boxes.detach(),shape),dim=-1)
        delta=self.output(F.gelu(self.input(features)))
        # Exactly equal at zero initialization. No sigmoid/logit roundtrip drift.
        center=(boxes[...,:2]+boxes[...,2:]*delta[...,:2].tanh()).clamp(0.,1.)
        size=(boxes[...,2:]*delta[...,2:].clamp(-2.,2.).exp()).clamp(1e-4,1.)
        return torch.cat((center,size),dim=-1)

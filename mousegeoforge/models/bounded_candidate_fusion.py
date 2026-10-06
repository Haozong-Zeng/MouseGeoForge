"""Experimental inference-only combination; production defaults remain off.

Independent V1 probabilities/tokens must be extracted upstream. Unique confident
queries are a restricted single-subject association heuristic, not multi-mouse
identity. This module never consumes labels, counts, matches or GT masks.
"""
import numpy as np
import torch
from torch import nn
from ..candidate_selector_features import build_selector_inputs,KP


class BoundedCandidateFusion(nn.Module):
    def __init__(self,local_head,geometry_head):
        super().__init__()
        if local_head.mode!='unary' or geometry_head.mode!='geometry':
            raise ValueError('Expected unary and geometry heads')
        self.local_head=local_head.eval().requires_grad_(False)
        self.geometry_head=geometry_head.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,predictions,projections=None,physical_views=None,pixels=None,
                *,local_enabled=False,fusion_enabled=False,view_valid=None):
        original=predictions['xy']
        if not isinstance(original,torch.Tensor) or original.ndim!=4 or original.shape[1:]!=(6,23,2):
            raise ValueError('Expected original XY[V,6,23,2] tensor')
        v=len(original);empty=torch.zeros(v,6,5,dtype=torch.bool,device=original.device)
        if not local_enabled and not fusion_enabled:
            return dict(xy=original,local_eligible=empty,fusion_eligible=empty,changed=empty)
        if pixels is None:raise ValueError('Actual input pixel sizes required')
        available=torch.ones(v,dtype=torch.bool) if view_valid is None else torch.as_tensor(view_valid,dtype=torch.bool).cpu()
        if available.shape!=(v,):raise ValueError('View validity must be [V]')
        # Missing camera calibration disables only fusion, not independent RGB.
        calibrated=projections is not None
        if projections is None:projections=np.zeros((v,3,4))
        if physical_views is None:
            if calibrated:raise ValueError('Physical camera IDs required for calibrated fusion')
            physical_views=list(range(v))
        features=build_selector_inputs(predictions,projections,physical_views,pixels,view_valid=available.numpy() if calibrated else np.zeros(v,bool))
        device=next(self.local_head.parameters()).device
        if next(self.geometry_head.parameters()).device!=device:raise ValueError('Head devices must match')
        features={k:x.to(device) for k,x in features.items()}
        xy=original.clone();local_eligible=empty.clone();fusion_eligible=empty.clone()
        if local_enabled:
            logits=torch.as_tensor(predictions['objectness_logits'],device=device)
            confident=logits.sigmoid()>=.5
            unique=confident&(confident.sum(-1,keepdim=True)==1)&torch.isfinite(logits).all(-1,keepdim=True)&available.to(device)[:,None]
            local_inputs=dict(features,eligible=unique[...,None].expand(-1,-1,5),geometry=torch.zeros_like(features['geometry']))
            local=self.local_head(**local_inputs,enabled=True)
            xy[:,:,KP]=local['xy'].to(original.device);local_eligible=local['eligible'].to(original.device)
        if fusion_enabled and calibrated:
            fused=self.geometry_head(**features,enabled=True);fusion_eligible=fused['eligible'].to(original.device)
            # Choosing geometry candidate0 is a deliberate return to original,
            # not 'no update': override local even if fused.active is false.
            xy[:,:,KP]=torch.where(fusion_eligible[...,None],fused['xy'].to(original.device),xy[:,:,KP])
        return dict(xy=xy,local_eligible=local_eligible,fusion_eligible=fusion_eligible,
                    changed=(xy[:,:,KP]!=original[:,:,KP]).any(-1))

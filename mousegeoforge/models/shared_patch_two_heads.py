"""One live release patch encoder, supplied depth/mask and keypoint heads.

Head adapters must preserve the cloud heads' exact feature/normalisation APIs.
There is no final-token cache or camera-head training in this composition.
"""
from __future__ import annotations

import torch
from torch import nn


class SharedPatchTwoHeads(nn.Module):
    def __init__(self, encoder, depth_head, keypoint_head, *, mean, std,
                 tap_indices=(5,11,17,23), normalize_taps=True):
        super().__init__()
        if tuple(sorted(set(tap_indices))) != tuple(tap_indices) or not tap_indices:
            raise ValueError('Unique ascending tap indices required')
        if not hasattr(encoder,'get_intermediate_layers'):
            raise TypeError('Encoder must expose get_intermediate_layers')
        if min(tap_indices)<0 or max(tap_indices)>=len(encoder.blocks):
            raise ValueError('Tap index outside encoder blocks')
        self.encoder=encoder; self.depth_head=depth_head; self.keypoint_head=keypoint_head
        self.tap_indices=tuple(tap_indices); self.normalize_taps=normalize_taps
        self.register_buffer('image_mean',torch.as_tensor(mean,dtype=torch.float32).reshape(1,3,1,1))
        self.register_buffer('image_std',torch.as_tensor(std,dtype=torch.float32).reshape(1,3,1,1))
        if not torch.isfinite(self.image_mean).all() or not torch.isfinite(self.image_std).all() or torch.any(self.image_std<=0):
            raise ValueError('Invalid release image normalisation')

    def train(self,mode=True):
        super().train(mode)
        # As in the cloud encoder adaptation: deterministic forward behaviour,
        # with autograd enabled for selected trainable encoder parameters.
        self.encoder.eval()
        return self

    def forward(self,rgb,*,task='both'):
        if task not in ('both','depth','keypoints'):
            raise ValueError('task must be both/depth/keypoints')
        if rgb.ndim!=4 or rgb.shape[1]!=3 or not torch.is_floating_point(rgb):
            raise ValueError('Expected floating RGB [N,3,H,W] in the registered crop frame')
        patch=int(self.encoder.patch_size)
        if any(x%patch for x in rgb.shape[-2:]):
            raise ValueError('Image dimensions must be divisible by patch size')
        taps=self.encoder.get_intermediate_layers((rgb-self.image_mean)/self.image_std,
            n=self.tap_indices,reshape=True,norm=self.normalize_taps)
        if len(taps)!=len(self.tap_indices):
            raise RuntimeError('Encoder did not return all registered taps')
        out={}
        if task in ('both','depth'): out['depth']=self.depth_head(taps,rgb)
        if task in ('both','keypoints'): out['keypoints']=self.keypoint_head(taps[-1],rgb)
        return out

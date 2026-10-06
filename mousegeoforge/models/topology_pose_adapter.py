"""Opt-in zero-residual decoder-topology experiment, not GroupPose reproduction.

GroupPose uses two sequential group attentions; this cheaper hypothesis uses a
single union mask (same instance OR same semantic type), versus full attention.
Frozen baseline and all queries remain present. No cross-view identity implied.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def instance_semantic_mask(instances, keypoints, device=None):
    if instances<1 or keypoints<1:
        raise ValueError('Positive instance and keypoint counts required')
    index=torch.arange(instances*keypoints,device=device)
    same_instance=index[:,None]//keypoints==index[None,:]//keypoints
    same_semantic=index[:,None]%keypoints==index[None,:]%keypoints
    return ~(same_instance|same_semantic)  # True prohibits direct attention.


class TopologyPoseAdapter(nn.Module):
    def __init__(self, baseline, topology='structured_union'):
        super().__init__()
        if topology not in ['full','structured_union']:
            raise ValueError('Unknown topology')
        self.topology=topology
        self.baseline=copy.deepcopy(baseline).eval().requires_grad_(False)
        self.decoder=copy.deepcopy(baseline.keypoint_decoder).requires_grad_(True)
        self.residual=nn.Linear(256,256)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        # This pilot updates only nose and four feet. Back, abdomen and tail
        # coordinates remain on the frozen path, not newly supervised.
        update=torch.zeros(baseline.keypoint_queries.num_embeddings,1)
        update[[0,14,15,16,17]]=1
        self.register_buffer('update_mask',update,persistent=False)

    def train(self,mode=True):
        super().train(mode);self.baseline.eval();return self

    def forward(self,instance_tokens,native_memory,predicted_boxes,*,enabled=False):
        # Baseline handles validation of all6 queries and produces the exact
        # original output when disabled. It cannot be updated by this adapter.
        with torch.no_grad():original=self.baseline(instance_tokens,native_memory,predicted_boxes)
        if not enabled:
            return original
        n,q,_=instance_tokens.shape;k=original['tokens'].shape[-2]
        memory=native_memory.detach().float()
        seeds=instance_tokens.detach()[:,:,None]+self.baseline.keypoint_queries.weight[None,None]
        mask=instance_semantic_mask(q,k,memory.device) if self.topology=='structured_union' else None
        context=self.decoder(seeds.flatten(1,2),memory.flatten(2).transpose(1,2),tgt_mask=mask).reshape(n,q,k,256)
        tokens=original['tokens'].detach()+self.residual(context)*self.update_mask
        coarse=self.baseline.coarse_keypoint_xy(tokens).sigmoid()
        maps=F.interpolate(memory.half().float(),(48,48),mode='bilinear',align_corners=False)
        xy=self.baseline.spatial_readout(tokens.reshape(n*q,k,256),coarse.reshape(n*q,k,2),maps[:,None].expand(-1,q,-1,-1,-1).reshape(n*q,256,48,48),predicted_boxes=predicted_boxes.detach().reshape(n*q,4),detach_pose_tokens=False)
        return dict(xy=xy.reshape(n,q,k,2),coarse=coarse,tokens=tokens)

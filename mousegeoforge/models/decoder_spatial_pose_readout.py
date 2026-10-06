"""Offline/native-memory decoder-unfreezing experiment; no production factory."""
import torch
from torch import nn
from torch.nn import functional as F
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from .core23_spatial_readout import Core23SpatialReadout


class DecoderSpatialPoseReadout(nn.Module):
    def __init__(self,head=None):
        super().__init__()
        head=MultiInstanceMouseHead(num_keypoints=23) if head is None else head
        self.keypoint_decoder=head.keypoint_decoder
        self.keypoint_queries=head.keypoint_queries
        self.coarse_keypoint_xy=head.coarse_keypoint_xy
        self.spatial_readout=Core23SpatialReadout()
        self.keypoint_queries.requires_grad_(False)
        self.coarse_keypoint_xy.requires_grad_(False)
        self.set_decoder_trainable(False)

    def set_decoder_trainable(self,enabled):
        self.keypoint_decoder.requires_grad_(bool(enabled))

    def forward(self,instance_tokens,native_memory,predicted_boxes,*,allow_input_grad=False):
        if instance_tokens.ndim!=3 or instance_tokens.shape[1:]!=(6,256):raise ValueError('All6 instance queries required')
        n=instance_tokens.shape[0]
        if native_memory.ndim!=4 or native_memory.shape[:2]!=(n,256) or predicted_boxes.shape!=(n,6,4):raise ValueError('Invalid native memory or predicted boxes')
        # Opt-in experimental bridge; default leaves cached/native training
        # behavior unchanged and does not enable any encoder parameters itself.
        memory=(native_memory if allow_input_grad else native_memory.detach()).float()
        source_tokens=instance_tokens if allow_input_grad else instance_tokens.detach()
        seeds=source_tokens[:,:,None]+self.keypoint_queries.weight[None,None]
        # Do NOT select matched instances before this self-attention: each
        # image must retain all138 query tokens including unmatched queries.
        tokens=self.keypoint_decoder(seeds.flatten(1,2),memory.flatten(2).transpose(1,2)).reshape(n,6,23,256)
        coarse=self.coarse_keypoint_xy(tokens).sigmoid()
        maps=F.interpolate(memory.half().float(),(48,48),mode='bilinear',align_corners=False)
        expanded=maps[:,None].expand(-1,6,-1,-1,-1).reshape(n*6,256,48,48)
        xy=self.spatial_readout(tokens.reshape(n*6,23,256),coarse.reshape(n*6,23,2),expanded,
                               predicted_boxes=predicted_boxes.detach().reshape(n*6,4),detach_pose_tokens=False,detach_memory=not allow_input_grad)
        return dict(xy=xy.reshape(n,6,23,2),coarse=coarse,tokens=tokens)

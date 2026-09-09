"""Composition over VGGT-Omega: shared trainable adapter, optional depth bridge.

This module does not implement temporal attention or a surface-to-joint offset.
The two task heads remain present/supervised whether explicit coupling is on/off.
"""
from __future__ import annotations

from contextlib import nullcontext
import torch
from torch import nn
from .adapters import LocalDepthFeatureBridge, SharedTokenAdapter


class DepthPoseModel(nn.Module):
    def __init__(self, aggregator, dense_head, instance_pose_head, *, coupling='off',
                 bottleneck=64, freeze_encoder=True, detach_depth_features=True):
        super().__init__()
        if coupling not in ('off', 'local_depth_features'):
            raise ValueError('coupling must be off or local_depth_features')
        self.aggregator = aggregator
        self.dense_head = dense_head
        self.instance_pose_head = instance_pose_head
        self.coupling = coupling
        self.detach_depth_features = detach_depth_features
        dim = dense_head.norm.normalized_shape[0]
        layers = tuple(dense_head.intermediate_layer_idx)
        if layers != tuple(instance_pose_head.intermediate_layer_idx):
            raise ValueError('Depth and pose must consume the same cached layer set')
        self.shared_adapter = SharedTokenAdapter(dim, layers, bottleneck)
        self.depth_pose_bridge = LocalDepthFeatureBridge(
            instance_pose_head.hidden_dim, dense_head.proj.in_channels)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.aggregator.requires_grad_(False)

    @classmethod
    def from_spatial_model(cls, model, **kwargs):
        if model.dense_head is None or model.instance_pose_head is None:
            raise ValueError('Both depth and multi-instance heads must be enabled')
        return cls(model.aggregator, model.dense_head, model.instance_pose_head, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.aggregator.eval()
        return self

    def forward(self, images, *, coupling=None):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError('Expected [B,V,3,H,W]; time must not be flattened into views')
        if any(size % self.instance_pose_head.patch_size for size in images.shape[-2:]):
            raise ValueError('Image dimensions must be divisible by patch size')
        mode = self.coupling if coupling is None else coupling
        if mode not in ('off', 'local_depth_features'):
            raise ValueError('Unknown coupling mode')
        if self.freeze_encoder:
            self.aggregator.eval()
        amp = (torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
                             else torch.float16) if images.is_cuda else nullcontext())
        grad = torch.no_grad() if self.freeze_encoder else nullcontext()
        with grad, amp:
            cached, patch_start = self.aggregator(images)
        # Match upstream decoder fp32 path, even inside caller autocast.
        with torch.autocast(device_type=images.device.type, enabled=False):
            adapted = self.shared_adapter(cached, patch_start)
            depth, confidence, features = self.dense_head(
                adapted, images, patch_start, return_features=True)
            refiner = None
            if mode == 'local_depth_features':
                def refiner(**kwargs):
                    return self.depth_pose_bridge(
                        **kwargs, depth_features=features.flatten(0, 1),
                        detach_depth=self.detach_depth_features)
            pose = self.instance_pose_head(adapted, images, patch_start,
                                           keypoint_token_refiner=refiner)
        return {**pose, 'depth': depth, 'depth_conf': confidence}

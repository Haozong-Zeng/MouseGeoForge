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
        if images.ndim != 5 or images.shape[2] != 3 or min(images.shape[:2]) < 1:
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
        return self._decode(images, cached, patch_start, coupling=mode)

    def forward_ragged(self, image_sets, *, coupling=None):
        """Run independent [V,3,H,W] sets without fake-view padding.

        Each result has B=1 and its own real view count. This correctness-first
        path makes separate forwards; use equal-V/H/W buckets for throughput.
        Losses should be averaged per scene, not summed over its view count.
        """
        if not isinstance(image_sets, (list, tuple)) or not image_sets:
            raise ValueError('At least one image set is required')
        for images in image_sets:
            if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
                raise ValueError('Each image set must have shape [V,3,H,W], V>=1')
        return [self.forward(images.unsqueeze(0), coupling=coupling) for images in image_sets]

    def forward_cached(self, images, cached, patch_start, *, coupling=None):
        """Decode deterministic frozen-encoder features (no augmentation changes).

        This is an optimization for controlled spatial experiments, not a time
        cache. Callers must bind features to exact RGB, view order and encoder.
        """
        if not self.freeze_encoder:
            raise ValueError('Cached training requires a frozen encoder')
        return self._decode(images, cached, patch_start, coupling=coupling)

    def _decode(self, images, cached, patch_start, *, coupling=None):
        if images.ndim != 5 or images.shape[2] != 3 or min(images.shape[:2]) < 1:
            raise ValueError('Expected [B,V,3,H,W]')
        mode = self.coupling if coupling is None else coupling
        if mode not in ('off', 'local_depth_features'):
            raise ValueError('Unknown coupling mode')
        h, w = images.shape[-2:]
        patch = self.instance_pose_head.patch_size
        if h % patch or w % patch:
            raise ValueError('Image dimensions must be divisible by patch size')
        for i in self.dense_head.intermediate_layer_idx:
            x = cached[i]
            if x is None or x.shape[:3] != (*images.shape[:2], patch_start+(h//patch)*(w//patch)):
                raise ValueError('Cached feature geometry does not match RGB')
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

"""Small residual adapters; no temporal state and no surface/joint equality."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class SharedTokenAdapter(nn.Module):
    def __init__(self, dim: int, layers=(4, 11, 17, 23), bottleneck: int = 64):
        super().__init__()
        if bottleneck < 1:
            raise ValueError('bottleneck must be positive')
        self.blocks = nn.ModuleDict({str(i): nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, bottleneck), nn.GELU(), nn.Linear(bottleneck, dim)
        ) for i in layers})
        for block in self.blocks.values():
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def forward(self, cached, patch_start):
        result = list(cached)
        for key, block in self.blocks.items():
            i = int(key)
            if cached[i] is None:
                raise ValueError(f'Missing cached layer {i}')
            tokens = cached[i].float()
            patches = tokens[:, :, patch_start:]
            result[i] = torch.cat((tokens[:, :, :patch_start], patches + block(patches)), dim=2)
        return result


class LocalDepthFeatureBridge(nn.Module):
    """Each unordered query samples local depth features around its keypoints.

Predicted instance masks are detached soft priors, not guaranteed ownership.
The bridge consumes decoder features, not sensor/GT depth or skeleton labels.
Zero output initialization gives an exact initial residual identity. All-invalid
windows are explicit identity even after training; no all-masked softmax NaNs.
"""
    def __init__(self, pose_dim: int, depth_dim: int, sample_side=3, radius=0.10):
        super().__init__()
        if sample_side < 1 or not 0 < radius <= 1:
            raise ValueError('Invalid local sampling geometry')
        self.radius = radius
        self.sample_side = sample_side
        self.query = nn.Linear(pose_dim, pose_dim)
        self.key = nn.Linear(depth_dim, pose_dim)
        self.value = nn.Linear(depth_dim, pose_dim)
        self.output = nn.Linear(pose_dim, pose_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        axis = torch.linspace(-1, 1, sample_side) if sample_side > 1 else torch.zeros(1)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('offsets', torch.stack((xx, yy), -1).reshape(-1, 2), persistent=False)

    def forward(self, *, tokens, coarse_xy, mask_logits, depth_features, detach_depth=True):
        n, q, k, d = tokens.shape
        if depth_features.ndim != 4 or depth_features.shape[0] != n:
            raise ValueError('Depth features must be [B*V,C,H,W] in the same view order')
        if coarse_xy.shape != (n, q, k, 2) or mask_logits.shape[:2] != (n, q):
            raise ValueError('Query dimensions do not match')
        features = depth_features.detach() if detach_depth else depth_features
        finite_map = torch.isfinite(features).all(1, keepdim=True).float()
        features = torch.nan_to_num(features, nan=0., posinf=0., neginf=0.)
        valid_xy = torch.isfinite(coarse_xy).all(-1)
        xy = torch.nan_to_num(coarse_xy, nan=-2., posinf=-2., neginf=-2.)
        samples = xy[..., None, :] + self.offsets.to(xy) * self.radius
        valid = ((samples >= 0) & (samples <= 1)).all(-1) & valid_xy[..., None]
        grid = samples.mul(2).sub(1).reshape(n, q*k, -1, 2)
        local = F.grid_sample(features, grid, align_corners=False, padding_mode='border')
        local = local.permute(0, 2, 3, 1).reshape(n, q, k, -1, features.shape[1])
        observed = F.grid_sample(finite_map, grid, mode='nearest', align_corners=False,
                                 padding_mode='border').reshape(n, q, k, -1) > .5
        valid = valid & observed
        masks = torch.nan_to_num(mask_logits.detach(), nan=-20.).sigmoid()
        prior = F.grid_sample(masks.reshape(n*q, 1, *masks.shape[-2:]),
                              grid.reshape(n*q, k, -1, 2), align_corners=False,
                              padding_mode='zeros').reshape(n, q, k, -1)
        score = (self.query(tokens)[..., None, :] * self.key(local)).sum(-1) / math.sqrt(d)
        score = score + prior.clamp_min(.01).log()
        score = score.masked_fill(~valid, -1e4)
        weights = score.softmax(-1) * valid.to(score.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        summary = (weights[..., None] * self.value(local)).sum(-2)
        delta = self.output(summary) * valid.any(-1, keepdim=True)
        return tokens + delta

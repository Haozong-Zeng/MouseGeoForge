"""Small calibrated ray cost head for frozen image features, camera-Z in mm.

This is a learnability baseline, not a production MVS architecture. No target
mask, depth, keypoints or per-image GT depth bounds enter forward().
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def world_rays(xy, k, e):
    uv = torch.cat([xy, torch.ones_like(xy[:, :1])], -1)
    camera = uv @ torch.linalg.inv(k).T
    return -e[:3, 3] @ e[:3, :3], camera @ e[:3, :3]


def ray_bounds(xy, k, e, aabb):
    origin, directions = world_rays(xy, k, e)
    parallel = directions.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(directions), directions)
    first, second = (aabb[0] - origin) / safe, (aabb[1] - origin) / safe
    lower, upper = torch.minimum(first, second), torch.maximum(first, second)
    lower = torch.where(parallel, torch.full_like(lower, -torch.inf), lower)
    upper = torch.where(parallel, torch.full_like(upper, torch.inf), upper)
    near, far = lower.max(-1).values.clamp_min(1.), upper.min(-1).values.clamp_max(2000.)
    outside_parallel = (parallel & ((origin < aabb[0]) | (origin > aabb[1]))).any(-1)
    valid = (far > near) & ~outside_parallel
    far = torch.where(valid, far, near + 1.)
    return origin, directions, near, far, valid


def project_world(world, k, e):
    camera = world @ e[:3, :3].T + e[:3, 3]
    homogeneous = camera @ k.T
    return homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(1e-6), camera[..., 2]


class CalibratedRayDepth(nn.Module):
    def __init__(self, in_channels=1024, channels=32, planes=96):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.GroupNorm(4, channels), nn.GELU())
        self.score = nn.Sequential(nn.Linear(channels + 8, 64), nn.GELU(),
                                   nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.normal_(self.score[-1].weight, std=.001)
        nn.init.zeros_(self.score[-1].bias)
        self.log_temperature = nn.Parameter(torch.tensor(2.3))
        self.planes = planes

    def encode(self, features):
        return F.normalize(self.project(features.float()), dim=1)

    def forward(self, encoded, xy, reference, k, e, aabb, image_side=384, use_cost=True):
        origin, directions, near, far, valid = ray_bounds(xy, k[reference], e[reference], aabb)
        fractions = torch.linspace(0, 1, self.planes, device=xy.device)
        z = near[:, None] + (far - near)[:, None] * fractions
        world = origin + directions[:, None, :] * z[..., None]
        grid_ref = (2 * (xy + .5) / image_side - 1)[None, :, None]
        ref = F.grid_sample(encoded[reference:reference+1], grid_ref, align_corners=False)[0, :, :, 0].T
        costs, visibility = [], []
        for source in range(3):
            if source == reference:
                continue
            uv, source_z = project_world(world, k[source], e[source])
            grid = (2 * (uv + .5) / image_side - 1)[None]
            sampled = F.grid_sample(encoded[source:source+1], grid, align_corners=False)[0].permute(1, 2, 0)
            inside = (source_z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= image_side-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= image_side-1)
            cosine = (F.normalize(sampled, dim=-1) * F.normalize(ref, dim=-1)[:, None]).sum(-1)
            costs.append(cosine * inside if use_cost else torch.zeros_like(cosine))
            visibility.append(inside.float() if use_cost else torch.zeros_like(cosine))
        ref = ref[:, None].expand(-1, self.planes, -1)
        world_normal = 2 * (world - aabb[0]) / (aabb[1] - aabb[0]) - 1
        extras = torch.stack([*costs, *visibility, z/1000], -1)
        scores = self.score(torch.cat([ref, extras, world_normal], -1)).squeeze(-1)
        count = torch.stack(visibility).sum(0)
        matching = torch.stack(costs).sum(0) / count.clamp_min(1)
        scores = scores + self.log_temperature.exp().clamp_max(50) * matching
        probability = scores.softmax(-1)
        depth = (probability * z).sum(-1)
        confidence = probability.max(-1).values
        return depth, confidence, valid


def supervised_loss(prediction, target, valid):
    if not valid.any():
        raise ValueError('No supervised rays intersect the frozen capture volume')
    return (prediction[valid].clamp_min(1).log() - target[valid].clamp_min(1).log()).abs().mean()

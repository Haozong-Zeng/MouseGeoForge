"""Reliability-aware perspective triangulation, without GT or instance association.

Inputs are one already-associated instance/keypoint across cameras, in pixels.
The caller must associate instances first. A valid solve is not a guarantee of
anatomical correctness: two views cannot identify an arbitrary wrong observation.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import torch


@dataclass
class TriangulationResult:
    xyz: torch.Tensor
    supported: bool
    reason: str
    inliers: torch.Tensor
    reprojection_px: torch.Tensor
    max_ray_angle_deg: float = 0.0


def project(projections, xyz):
    q = projections[:, :3, :3] @ xyz + projections[:, :3, 3]
    z = q[:, 2]
    safe_z = torch.where(z.abs() > 1e-9, z, torch.full_like(z, 1e-9))
    return q[:, :2] / safe_z[:, None], z


def _linear(xy, projections, weights):
    rows = xy[:, :, None] * projections[:, None, 2, :] - projections[:, :2, :]
    design = (rows[:, :, :3] * weights.sqrt()[..., None]).reshape(-1, 3)
    rhs = (-rows[:, :, 3] * weights.sqrt()).flatten()
    normal = design.T @ design
    damping = normal.diagonal().mean().detach().clamp_min(1e-12) * 1e-10
    return torch.linalg.solve(normal + torch.eye(3, device=normal.device, dtype=normal.dtype)*damping,
                              design.T @ rhs)


def triangulate_robust(xy, projections, valid=None, *, sigma_px=None,
                       uncertainty_calibrated=False, robust=True,
                       reprojection_threshold_px=8., min_ray_angle_deg=1., iterations=5,
                       huber_delta_px=None):
    """Pair consensus + detached IRLS/Gauss-Newton with cheirality and angle gates.

    sigma_px is a per-axis standard deviation [V,2], NOT visibility probability.
    It is rejected unless the caller explicitly attests to a calibrated model.
    Weights and discrete selection are detached; coordinates retain gradients.
    Unsupported xyz is NaN, never a fabricated zero or silently prior-filled pose.
    huber_delta_px=None preserves the legacy 3-sigma Huber kernel. A positive
    value opts into a pixel-space kernel independent of sigma; sigma then only
    supplies measurement precision. The 1-pixel sigma floor still applies.
    When resizing images, scale pixel thresholds and sigma with the coordinates.
    """
    if xy.ndim != 2 or xy.shape[-1] != 2 or projections.shape != (len(xy), 3, 4):
        raise ValueError('Expected xy [V,2], projections [V,3,4]')
    if reprojection_threshold_px <= 0 or min_ray_angle_deg < 0:
        raise ValueError('Invalid triangulation thresholds')
    if huber_delta_px is not None and (not math.isfinite(huber_delta_px) or huber_delta_px <= 0):
        raise ValueError('huber_delta_px must be finite and positive, or None for legacy')
    if sigma_px is not None and not uncertainty_calibrated:
        raise ValueError('Uncalibrated uncertainty must not drive triangulation')
    dtype = torch.float64 if xy.dtype == torch.float64 else torch.float32
    xy, projections = xy.to(dtype), projections.to(dtype)
    valid = torch.ones(len(xy), dtype=torch.bool, device=xy.device) if valid is None else valid.bool().clone()
    valid &= torch.isfinite(xy).all(-1) & torch.isfinite(projections).all((-2, -1))
    sigma = torch.ones_like(xy) if sigma_px is None else sigma_px.detach().to(xy)
    if sigma.shape != xy.shape:
        raise ValueError('sigma_px must have shape [V,2]')
    valid &= torch.isfinite(sigma).all(-1) & (sigma > 0).all(-1)
    sigma = sigma.clamp_min(1.)
    empty = torch.zeros_like(valid)
    def failure(reason):
        return TriangulationResult(xy.new_full((3,), float('nan')), False, reason,
                                   empty, xy.new_full((len(xy),), float('nan')))
    idx = torch.where(valid)[0]
    if len(idx) < 2:
        return failure('insufficient_views')
    u, p, s = xy[idx], projections[idx], sigma[idx]
    if (torch.linalg.det(p[:, :, :3]).abs() < 1e-12).any():
        return failure('singular_camera')
    weights = s.reciprocal().square()
    xyz = _linear(u, p, weights)
    # More than two observations supply redundancy. Never claim two-view RANSAC.
    if robust and len(idx) >= 3:
        best_score, best_mask = None, None
        for pair in combinations(range(len(idx)), 2):
            select = list(pair)
            candidate = _linear(u[select], p[select], weights[select])
            pred, z = project(p, candidate)
            error = (pred-u).norm(dim=-1)
            mask = (error.detach() <= reprojection_threshold_px) & (z.detach() > 0)
            score = (int(mask.sum()), -float(error.detach().clamp_max(reprojection_threshold_px).sum()))
            if best_score is None or score > best_score:
                best_score, best_mask = score, mask
        if best_mask is not None and int(best_mask.sum()) >= 2:
            weights = weights * best_mask[:, None]
            xyz = _linear(u, p, weights)
    if not torch.isfinite(xyz).all():
        return failure('nonfinite_solve')
    for _ in range(iterations if robust else 0):
        pred, z = project(p, xyz)
        if (z.abs() < 1e-8).any():
            return failure('camera_plane')
        residual = pred-u
        if huber_delta_px is None:
            standardized = (residual.detach()/s).norm(dim=-1)
            huber = (3./standardized.clamp_min(1e-6)).clamp_max(1.)
        else:
            error_px = residual.detach().norm(dim=-1)
            huber = (huber_delta_px/error_px.clamp_min(1e-6)).clamp_max(1.)
        w = weights * huber[:, None]
        jac = (p[:, :2, :3] - pred[:, :, None]*p[:, None, 2, :3]) / z[:, None, None]
        j = (jac*w.sqrt()[..., None]).reshape(-1, 3)
        r = (residual*w.sqrt()).flatten()
        normal = j.T@j
        damp = normal.diagonal().mean().detach().clamp_min(1e-12)*1e-6
        delta = torch.linalg.solve(normal+torch.eye(3, device=xy.device, dtype=dtype)*damp, j.T@r)
        xyz = xyz-delta
    pred, z = project(p, xyz)
    err = (pred-u).norm(dim=-1)
    inlier = (err.detach() <= reprojection_threshold_px) & (z.detach() > 0) & (weights.sum(-1)>0)
    if int(inlier.sum()) < 2:
        return failure('reprojection_or_cheirality')
    # Acute line angle: antiparallel rays are also degenerate.
    rays = torch.linalg.solve(p[:, :, :3], torch.cat((u, torch.ones_like(u[:, :1])), -1)[..., None]).squeeze(-1)
    rays = torch.nn.functional.normalize(rays.detach()[inlier], dim=-1)
    cos = (rays@rays.T).abs().clamp(0., 1.)
    angle = float(torch.rad2deg(torch.acos(cos)).max())
    if angle < min_ray_angle_deg:
        return failure('degenerate_ray_angle')
    all_inliers = empty.clone(); all_inliers[idx] = inlier
    all_errors = xy.new_full((len(xy),), float('nan')); all_errors[idx] = err.detach()
    reason = 'two_view_supported_no_outlier_redundancy' if len(idx) == 2 else 'multiview_consensus'
    return TriangulationResult(xyz, True, reason, all_inliers, all_errors, angle)

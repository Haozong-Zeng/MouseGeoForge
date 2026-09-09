"""Losses and differentiable multiview fusion for the VGGT-Omega keypoint head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from training.tail_curve_losses import (
    TailCurveLossConfig,
    strict_point_weights,
    tail_curve_loss,
)
from vggt_omega.utils.pose_enc import encoding_to_camera


def heatmap_soft_argmax(
    logits: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-resolution xy means and peak probabilities."""
    xy, peak, _ = heatmap_soft_argmax_with_covariance(logits, image_size)
    return xy, peak


def heatmap_soft_argmax_with_covariance(
    logits: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full-resolution means, peak probabilities, and 2D covariance."""
    batch, views, keypoints, height, width = logits.shape
    probabilities = logits.reshape(batch, views, keypoints, -1).softmax(dim=-1)
    dtype = probabilities.dtype
    device = probabilities.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    x_mean = (probabilities * x_grid.reshape(-1)).sum(dim=-1)
    y_mean = (probabilities * y_grid.reshape(-1)).sum(dim=-1)
    image_height, image_width = image_size
    x_mean = (x_mean + 0.5) * (image_width / float(width)) - 0.5
    y_mean = (y_mean + 0.5) * (image_height / float(height)) - 0.5
    xy = torch.stack([x_mean, y_mean], dim=-1)
    full_x = (x_grid + 0.5) * (image_width / float(width)) - 0.5
    full_y = (y_grid + 0.5) * (image_height / float(height)) - 0.5
    centered_x = full_x.reshape(-1) - x_mean[..., None]
    centered_y = full_y.reshape(-1) - y_mean[..., None]
    covariance_xx = (probabilities * centered_x.square()).sum(dim=-1)
    covariance_yy = (probabilities * centered_y.square()).sum(dim=-1)
    covariance_xy = (probabilities * centered_x * centered_y).sum(dim=-1)
    covariance = torch.stack(
        [
            covariance_xx,
            covariance_xy,
            covariance_xy,
            covariance_yy,
        ],
        dim=-1,
    ).reshape(batch, views, keypoints, 2, 2)
    return xy, probabilities.max(dim=-1).values, covariance


def gaussian_heatmap_cross_entropy(
    logits: torch.Tensor,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
    image_size: tuple[int, int],
    sigma_pixels: float,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy against normalized 2D Gaussian targets."""
    _, _, _, height, width = logits.shape
    image_height, image_width = image_size
    dtype = logits.dtype
    device = logits.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    target_x = (target_xy[..., 0] + 0.5) * (width / float(image_width)) - 0.5
    target_y = (target_xy[..., 1] + 0.5) * (height / float(image_height)) - 0.5
    sigma_x = max(float(sigma_pixels) * width / float(image_width), 0.5)
    sigma_y = max(float(sigma_pixels) * height / float(image_height), 0.5)
    squared_distance = (
        (x_grid - target_x[..., None, None]).square() / (sigma_x * sigma_x)
        + (y_grid - target_y[..., None, None]).square() / (sigma_y * sigma_y)
    )
    target = torch.exp(-0.5 * squared_distance)
    target = target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    log_probabilities = logits.reshape(*logits.shape[:3], -1).log_softmax(dim=-1)
    log_probabilities = log_probabilities.view_as(logits)
    cross_entropy = -(target * log_probabilities).sum(dim=(-2, -1))
    if valid.any():
        weights = valid.to(dtype=logits.dtype)
        if point_weights is not None:
            weights = weights * point_weights.to(dtype=logits.dtype)[None, None, :]
        return (cross_entropy * weights).sum() / weights.sum().clamp_min(1.0)
    return logits.mean() * 0.0


def _weighted_point_smooth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    point_weights: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    per_point = F.smooth_l1_loss(
        predicted,
        target,
        reduction="none",
        beta=beta,
    ).mean(dim=-1)
    weights = valid.to(dtype=predicted.dtype) * point_weights[None, :]
    return (per_point * weights).sum() / weights.sum().clamp_min(1.0)


def triangulate_gaussian_rays(
    xy: torch.Tensor,
    visibility_logits: torch.Tensor,
    pixel_sigma: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    view_valid: torch.Tensor,
    pixel_covariance: torch.Tensor | None = None,
    robust_iterations: int = 2,
    robust_delta: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Gaussian ray tubes with uncertainty and robust view reweighting."""
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    fx = intrinsics[..., 0, 0, None]
    fy = intrinsics[..., 1, 1, None]
    cx = intrinsics[..., 0, 2, None]
    cy = intrinsics[..., 1, 2, None]

    camera_directions = torch.stack(
        [
            (xy[..., 0] - cx) / fx.clamp_min(1e-6),
            (xy[..., 1] - cy) / fy.clamp_min(1e-6),
            torch.ones_like(xy[..., 0]),
        ],
        dim=-1,
    )
    world_directions = torch.einsum(
        "bvij,bvkj->bvki",
        rotation.transpose(-1, -2),
        camera_directions,
    )
    world_directions = F.normalize(world_directions, dim=-1, eps=1e-8)
    camera_centers = -torch.einsum(
        "bvij,bvj->bvi",
        rotation.transpose(-1, -2),
        translation,
    )

    identity = torch.eye(3, dtype=xy.dtype, device=xy.device)
    ray_projector = identity - (
        world_directions[..., :, None] * world_directions[..., None, :]
    )
    focal = torch.sqrt((fx * fy).clamp_min(1e-6))
    effective_pixel_sigma = pixel_sigma
    if pixel_covariance is not None:
        heatmap_variance = 0.5 * pixel_covariance.diagonal(
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1)
        effective_pixel_sigma = torch.sqrt(
            pixel_sigma.square() + heatmap_variance.clamp_min(0.0)
        )
    angular_sigma = (effective_pixel_sigma / focal).clamp(min=1e-4, max=0.5)
    base_weights = torch.sigmoid(visibility_logits) / angular_sigma.square()
    base_weights = base_weights * view_valid.to(dtype=base_weights.dtype)
    weights = base_weights

    def solve(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        precision = (weight[..., None, None] * ray_projector).sum(dim=1)
        right_hand = torch.einsum(
            "bvkij,bvi,bvk->bkj",
            ray_projector,
            camera_centers,
            weight,
        )
        diagonal_scale = precision.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        regularizer = (diagonal_scale * 1e-6).clamp_min(1e-6)
        regularized_precision = (
            precision + regularizer[..., None, None] * identity
        )
        solved_center = torch.linalg.solve(
            regularized_precision,
            right_hand[..., None],
        ).squeeze(-1)
        return solved_center, torch.linalg.inv(regularized_precision)

    center, covariance = solve(weights)
    valid_views = view_valid.sum(dim=1)
    for _ in range(max(int(robust_iterations), 0)):
        camera_points = torch.einsum(
            "bvij,bkj->bvki",
            rotation,
            center,
        ) + translation[:, :, None, :]
        homogeneous = torch.einsum(
            "bvij,bvkj->bvki",
            intrinsics,
            camera_points,
        )
        projected = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-6)
        residual = torch.linalg.vector_norm(projected - xy, dim=-1)
        normalized = residual / effective_pixel_sigma.clamp_min(1.0)
        robust_weight = 1.0 / (
            1.0 + (normalized / max(float(robust_delta), 1e-3)).square()
        )
        # Two-view triangulation cannot identify which ray is the outlier.
        robust_weight = torch.where(
            (valid_views >= 3)[:, None, :],
            robust_weight.detach(),
            torch.ones_like(robust_weight),
        )
        weights = base_weights * robust_weight
        center, covariance = solve(weights)
    return center, covariance, valid_views


def keypoint_head_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    target_xy: torch.Tensor,
    target_world: torch.Tensor,
    point_valid: torch.Tensor,
    view_valid: torch.Tensor,
    visibility_target: torch.Tensor,
    camera_available: torch.Tensor,
    heatmap_sigma: float,
    heatmap_weight: float,
    visibility_weight: float,
    center3d_weight: float,
    predicted_rig_weight: float,
    fixed_rig_weight: float,
    keypoint_names: list[str] | tuple[str, ...] | None = None,
    surface_visible: torch.Tensor | None = None,
    tail_curve_config: TailCurveLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = predictions["keypoint_heatmap_logits"]
    visibility_logits = predictions["keypoint_visibility_logits"]
    pixel_sigma = predictions["keypoint_pixel_sigma"]
    image_size = batch["images"].shape[-2:]
    xy, peak_probability, heatmap_covariance = (
        heatmap_soft_argmax_with_covariance(logits, image_size)
    )

    point_loss_weights = torch.ones(
        logits.shape[2],
        dtype=logits.dtype,
        device=logits.device,
    )
    if keypoint_names is not None and tail_curve_config is not None:
        point_loss_weights = strict_point_weights(
            keypoint_names,
            tail_curve_config.interior_point_weight,
            dtype=logits.dtype,
            device=logits.device,
        )

    heatmap_valid = view_valid & point_valid[:, None, :]
    heatmap_loss = gaussian_heatmap_cross_entropy(
        logits,
        target_xy,
        heatmap_valid,
        image_size,
        heatmap_sigma,
        point_loss_weights,
    )
    visibility_supervised = point_valid[:, None, :] & camera_available[:, :, None]
    if visibility_supervised.any():
        visibility_loss = F.binary_cross_entropy_with_logits(
            visibility_logits[visibility_supervised],
            visibility_target[visibility_supervised].to(dtype=visibility_logits.dtype),
        )
    else:
        visibility_loss = visibility_logits.mean() * 0.0

    pixel_error = torch.linalg.vector_norm(xy - target_xy, dim=-1)
    mean_pixel_error = (
        pixel_error[heatmap_valid].mean()
        if heatmap_valid.any()
        else pixel_error.mean() * 0.0
    )

    fusion_valid = heatmap_valid & camera_available[:, :, None]
    fixed_center, fixed_covariance, fixed_view_count = triangulate_gaussian_rays(
        xy,
        visibility_logits,
        pixel_sigma,
        batch["extrinsics"],
        batch["intrinsics"],
        fusion_valid,
        pixel_covariance=heatmap_covariance,
    )
    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        image_size,
        build_intrinsics=True,
    )
    predicted_center, predicted_covariance, predicted_view_count = triangulate_gaussian_rays(
        xy,
        visibility_logits,
        pixel_sigma,
        pred_extrinsics,
        pred_intrinsics,
        fusion_valid,
        pixel_covariance=heatmap_covariance,
    )
    center_valid = point_valid & (fixed_view_count >= 2) & (predicted_view_count >= 2)
    fixed_residual = torch.linalg.vector_norm(fixed_center - target_world, dim=-1)
    predicted_residual = torch.linalg.vector_norm(predicted_center - target_world, dim=-1)
    if center_valid.any():
        fixed_center_loss = _weighted_point_smooth_l1(
            fixed_center,
            target_world,
            center_valid,
            point_loss_weights,
            beta=0.01,
        )
        predicted_center_loss = _weighted_point_smooth_l1(
            predicted_center,
            target_world,
            center_valid,
            point_loss_weights,
            beta=0.01,
        )
        fixed_mae = fixed_residual[center_valid].mean()
        predicted_mae = predicted_residual[center_valid].mean()
    else:
        fixed_center_loss = fixed_center.mean() * 0.0
        predicted_center_loss = predicted_center.mean() * 0.0
        fixed_mae = fixed_residual.mean() * 0.0
        predicted_mae = predicted_residual.mean() * 0.0

    center_loss = (
        fixed_rig_weight * fixed_center_loss
        + predicted_rig_weight * predicted_center_loss
    )
    tail_loss = logits.mean() * 0.0
    tail_logs: dict[str, torch.Tensor] = {}
    if keypoint_names is not None and tail_curve_config is not None:
        tail_loss, tail_logs = tail_curve_loss(
            keypoint_names=keypoint_names,
            predicted_xy=xy,
            target_xy=target_xy,
            visibility_logits=visibility_logits,
            pixel_sigma=pixel_sigma,
            fixed_center=fixed_center,
            predicted_center=predicted_center,
            target_world=target_world,
            point_valid=point_valid,
            view_valid=view_valid,
            surface_visible=(
                surface_visible if surface_visible is not None else view_valid
            ),
            fixed_view_count=fixed_view_count,
            predicted_view_count=predicted_view_count,
            image_size=image_size,
            fixed_rig_weight=fixed_rig_weight,
            predicted_rig_weight=predicted_rig_weight,
            config=tail_curve_config,
        )
    total = (
        heatmap_weight * heatmap_loss
        + visibility_weight * visibility_loss
        + center3d_weight * center_loss
        + (
            tail_curve_config.weight * tail_loss
            if tail_curve_config is not None
            else tail_loss
        )
    )
    logs = {
        "loss_keypoint_head": total,
        "loss_keypoint_heatmap": heatmap_loss,
        "loss_keypoint_visibility": visibility_loss,
        "loss_keypoint_center3d": center_loss,
        "loss_keypoint_center3d_fixed": fixed_center_loss,
        "loss_keypoint_center3d_predicted": predicted_center_loss,
        "keypoint_head_pixel_error": mean_pixel_error,
        "keypoint_head_fixed_mae": fixed_mae,
        "keypoint_head_predicted_mae": predicted_mae,
        "keypoint_head_peak_probability": (
            peak_probability[heatmap_valid].mean()
            if heatmap_valid.any()
            else peak_probability.mean() * 0.0
        ),
        "keypoint_head_pixel_sigma": (
            pixel_sigma[heatmap_valid].mean()
            if heatmap_valid.any()
            else pixel_sigma.mean()
        ),
        "keypoint_head_heatmap_cov_trace": (
            heatmap_covariance.diagonal(dim1=-2, dim2=-1)
            .sum(dim=-1)[heatmap_valid]
            .mean()
            if heatmap_valid.any()
            else heatmap_covariance.mean() * 0.0
        ),
        "keypoint_head_3d_points": center_valid.sum().to(dtype=logits.dtype),
        "keypoint_head_fixed_cov_trace": (
            fixed_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[center_valid].mean()
            if center_valid.any()
            else fixed_covariance.mean() * 0.0
        ),
        "keypoint_head_predicted_cov_trace": (
            predicted_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[center_valid].mean()
            if center_valid.any()
            else predicted_covariance.mean() * 0.0
        ),
        "keypoint_head_tail_interior_point_weight": logits.new_tensor(
            tail_curve_config.interior_point_weight
            if tail_curve_config is not None
            else 1.0
        ),
    }
    logs.update(tail_logs)
    if tail_curve_config is not None:
        logs["loss_keypoint_tail_curve_weighted"] = (
            tail_curve_config.weight * tail_loss
        )
    return total, logs

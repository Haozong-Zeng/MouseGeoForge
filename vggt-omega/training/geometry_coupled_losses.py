"""Geometry-coupled objectives for small-domain VGGT-Omega fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vggt_omega.utils.pose_enc import encoding_to_camera


def squeeze_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5:
        return depth[..., 0]
    if depth.ndim != 4:
        raise ValueError(f"Expected BxVxHxW or BxVxHxWx1 depth, got {tuple(depth.shape)}")
    return depth


def masked_weighted_l1(
    residual: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if residual.shape[: valid.ndim] != valid.shape:
        raise ValueError(
            f"Residual/valid shape mismatch: residual={tuple(residual.shape)} valid={tuple(valid.shape)}"
        )
    expanded_valid = valid
    while expanded_valid.ndim < residual.ndim:
        expanded_valid = expanded_valid.unsqueeze(-1)
    if expanded_valid.sum() < 1:
        return residual.mean() * 0.0
    absolute = residual.abs()
    if weights is None:
        expanded_weights = torch.ones_like(valid, dtype=absolute.dtype)
    else:
        expanded_weights = weights.to(dtype=absolute.dtype)
    while expanded_weights.ndim < residual.ndim:
        expanded_weights = expanded_weights.unsqueeze(-1)
    expanded_weights = expanded_weights.expand_as(absolute)
    expanded_valid = expanded_valid.expand_as(absolute)
    numerator = (absolute * expanded_weights * expanded_valid).sum()
    denominator = (expanded_weights * expanded_valid).sum().clamp_min(1e-6)
    return numerator / denominator


def spatial_gradient_l1(
    residual: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    residual_x = residual[..., :, 1:, :] - residual[..., :, :-1, :] if residual.ndim == 5 else (
        residual[..., :, 1:] - residual[..., :, :-1]
    )
    weight_x = None
    if weights is not None:
        weight_x = 0.5 * (weights[..., :, 1:] + weights[..., :, :-1])
    losses.append(masked_weighted_l1(residual_x, valid_x, weight_x))

    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    residual_y = residual[..., 1:, :, :] - residual[..., :-1, :, :] if residual.ndim == 5 else (
        residual[..., 1:, :] - residual[..., :-1, :]
    )
    weight_y = None
    if weights is not None:
        weight_y = 0.5 * (weights[..., 1:, :] + weights[..., :-1, :])
    losses.append(masked_weighted_l1(residual_y, valid_y, weight_y))
    return torch.stack(losses).mean()


def official_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    pixel_weights: torch.Tensor | None = None,
    gradient_weight: float = 1.0,
    confidence: torch.Tensor | None = None,
    confidence_alpha: float = 0.0,
) -> dict[str, torch.Tensor]:
    """VGGT-Omega relative-depth and gradient objective.

    Confidence is intentionally optional. The paper recommends disabling the
    aleatoric term for small fine-tuning sets and calibrating it separately.
    """

    pred = squeeze_depth(pred_depth)
    target = squeeze_depth(target_depth)
    valid = valid & torch.isfinite(pred) & torch.isfinite(target) & (pred > 1e-6) & (target > 1e-6)
    residual = pred - target
    relative_weight = 1.0 + target.clamp_min(1e-6).reciprocal()
    weights = relative_weight
    confidence_term = residual.mean() * 0.0
    if confidence is not None:
        conf = squeeze_depth(confidence).clamp(1e-4, 100.0)
        weights = weights * conf
        if valid.any() and confidence_alpha > 0:
            confidence_term = -float(confidence_alpha) * torch.log(conf[valid]).mean()
    if pixel_weights is not None:
        weights = weights * pixel_weights.to(dtype=weights.dtype)
    value = masked_weighted_l1(residual, valid, weights)
    gradient = spatial_gradient_l1(residual, valid, weights)
    total = value + float(gradient_weight) * gradient + confidence_term
    return {
        "loss_depth_official": total,
        "loss_depth_official_value": value,
        "loss_depth_official_gradient": gradient,
        "loss_depth_official_confidence": confidence_term,
        "depth_official_mae": masked_weighted_l1(residual, valid),
    }


def depth_to_world_points(
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
    """Unproject depth through OpenCV camera-from-world matrices."""

    depth = squeeze_depth(depth)
    stride = max(int(stride), 1)
    depth = depth[..., ::stride, ::stride]
    batch_size, views, height, width = depth.shape
    full_height = height * stride
    full_width = width * stride
    y = torch.arange(0, full_height, stride, device=depth.device, dtype=depth.dtype)[:height]
    x = torch.arange(0, full_width, stride, device=depth.device, dtype=depth.dtype)[:width]
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.view(1, 1, height, width)
    grid_y = grid_y.view(1, 1, height, width)
    fx = intrinsics[..., 0, 0, None, None]
    fy = intrinsics[..., 1, 1, None, None]
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    camera = torch.stack(
        [
            (grid_x - cx) / fx.clamp_min(1e-6) * depth,
            (grid_y - cy) / fy.clamp_min(1e-6) * depth,
            depth,
        ],
        dim=-1,
    )
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return torch.einsum(
        "bvij,bvhwj->bvhwi",
        rotation.transpose(-1, -2),
        camera - translation[:, :, None, None, :],
    )


def official_point_loss(
    pred_depth: torch.Tensor,
    pred_pose: torch.Tensor,
    target_depth: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    valid: torch.Tensor,
    image_size_hw: tuple[int, int],
    *,
    stride: int = 1,
    gradient_weight: float = 1.0,
    confidence: torch.Tensor | None = None,
    confidence_alpha: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Official-style point-map loss coupling predicted depth and cameras."""

    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pred_pose,
        image_size_hw,
        build_intrinsics=True,
    )
    pred_points = depth_to_world_points(
        pred_depth,
        pred_extrinsics,
        pred_intrinsics,
        stride=stride,
    )
    with torch.no_grad():
        target_points = depth_to_world_points(
            target_depth,
            target_extrinsics,
            target_intrinsics,
            stride=stride,
        )
    target = squeeze_depth(target_depth)[..., ::stride, ::stride]
    valid = valid[..., ::stride, ::stride]
    valid = (
        valid
        & torch.isfinite(pred_points).all(dim=-1)
        & torch.isfinite(target_points).all(dim=-1)
        & torch.isfinite(target)
        & (target > 1e-6)
    )
    residual = pred_points - target_points
    weights = 1.0 + target.clamp_min(1e-6).reciprocal()
    confidence_term = residual.mean() * 0.0
    if confidence is not None:
        conf = squeeze_depth(confidence)[..., ::stride, ::stride].clamp(1e-4, 100.0)
        weights = weights * conf
        if valid.any() and confidence_alpha > 0:
            confidence_term = -float(confidence_alpha) * torch.log(conf[valid]).mean()
    value = masked_weighted_l1(residual, valid, weights)
    gradient = spatial_gradient_l1(residual, valid, weights)
    total = value + float(gradient_weight) * gradient + confidence_term
    euclidean = torch.linalg.vector_norm(residual, dim=-1)
    mae = euclidean[valid].mean() if valid.any() else euclidean.mean() * 0.0
    return {
        "loss_point": total,
        "loss_point_value": value,
        "loss_point_gradient": gradient,
        "loss_point_confidence": confidence_term,
        "point_mae_normalized": mae,
        "point_valid_pixels": valid.sum().to(dtype=pred_points.dtype),
    }


def _sample_map_at_xy(values: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, views, height, width = values.shape
    points = xy.shape[-2]
    grid = xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        values.reshape(batch_size * views, 1, height, width),
        grid.reshape(batch_size * views, points, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch_size, views, points)
    in_frame = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    return sampled, in_frame


def unproject_sparse_points(
    xy: torch.Tensor,
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    fx = intrinsics[..., 0, 0, None]
    fy = intrinsics[..., 1, 1, None]
    cx = intrinsics[..., 0, 2, None]
    cy = intrinsics[..., 1, 2, None]
    camera = torch.stack(
        [
            (xy[..., 0] - cx) / fx.clamp_min(1e-6) * depth,
            (xy[..., 1] - cy) / fy.clamp_min(1e-6) * depth,
            depth,
        ],
        dim=-1,
    )
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return torch.einsum(
        "bvij,bvkj->bvki",
        rotation.transpose(-1, -2),
        camera - translation[:, :, None, :],
    )


def project_sparse_points(
    world: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = torch.einsum("bvij,bkj->bvki", extrinsics[..., :3, :3], world)
    camera = camera + extrinsics[..., :3, 3][:, :, None, :]
    homogeneous = torch.einsum("bvij,bvkj->bvki", intrinsics, camera)
    uv = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-6)
    return uv, camera[..., 2]


def camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return -torch.einsum(
        "...ij,...j->...i",
        rotation.transpose(-1, -2),
        translation,
    )


def median_pairwise_camera_baseline(extrinsics: torch.Tensor) -> torch.Tensor:
    centers = camera_centers(extrinsics)
    distances = []
    for first in range(centers.shape[1]):
        for second in range(first + 1, centers.shape[1]):
            distances.append(
                torch.linalg.vector_norm(
                    centers[:, first] - centers[:, second],
                    dim=-1,
                )
            )
    if not distances:
        return centers.new_ones(centers.shape[0])
    return torch.stack(distances, dim=-1).median(dim=-1).values


@torch.no_grad()
def relative_rig_frame_errors(
    pred_pose: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame relative rotation and normalized-center target errors."""

    pred_extrinsics, _ = encoding_to_camera(
        pred_pose,
        batch["images"].shape[-2:],
        build_intrinsics=True,
    )
    target_extrinsics = batch["extrinsics"].to(dtype=pred_extrinsics.dtype)
    available = batch.get(
        "camera_available",
        torch.ones(
            pred_extrinsics.shape[:2],
            dtype=torch.bool,
            device=pred_extrinsics.device,
        ),
    ).bool()
    pred_rotations = pred_extrinsics[..., :3, :3]
    target_rotations = target_extrinsics[..., :3, :3]
    pred_centers = camera_centers(pred_extrinsics)
    target_centers = camera_centers(target_extrinsics)

    rotation_sum = pred_pose.new_zeros(pred_pose.shape[0])
    rotation_count = pred_pose.new_zeros(pred_pose.shape[0])
    pred_baselines = []
    target_baselines = []
    baseline_valid = []
    for first in range(pred_extrinsics.shape[1]):
        for second in range(first + 1, pred_extrinsics.shape[1]):
            valid = available[:, first] & available[:, second]
            pred_relative = (
                pred_rotations[:, second]
                @ pred_rotations[:, first].transpose(-1, -2)
            )
            target_relative = (
                target_rotations[:, second]
                @ target_rotations[:, first].transpose(-1, -2)
            )
            delta = pred_relative @ target_relative.transpose(-1, -2)
            cosine = (
                (delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            ).clamp(-1.0, 1.0)
            angle = torch.rad2deg(
                torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            )
            rotation_sum += torch.where(valid, angle, torch.zeros_like(angle))
            rotation_count += valid.to(rotation_count.dtype)
            pred_baselines.append(
                torch.linalg.vector_norm(
                    pred_centers[:, second] - pred_centers[:, first],
                    dim=-1,
                )
            )
            target_baselines.append(
                torch.linalg.vector_norm(
                    target_centers[:, second] - target_centers[:, first],
                    dim=-1,
                )
            )
            baseline_valid.append(valid)
    rotation_error = rotation_sum / rotation_count.clamp_min(1.0)

    pair_valid = torch.stack(baseline_valid, dim=-1)
    pair_count = pair_valid.sum(dim=-1).clamp_min(1)
    pred_scale = (
        (torch.stack(pred_baselines, dim=-1) * pair_valid).sum(dim=-1)
        / pair_count
    ).clamp_min(1e-6)
    target_scale = (
        (torch.stack(target_baselines, dim=-1) * pair_valid).sum(dim=-1)
        / pair_count
    ).clamp_min(1e-6)
    pred_relative_centers = torch.einsum(
        "bij,bvj->bvi",
        pred_rotations[:, 0],
        pred_centers - pred_centers[:, :1],
    ) / pred_scale[:, None, None]
    target_relative_centers = torch.einsum(
        "bij,bvj->bvi",
        target_rotations[:, 0],
        target_centers - target_centers[:, :1],
    ) / target_scale[:, None, None]
    center_valid = available & available[:, :1]
    center_norm = torch.linalg.vector_norm(
        pred_relative_centers - target_relative_centers,
        dim=-1,
    )
    center_error = (
        (center_norm * center_valid).sum(dim=-1)
        / center_valid.sum(dim=-1).clamp_min(1)
    )
    return rotation_error, center_error


def relative_rig_camera_loss(
    pred_pose: torch.Tensor,
    batch: dict[str, Any],
    *,
    rotation_weight: float = 1.0,
    center_weight: float = 1.0,
    baseline_weight: float = 0.5,
    focal_weight: float = 0.25,
    frame_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise camera geometry without depending on world gauge or scale."""

    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pred_pose,
        batch["images"].shape[-2:],
        build_intrinsics=True,
    )
    target_extrinsics = batch["extrinsics"].to(dtype=pred_extrinsics.dtype)
    target_intrinsics = batch["intrinsics"].to(dtype=pred_intrinsics.dtype)
    available = batch.get(
        "camera_available",
        torch.ones(
            pred_extrinsics.shape[:2],
            dtype=torch.bool,
            device=pred_extrinsics.device,
        ),
    ).bool()
    if frame_mask is not None:
        available = available & frame_mask.to(
            device=available.device,
            dtype=torch.bool,
        )[:, None]
    views = pred_extrinsics.shape[1]
    zero = pred_pose.mean() * 0.0

    pred_rotations = pred_extrinsics[..., :3, :3]
    target_rotations = target_extrinsics[..., :3, :3]
    rotation_terms = []
    rotation_angles = []
    pair_valid_masks = []
    pred_baselines = []
    target_baselines = []
    pred_centers = camera_centers(pred_extrinsics)
    target_centers = camera_centers(target_extrinsics)
    for first in range(views):
        for second in range(first + 1, views):
            pair_valid = available[:, first] & available[:, second]
            if not pair_valid.any():
                continue
            pred_relative = (
                pred_rotations[:, second]
                @ pred_rotations[:, first].transpose(-1, -2)
            )
            target_relative = (
                target_rotations[:, second]
                @ target_rotations[:, first].transpose(-1, -2)
            )
            delta = pred_relative @ target_relative.transpose(-1, -2)
            cosine = (
                (delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            ).clamp(-1.0, 1.0)
            rotation_terms.append((1.0 - cosine)[pair_valid].mean())
            rotation_angles.append(
                torch.rad2deg(torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)))[
                    pair_valid
                ].mean()
            )
            pair_valid_masks.append(pair_valid)
            pred_baselines.append(
                torch.linalg.vector_norm(
                    pred_centers[:, second] - pred_centers[:, first],
                    dim=-1,
                )
            )
            target_baselines.append(
                torch.linalg.vector_norm(
                    target_centers[:, second] - target_centers[:, first],
                    dim=-1,
                )
            )

    if not rotation_terms:
        return {
            "loss_relative_rig": zero,
            "loss_relative_rig_rotation": zero,
            "loss_relative_rig_center": zero,
            "loss_relative_rig_baseline": zero,
            "loss_relative_rig_focal": zero,
            "relative_rig_rotation_error_deg": zero,
            "relative_rig_center_error": zero,
            "relative_rig_baseline_ratio_error": zero,
        }

    rotation_loss = torch.stack(rotation_terms).mean()
    rotation_error_deg = torch.stack(rotation_angles).mean()
    pred_baseline_stack = torch.stack(pred_baselines, dim=-1)
    target_baseline_stack = torch.stack(target_baselines, dim=-1)
    pair_available = torch.stack(pair_valid_masks, dim=-1)
    pair_count = pair_available.sum(dim=-1).clamp_min(1)
    pred_scale = (
        (pred_baseline_stack * pair_available).sum(dim=-1) / pair_count
    ).clamp_min(1e-6)
    target_scale = (
        (target_baseline_stack * pair_available).sum(dim=-1) / pair_count
    ).clamp_min(1e-6)
    pred_ratio = pred_baseline_stack / pred_scale[:, None]
    target_ratio = target_baseline_stack / target_scale[:, None]
    baseline_residual = pred_ratio - target_ratio
    baseline_loss = F.smooth_l1_loss(
        baseline_residual[pair_available],
        torch.zeros_like(baseline_residual[pair_available]),
    )
    baseline_ratio_error = baseline_residual[pair_available].abs().mean()

    anchor_valid = available[:, :1]
    pred_anchor_relative = pred_centers - pred_centers[:, :1]
    target_anchor_relative = target_centers - target_centers[:, :1]
    pred_anchor_relative = torch.einsum(
        "bij,bvj->bvi",
        pred_rotations[:, 0],
        pred_anchor_relative,
    ) / pred_scale[:, None, None]
    target_anchor_relative = torch.einsum(
        "bij,bvj->bvi",
        target_rotations[:, 0],
        target_anchor_relative,
    ) / target_scale[:, None, None]
    center_valid = available & anchor_valid
    center_residual = pred_anchor_relative - target_anchor_relative
    center_loss = F.smooth_l1_loss(
        center_residual[center_valid],
        torch.zeros_like(center_residual[center_valid]),
    )
    center_error = torch.linalg.vector_norm(center_residual, dim=-1)[
        center_valid
    ].mean()

    pred_focal = torch.stack(
        [pred_intrinsics[..., 0, 0], pred_intrinsics[..., 1, 1]],
        dim=-1,
    )
    target_focal = torch.stack(
        [target_intrinsics[..., 0, 0], target_intrinsics[..., 1, 1]],
        dim=-1,
    )
    focal_residual = torch.log(pred_focal.clamp_min(1e-6)) - torch.log(
        target_focal.clamp_min(1e-6)
    )
    focal_loss = F.smooth_l1_loss(
        focal_residual[available],
        torch.zeros_like(focal_residual[available]),
    )
    total = (
        float(rotation_weight) * rotation_loss
        + float(center_weight) * center_loss
        + float(baseline_weight) * baseline_loss
        + float(focal_weight) * focal_loss
    )
    return {
        "loss_relative_rig": total,
        "loss_relative_rig_rotation": rotation_loss,
        "loss_relative_rig_center": center_loss,
        "loss_relative_rig_baseline": baseline_loss,
        "loss_relative_rig_focal": focal_loss,
        "relative_rig_rotation_error_deg": rotation_error_deg,
        "relative_rig_center_error": center_error,
        "relative_rig_baseline_ratio_error": baseline_ratio_error,
    }


def shared_rig_temporal_loss(
    pred_pose: torch.Tensor,
    batch: dict[str, Any],
    *,
    frame_mask: torch.Tensor | None = None,
    outlier_rotation_deg: float = 15.0,
    outlier_center_error: float = 0.35,
    rotation_weight: float = 1.0,
    center_weight: float = 1.0,
    baseline_weight: float = 1.0,
    focal_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Keep one predicted relative rig across high-confidence frames of a trial."""

    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pred_pose,
        batch["images"].shape[-2:],
        build_intrinsics=True,
    )
    frame_count, views = pred_extrinsics.shape[:2]
    zero = pred_pose.mean() * 0.0
    if frame_mask is None:
        frame_mask = torch.ones(
            frame_count,
            dtype=torch.bool,
            device=pred_pose.device,
        )
    else:
        frame_mask = frame_mask.to(device=pred_pose.device, dtype=torch.bool)
    confidence = batch.get(
        "rig_frame_confidence",
        torch.ones(frame_count, device=pred_pose.device),
    ).to(device=pred_pose.device, dtype=pred_extrinsics.dtype)
    coverage = batch.get(
        "rig_frame_coverage",
        torch.ones(frame_count, device=pred_pose.device),
    ).to(device=pred_pose.device, dtype=pred_extrinsics.dtype)
    frame_weights = (confidence * coverage).clamp_min(1e-4)

    rotations = pred_extrinsics[..., :3, :3]
    centers = camera_centers(pred_extrinsics)
    relative_rotations = []
    baselines = []
    for first in range(views):
        for second in range(first + 1, views):
            relative_rotations.append(
                rotations[:, second] @ rotations[:, first].transpose(-1, -2)
            )
            baselines.append(
                torch.linalg.vector_norm(
                    centers[:, second] - centers[:, first],
                    dim=-1,
                )
            )
    if not relative_rotations or int(frame_mask.sum()) < 2:
        return {
            "loss_shared_rig_temporal": zero,
            "loss_shared_rig_temporal_rotation": zero,
            "loss_shared_rig_temporal_center": zero,
            "loss_shared_rig_temporal_baseline": zero,
            "loss_shared_rig_temporal_focal": zero,
            "shared_rig_temporal_rotation_error_deg": zero,
            "shared_rig_temporal_center_error": zero,
            "shared_rig_temporal_baseline_error": zero,
            "shared_rig_candidate_frames": frame_mask.sum().to(zero.dtype),
            "shared_rig_accepted_frames": frame_mask.sum().to(zero.dtype),
            "shared_rig_outlier_frames": zero,
        }

    relative_rotation_stack = torch.stack(relative_rotations, dim=1)
    baseline_stack = torch.stack(baselines, dim=-1)
    baseline_scale = baseline_stack.mean(dim=-1).clamp_min(1e-6)
    baseline_ratio = baseline_stack / baseline_scale[:, None]
    anchor_relative_centers = centers - centers[:, :1]
    anchor_relative_centers = torch.einsum(
        "bij,bvj->bvi",
        rotations[:, 0],
        anchor_relative_centers,
    ) / baseline_scale[:, None, None]
    focal_feature = torch.log(
        torch.stack(
            [pred_intrinsics[..., 0, 0], pred_intrinsics[..., 1, 1]],
            dim=-1,
        ).clamp_min(1e-6)
    )

    # Choose a robust medoid before filtering, so one bad prediction cannot drag
    # the temporal consensus toward itself.
    with torch.no_grad():
        candidate_indices = torch.nonzero(frame_mask, as_tuple=False).flatten()
        medoid_costs = []
        for first in candidate_indices:
            pair_costs = []
            for second in candidate_indices:
                delta = (
                    relative_rotation_stack[first]
                    @ relative_rotation_stack[second].transpose(-1, -2)
                )
                cosine = (
                    (delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
                ).clamp(-1.0, 1.0)
                rotation_error = torch.rad2deg(
                    torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
                ).mean()
                center_error = torch.linalg.vector_norm(
                    anchor_relative_centers[first] - anchor_relative_centers[second],
                    dim=-1,
                ).mean()
                pair_costs.append(
                    rotation_error / max(float(outlier_rotation_deg), 1e-6)
                    + center_error / max(float(outlier_center_error), 1e-6)
                )
            medoid_costs.append(torch.stack(pair_costs).median())
        medoid = candidate_indices[torch.stack(medoid_costs).argmin()]
        delta = (
            relative_rotation_stack
            @ relative_rotation_stack[medoid].transpose(-1, -2)
        )
        cosine = (
            (delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        rotation_to_medoid = torch.rad2deg(
            torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        ).mean(dim=-1)
        center_to_medoid = torch.linalg.vector_norm(
            anchor_relative_centers - anchor_relative_centers[medoid],
            dim=-1,
        ).mean(dim=-1)
        accepted = (
            frame_mask
            & (rotation_to_medoid <= float(outlier_rotation_deg))
            & (center_to_medoid <= float(outlier_center_error))
        )
        accepted[medoid] = True

    accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten()
    if accepted_indices.numel() < 2:
        return {
            "loss_shared_rig_temporal": zero,
            "loss_shared_rig_temporal_rotation": zero,
            "loss_shared_rig_temporal_center": zero,
            "loss_shared_rig_temporal_baseline": zero,
            "loss_shared_rig_temporal_focal": zero,
            "shared_rig_temporal_rotation_error_deg": zero,
            "shared_rig_temporal_center_error": zero,
            "shared_rig_temporal_baseline_error": zero,
            "shared_rig_candidate_frames": frame_mask.sum().to(zero.dtype),
            "shared_rig_accepted_frames": accepted.sum().to(zero.dtype),
            "shared_rig_outlier_frames": (frame_mask & ~accepted).sum().to(zero.dtype),
        }

    pair_weights = []
    rotation_losses = []
    rotation_errors = []
    center_losses = []
    center_errors = []
    baseline_losses = []
    baseline_errors = []
    focal_losses = []
    for first_pos, first in enumerate(accepted_indices):
        for second in accepted_indices[first_pos + 1 :]:
            pair_weight = frame_weights[first] * frame_weights[second]
            delta = (
                relative_rotation_stack[first]
                @ relative_rotation_stack[second].transpose(-1, -2)
            )
            cosine = (
                (delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            ).clamp(-1.0, 1.0)
            rotation_losses.append((1.0 - cosine).mean())
            rotation_errors.append(
                torch.rad2deg(
                    torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
                ).mean()
            )
            center_residual = (
                anchor_relative_centers[first] - anchor_relative_centers[second]
            )
            center_losses.append(
                F.smooth_l1_loss(center_residual, torch.zeros_like(center_residual))
            )
            center_errors.append(
                torch.linalg.vector_norm(center_residual, dim=-1).mean()
            )
            baseline_residual = baseline_ratio[first] - baseline_ratio[second]
            baseline_losses.append(
                F.smooth_l1_loss(
                    baseline_residual,
                    torch.zeros_like(baseline_residual),
                )
            )
            baseline_errors.append(baseline_residual.abs().mean())
            focal_residual = focal_feature[first] - focal_feature[second]
            focal_losses.append(
                F.smooth_l1_loss(focal_residual, torch.zeros_like(focal_residual))
            )
            pair_weights.append(pair_weight)

    weights = torch.stack(pair_weights)
    weights = weights / weights.sum().clamp_min(1e-6)

    def weighted(values: list[torch.Tensor]) -> torch.Tensor:
        return (torch.stack(values) * weights).sum()

    rotation_loss = weighted(rotation_losses)
    center_loss = weighted(center_losses)
    baseline_loss = weighted(baseline_losses)
    focal_loss = weighted(focal_losses)
    total = (
        float(rotation_weight) * rotation_loss
        + float(center_weight) * center_loss
        + float(baseline_weight) * baseline_loss
        + float(focal_weight) * focal_loss
    )
    return {
        "loss_shared_rig_temporal": total,
        "loss_shared_rig_temporal_rotation": rotation_loss,
        "loss_shared_rig_temporal_center": center_loss,
        "loss_shared_rig_temporal_baseline": baseline_loss,
        "loss_shared_rig_temporal_focal": focal_loss,
        "shared_rig_temporal_rotation_error_deg": weighted(rotation_errors),
        "shared_rig_temporal_center_error": weighted(center_errors),
        "shared_rig_temporal_baseline_error": weighted(baseline_errors),
        "shared_rig_candidate_frames": frame_mask.sum().to(zero.dtype),
        "shared_rig_accepted_frames": accepted.sum().to(zero.dtype),
        "shared_rig_outlier_frames": (frame_mask & ~accepted).sum().to(zero.dtype),
    }


def multiview_keypoint_geometry_loss(
    pred_depth: torch.Tensor,
    pred_pose: torch.Tensor,
    batch: dict[str, Any],
    *,
    predicted_rig_weight: float,
    fixed_rig_weight: float,
    pairwise_weight: float,
    reprojection_weight: float,
) -> dict[str, torch.Tensor]:
    """Couple synchronized real keypoint depth and camera predictions in 3D."""

    depth = squeeze_depth(pred_depth)
    xy = batch["geometry_xy"]
    sampled_depth, in_frame = _sample_map_at_xy(depth, xy)
    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pred_pose,
        batch["images"].shape[-2:],
        build_intrinsics=True,
    )
    identity = torch.eye(
        3,
        device=pred_extrinsics.device,
        dtype=pred_extrinsics.dtype,
    )
    first_rotation_error = torch.linalg.matrix_norm(
        pred_extrinsics[:, 0, :3, :3] - identity,
        dim=(-2, -1),
    ).mean()
    first_translation_error = torch.linalg.vector_norm(
        pred_extrinsics[:, 0, :3, 3],
        dim=-1,
    ).mean()
    predicted_baseline = median_pairwise_camera_baseline(pred_extrinsics)
    fixed_baseline = median_pairwise_camera_baseline(batch["extrinsics"])
    baseline_ratio = (
        predicted_baseline / fixed_baseline.clamp_min(1e-6)
    ).mean()
    predicted_world = unproject_sparse_points(
        xy,
        sampled_depth,
        pred_extrinsics,
        pred_intrinsics,
    )
    fixed_world = unproject_sparse_points(
        xy,
        sampled_depth,
        batch["extrinsics"],
        batch["intrinsics"],
    )
    target_world = batch["geometry_world"][:, None, :, :]
    valid = (
        batch["geometry_view_valid"]
        & batch["geometry_valid"][:, None, :]
        & in_frame
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 1e-6)
        & torch.isfinite(predicted_world).all(dim=-1)
        & torch.isfinite(fixed_world).all(dim=-1)
    )
    weights = batch["geometry_weights"].to(dtype=depth.dtype)
    predicted_residual = predicted_world - target_world
    fixed_residual = fixed_world - target_world
    predicted_loss = masked_weighted_l1(predicted_residual, valid, weights)
    fixed_loss = masked_weighted_l1(fixed_residual, valid, weights)

    pairwise_losses = []
    pairwise_distances = []
    views = depth.shape[1]
    for first in range(views):
        for second in range(first + 1, views):
            pair_valid = valid[:, first] & valid[:, second]
            residual = predicted_world[:, first] - predicted_world[:, second]
            pair_weight = 0.5 * (weights[:, first] + weights[:, second])
            pairwise_losses.append(masked_weighted_l1(residual, pair_valid, pair_weight))
            distance = torch.linalg.vector_norm(residual, dim=-1)
            if pair_valid.any():
                pairwise_distances.append(distance[pair_valid])
    pairwise_loss = (
        torch.stack(pairwise_losses).mean()
        if pairwise_losses
        else predicted_world.mean() * 0.0
    )

    projected_uv, projected_depth = project_sparse_points(
        batch["geometry_world"],
        pred_extrinsics,
        pred_intrinsics,
    )
    fixed_projected_uv, fixed_projected_depth = project_sparse_points(
        batch["geometry_world"],
        batch["extrinsics"],
        batch["intrinsics"],
    )
    reprojection_valid = (
        batch["geometry_view_valid"]
        & batch["geometry_valid"][:, None, :]
        & torch.isfinite(projected_uv).all(dim=-1)
        & (projected_depth > 1e-6)
    )
    normalizer = depth.new_tensor([max(depth.shape[-1], 1), max(depth.shape[-2], 1)])
    reprojection_residual = (projected_uv - xy) / normalizer
    reprojection_loss = masked_weighted_l1(
        reprojection_residual,
        reprojection_valid,
        weights,
    )
    reprojection_px = torch.linalg.vector_norm(projected_uv - xy, dim=-1)
    fixed_reprojection_valid = (
        batch["geometry_view_valid"]
        & batch["geometry_valid"][:, None, :]
        & torch.isfinite(fixed_projected_uv).all(dim=-1)
        & (fixed_projected_depth > 1e-6)
    )
    fixed_reprojection_px = torch.linalg.vector_norm(fixed_projected_uv - xy, dim=-1)

    scale = batch["scale"].to(dtype=depth.dtype)[:, None, None]
    predicted_error_mm = torch.linalg.vector_norm(predicted_residual, dim=-1) * scale
    fixed_error_mm = torch.linalg.vector_norm(fixed_residual, dim=-1) * scale

    def valid_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values[mask].mean() if mask.any() else values.mean() * 0.0

    def semantic_mean(values: torch.Tensor, key: str) -> torch.Tensor:
        point_mask = batch.get(key)
        if point_mask is None:
            return values.mean() * 0.0
        return valid_mean(values, valid & point_mask[:, None, :])

    pairwise_spread_mm = predicted_world.mean() * 0.0
    if pairwise_distances:
        normalized_spread = torch.cat(pairwise_distances)
        mean_scale = batch["scale"].to(dtype=depth.dtype).mean()
        pairwise_spread_mm = normalized_spread.mean() * mean_scale

    total = (
        float(predicted_rig_weight) * predicted_loss
        + float(fixed_rig_weight) * fixed_loss
        + float(pairwise_weight) * pairwise_loss
        + float(reprojection_weight) * reprojection_loss
    )
    return {
        "loss_real_geometry": total,
        "loss_real_geometry_predicted_rig": predicted_loss,
        "loss_real_geometry_fixed_rig": fixed_loss,
        "loss_real_geometry_pairwise": pairwise_loss,
        "loss_real_geometry_reprojection": reprojection_loss,
        "geometry_predicted_mae_mm": valid_mean(predicted_error_mm, valid),
        "geometry_fixed_mae_mm": valid_mean(fixed_error_mm, valid),
        "geometry_palm_predicted_mae_mm": semantic_mean(
            predicted_error_mm,
            "geometry_is_palm",
        ),
        "geometry_palm_fixed_mae_mm": semantic_mean(
            fixed_error_mm,
            "geometry_is_palm",
        ),
        "geometry_finger_predicted_mae_mm": semantic_mean(
            predicted_error_mm,
            "geometry_is_finger",
        ),
        "geometry_face_predicted_mae_mm": semantic_mean(
            predicted_error_mm,
            "geometry_is_face",
        ),
        "geometry_pairwise_spread_mm": pairwise_spread_mm,
        "geometry_reprojection_error_px": valid_mean(
            reprojection_px,
            reprojection_valid,
        ),
        "geometry_label_reprojection_error_px": valid_mean(
            fixed_reprojection_px,
            fixed_reprojection_valid,
        ),
        "geometry_valid_points": valid.sum().to(dtype=depth.dtype),
        "geometry_gauge_first_rotation_error": first_rotation_error,
        "geometry_gauge_first_translation_error": first_translation_error,
        "geometry_gauge_baseline_ratio": baseline_ratio,
    }


def token_matching_loss(
    patch_tokens: torch.Tensor,
    batch: dict[str, Any],
    *,
    patch_size: int = 16,
    max_pairs: int = 256,
    depth_tolerance: float = 0.01,
    border: int = 4,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Geometry-derived positive and far-patch negative matching objective."""

    depths = squeeze_depth(batch["depths"])
    masks = batch["depth_masks"] & batch["depth_available"][..., None, None]
    extrinsics = batch["extrinsics"]
    intrinsics = batch["intrinsics"]
    batch_size, views, height, width = depths.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    patch_count = patch_height * patch_width
    if patch_tokens.shape[2] != patch_count:
        raise ValueError(
            f"Patch token count mismatch: tokens={patch_tokens.shape[2]} grid={patch_height}x{patch_width}"
        )

    row = torch.arange(patch_height, device=depths.device)
    col = torch.arange(patch_width, device=depths.device)
    grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
    query_index = (grid_row * patch_width + grid_col).reshape(-1)
    query_xy = torch.stack(
        [
            (grid_col.reshape(-1).to(depths.dtype) + 0.5) * patch_size,
            (grid_row.reshape(-1).to(depths.dtype) + 0.5) * patch_size,
        ],
        dim=-1,
    )
    query_x_int = query_xy[:, 0].round().long().clamp(0, width - 1)
    query_y_int = query_xy[:, 1].round().long().clamp(0, height - 1)
    positive_features = []
    positive_targets = []
    negative_features = []
    negative_targets = []

    with torch.no_grad():
        for batch_idx in range(batch_size):
            for query_view in range(views):
                query_depth = depths[batch_idx, query_view, query_y_int, query_x_int]
                query_valid = masks[batch_idx, query_view, query_y_int, query_x_int]
                query_world = unproject_sparse_points(
                    query_xy[None, None],
                    query_depth[None, None],
                    extrinsics[batch_idx : batch_idx + 1, query_view : query_view + 1],
                    intrinsics[batch_idx : batch_idx + 1, query_view : query_view + 1],
                )[0, 0]
                for target_view in range(views):
                    if target_view == query_view:
                        continue
                    target_uv, target_z = project_sparse_points(
                        query_world[None],
                        extrinsics[batch_idx : batch_idx + 1, target_view : target_view + 1],
                        intrinsics[batch_idx : batch_idx + 1, target_view : target_view + 1],
                    )
                    target_uv = target_uv[0, 0]
                    target_z = target_z[0, 0]
                    target_x = target_uv[:, 0].round().long().clamp(0, width - 1)
                    target_y = target_uv[:, 1].round().long().clamp(0, height - 1)
                    target_depth = depths[batch_idx, target_view, target_y, target_x]
                    target_valid = masks[batch_idx, target_view, target_y, target_x]
                    inside = (
                        (target_uv[:, 0] >= border)
                        & (target_uv[:, 0] < width - border)
                        & (target_uv[:, 1] >= border)
                        & (target_uv[:, 1] < height - border)
                    )
                    consistent = (
                        (target_depth - target_z).abs()
                        / target_depth.abs().clamp_min(1e-6)
                        <= float(depth_tolerance)
                    )
                    valid = query_valid & target_valid & inside & consistent & (target_z > 1e-6)
                    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
                    if valid_indices.numel() == 0:
                        continue
                    if valid_indices.numel() > max_pairs:
                        selection = torch.randperm(
                            valid_indices.numel(),
                            device=valid_indices.device,
                        )[:max_pairs]
                        valid_indices = valid_indices[selection]
                    target_patch_x = (target_uv[valid_indices, 0] / patch_size).floor().long()
                    target_patch_y = (target_uv[valid_indices, 1] / patch_size).floor().long()
                    target_patch_x = target_patch_x.clamp(0, patch_width - 1)
                    target_patch_y = target_patch_y.clamp(0, patch_height - 1)
                    target_index = target_patch_y * patch_width + target_patch_x
                    query_selected = query_index[valid_indices]
                    negative_index = (
                        target_index
                        + max(patch_width // 2, 1)
                        + max(patch_height // 2, 1) * patch_width
                    ) % patch_count
                    positive_features.append(
                        (batch_idx, query_view, target_view, query_selected, target_index)
                    )
                    negative_features.append(
                        (batch_idx, query_view, target_view, query_selected, negative_index)
                    )

    for batch_idx, query_view, target_view, query_idx, target_idx in positive_features:
        first = F.normalize(patch_tokens[batch_idx, query_view, query_idx].float(), dim=-1)
        second = F.normalize(patch_tokens[batch_idx, target_view, target_idx].float(), dim=-1)
        positive_targets.append((first * second).sum(dim=-1) / max(float(temperature), 1e-3))
    for batch_idx, query_view, target_view, query_idx, target_idx in negative_features:
        first = F.normalize(patch_tokens[batch_idx, query_view, query_idx].float(), dim=-1)
        second = F.normalize(patch_tokens[batch_idx, target_view, target_idx].float(), dim=-1)
        negative_targets.append((first * second).sum(dim=-1) / max(float(temperature), 1e-3))

    if not positive_targets:
        zero = patch_tokens.float().mean() * 0.0
        return {
            "loss_match": zero,
            "match_positive_pairs": zero,
            "match_negative_pairs": zero,
            "match_positive_similarity": zero,
            "match_negative_similarity": zero,
        }
    positive_logits = torch.cat(positive_targets)
    negative_logits = torch.cat(negative_targets)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
        )
        + F.binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits),
        )
    )
    temperature_value = max(float(temperature), 1e-3)
    return {
        "loss_match": loss,
        "match_positive_pairs": positive_logits.new_tensor(positive_logits.numel()),
        "match_negative_pairs": negative_logits.new_tensor(negative_logits.numel()),
        "match_positive_similarity": (positive_logits * temperature_value).mean(),
        "match_negative_similarity": (negative_logits * temperature_value).mean(),
    }

"""Hungarian matching and partial-label losses for multi-mouse instance pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InstancePoseLossWeights:
    objectness: float = 1.0
    box_l1: float = 5.0
    box_giou: float = 2.0
    mask_bce: float = 2.0
    mask_dice: float = 2.0
    keypoint_l1: float = 5.0
    keypoint_visibility: float = 0.5
    local_heatmap: float = 1.0
    reid: float = 0.25
    keypoint_3d: float = 1.0
    reprojection: float = 1.0
    no_object: float = 0.1


@torch.no_grad()
def hungarian_match_instances(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Match unordered prediction queries to labeled instances in each view."""
    objectness = predictions["instance_objectness_logits"]
    boxes = predictions["instance_boxes_cxcywh"]
    keypoints = predictions["instance_keypoint_xy"]
    batch, views, queries = objectness.shape
    matches: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for batch_index in range(batch):
        view_matches = []
        for view_index in range(views):
            target_indices = torch.nonzero(
                targets["instance_valid"][batch_index, view_index], as_tuple=False
            ).flatten()
            if target_indices.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=objectness.device)
                view_matches.append((empty, empty))
                continue
            if target_indices.numel() > queries:
                raise ValueError(
                    f"Found {target_indices.numel()} targets but only {queries} queries"
                )
            cost = -objectness[batch_index, view_index].sigmoid()[None, :]
            target_boxes = targets["boxes_cxcywh"][batch_index, view_index, target_indices]
            box_cost = torch.cdist(target_boxes, boxes[batch_index, view_index], p=1)
            if "box_valid" in targets:
                box_cost = box_cost * targets["box_valid"][
                    batch_index, view_index, target_indices
                ][:, None].to(box_cost.dtype)
            cost = cost + 5.0 * box_cost

            target_xy = targets["keypoint_xy"][batch_index, view_index, target_indices]
            target_valid = targets["keypoint_valid"][batch_index, view_index, target_indices]
            predicted_xy = keypoints[batch_index, view_index]
            point_cost = cost.new_zeros((target_indices.numel(), queries))
            for target_row in range(target_indices.numel()):
                valid = target_valid[target_row]
                if valid.any():
                    delta = (
                        predicted_xy[:, valid]
                        - target_xy[target_row, valid][None]
                    ).abs().mean(dim=(-2, -1))
                    point_cost[target_row] = delta
            cost = cost + 5.0 * point_cost
            prediction_indices, target_rows = _linear_sum_assignment(cost)
            view_matches.append(
                (prediction_indices, target_indices[target_rows])
            )
        matches.append(view_matches)
    return matches


def instance_pose_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: InstancePoseLossWeights = InstancePoseLossWeights(),
    *,
    image_size: tuple[int, int] | None = None,
    world_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute set-prediction losses while ignoring absent partial labels."""
    matches = hungarian_match_instances(predictions, targets)
    objectness = predictions["instance_objectness_logits"]
    object_target = torch.zeros_like(objectness)
    object_weight = torch.full_like(objectness, float(weights.no_object))
    matched_prediction: list[tuple[int, int, int]] = []
    matched_target: list[tuple[int, int, int]] = []
    for batch_index, view_matches in enumerate(matches):
        for view_index, (prediction_indices, target_indices) in enumerate(view_matches):
            if prediction_indices.numel() == 0:
                continue
            object_target[batch_index, view_index, prediction_indices] = 1.0
            object_weight[batch_index, view_index, prediction_indices] = 1.0
            matched_prediction.extend(
                (batch_index, view_index, int(index)) for index in prediction_indices
            )
            matched_target.extend(
                (batch_index, view_index, int(index)) for index in target_indices
            )
    objectness_loss = (
        F.binary_cross_entropy_with_logits(objectness, object_target, reduction="none")
        * object_weight
    ).sum() / object_weight.sum().clamp_min(1.0)

    zero = objectness.mean() * 0.0
    box_l1 = box_giou = mask_bce = mask_dice = zero
    keypoint_l1 = visibility_loss = local_heatmap_loss = zero
    if matched_prediction:
        device = objectness.device
        pred_index = tuple(
            torch.tensor([entry[axis] for entry in matched_prediction], device=device)
            for axis in range(3)
        )
        target_index = tuple(
            torch.tensor([entry[axis] for entry in matched_target], device=device)
            for axis in range(3)
        )
        predicted_boxes = predictions["instance_boxes_cxcywh"][pred_index]
        target_boxes = targets["boxes_cxcywh"][target_index]
        matched_box_valid = targets.get(
            "box_valid", targets["instance_valid"]
        )[target_index]
        if matched_box_valid.any():
            predicted_valid_boxes = predicted_boxes[matched_box_valid]
            target_valid_boxes = target_boxes[matched_box_valid]
            box_l1 = F.l1_loss(predicted_valid_boxes, target_valid_boxes)
            box_giou = (1.0 - torch.diagonal(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(predicted_valid_boxes),
                    box_cxcywh_to_xyxy(target_valid_boxes),
                )
            )).mean()

        predicted_xy = predictions["instance_keypoint_xy"][pred_index]
        target_xy = targets["keypoint_xy"][target_index]
        point_valid = targets["keypoint_valid"][target_index]
        if point_valid.any():
            keypoint_l1 = F.smooth_l1_loss(
                predicted_xy[point_valid],
                target_xy[point_valid],
                beta=0.01,
            )
            predicted_visibility = predictions[
                "instance_keypoint_visibility_logits"
            ][pred_index]
            visibility_target = targets.get("keypoint_visible", targets["keypoint_valid"])[target_index]
            visibility_valid = targets.get(
                "keypoint_visibility_valid", targets["keypoint_valid"]
            )[target_index]
            visibility_supervised = point_valid & visibility_valid
            if visibility_supervised.any():
                visibility_loss = F.binary_cross_entropy_with_logits(
                    predicted_visibility[visibility_supervised],
                    visibility_target[visibility_supervised].to(predicted_visibility.dtype),
                )
            local_heatmap_loss = _local_keypoint_heatmap_loss(
                predictions["instance_keypoint_local_heatmap_logits"][pred_index],
                predictions["instance_keypoint_coarse_xy"][pred_index],
                target_xy,
                point_valid,
            )

        if "masks" in targets:
            mask_valid = targets.get(
                "mask_valid",
                torch.ones_like(targets["instance_valid"]),
            )[target_index]
            if mask_valid.any():
                predicted_masks = predictions["instance_mask_logits"][pred_index][mask_valid]
                target_masks = targets["masks"][target_index][mask_valid].float()
                target_masks = F.interpolate(
                    target_masks[:, None],
                    size=predicted_masks.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
                mask_bce = F.binary_cross_entropy_with_logits(predicted_masks, target_masks)
                mask_dice = _dice_loss(predicted_masks, target_masks)

    reid_loss = _matched_reid_loss(predictions, targets, matches)
    keypoint_3d_loss, reprojection_loss, geometry_logs = (
        _matched_multiview_geometry_losses(
            predictions,
            targets,
            matches,
            image_size=image_size,
            world_scale=world_scale,
        )
    )
    total = (
        weights.objectness * objectness_loss
        + weights.box_l1 * box_l1
        + weights.box_giou * box_giou
        + weights.mask_bce * mask_bce
        + weights.mask_dice * mask_dice
        + weights.keypoint_l1 * keypoint_l1
        + weights.keypoint_visibility * visibility_loss
        + weights.local_heatmap * local_heatmap_loss
        + weights.reid * reid_loss
        + weights.keypoint_3d * keypoint_3d_loss
        + weights.reprojection * reprojection_loss
    )
    logs = {
        "loss_instance_pose": total,
        "loss_instance_objectness": objectness_loss,
        "loss_instance_box_l1": box_l1,
        "loss_instance_box_giou": box_giou,
        "loss_instance_mask_bce": mask_bce,
        "loss_instance_mask_dice": mask_dice,
        "loss_instance_keypoint_l1": keypoint_l1,
        "loss_instance_keypoint_visibility": visibility_loss,
        "loss_instance_local_heatmap": local_heatmap_loss,
        "loss_instance_reid": reid_loss,
        "loss_instance_keypoint_3d": keypoint_3d_loss,
        "loss_instance_reprojection": reprojection_loss,
        "matched_instances": object_target.sum(),
        **geometry_logs,
    }
    return total, logs


def single_mouse_batch_as_instances(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Adapt the legacy one-mouse batch to the new padded instance contract.

    New multi-mouse datasets should emit these tensors directly with ``M > 1``.
    """
    images = batch["images"]
    batch_size, views, _, height, width = images.shape
    keypoint_xy = batch["keypoint_uv"].clone()
    keypoint_xy[..., 0] /= float(width)
    keypoint_xy[..., 1] /= float(height)
    keypoint_xy = keypoint_xy[:, :, None]
    point_valid = (
        batch["keypoint_valid"][:, None, :]
        & batch["keypoint_in_frame"]
    )[:, :, None]
    instance_valid = batch["keypoints_available"][:, None, None].expand(
        batch_size, views, 1
    ).clone()
    masks = batch.get("body_masks")
    if masks is None:
        masks = torch.zeros(
            (batch_size, views, height, width),
            dtype=torch.bool,
            device=images.device,
        )
    mask_valid = masks.flatten(-2).any(dim=-1)[:, :, None] & instance_valid
    boxes = _boxes_from_masks_or_keypoints(masks, keypoint_xy[:, :, 0], point_valid[:, :, 0])
    result = {
        "instance_valid": instance_valid,
        "instance_ids": torch.zeros(
            (batch_size, views, 1), dtype=torch.long, device=images.device
        ),
        "boxes_cxcywh": boxes[:, :, None],
        "box_valid": mask_valid,
        "masks": masks[:, :, None],
        "mask_valid": mask_valid,
        "keypoint_xy": keypoint_xy,
        "keypoint_valid": point_valid,
        "keypoint_visible": batch["keypoint_surface_visible"][:, :, None],
        "keypoint_visibility_valid": point_valid,
    }
    if "keypoint_joint_world_xyz" in batch:
        result["keypoint_world_xyz"] = batch["keypoint_joint_world_xyz"][:, None]
        result["world_valid"] = batch["keypoint_valid"][:, None]
    for key in ("intrinsics", "extrinsics", "camera_available"):
        if key in batch:
            result[key] = batch[key]
    return result


def instance_targets_from_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Use native padded multi-instance labels or adapt a legacy single mouse."""
    if "instance_valid" not in batch:
        return single_mouse_batch_as_instances(batch)
    required = (
        "instance_valid",
        "instance_ids",
        "boxes_cxcywh",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Incomplete multi-instance batch; missing {missing}")
    aliases = {
        "keypoint_xy": ("instance_keypoint_xy", "keypoint_xy"),
        "keypoint_valid": ("instance_keypoint_valid", "keypoint_valid"),
        "keypoint_visible": ("instance_keypoint_visible", "keypoint_visible"),
        "keypoint_visibility_valid": (
            "instance_keypoint_visibility_valid",
            "keypoint_visibility_valid",
        ),
        "keypoint_world_xyz": (
            "instance_keypoint_world_xyz",
            "keypoint_world_xyz",
        ),
        "world_valid": ("instance_world_valid", "world_valid"),
    }
    resolved: dict[str, torch.Tensor] = {
        key: batch[key] for key in required
    }
    for target_key, candidates in aliases.items():
        source_key = next((key for key in candidates if key in batch), None)
        if source_key is not None:
            resolved[target_key] = batch[source_key]
    missing_keypoints = [
        key
        for key in ("keypoint_xy", "keypoint_valid", "keypoint_visible")
        if key not in resolved
    ]
    if missing_keypoints:
        raise KeyError(
            f"Incomplete multi-instance keypoints; missing {missing_keypoints}"
        )
    optional = (
        "box_valid",
        "masks",
        "mask_valid",
        "intrinsics",
        "extrinsics",
        "camera_available",
    )
    resolved.update({key: batch[key] for key in optional if key in batch})
    return resolved


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def generalized_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first[:, None]
    second = second[None]
    intersection_min = torch.maximum(first[..., :2], second[..., :2])
    intersection_max = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp_min(1e-8)
    enclosing_min = torch.minimum(first[..., :2], second[..., :2])
    enclosing_max = torch.maximum(first[..., 2:], second[..., 2:])
    enclosing = (enclosing_max - enclosing_min).clamp_min(0).prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-8)


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular Hungarian algorithm; rows are targets and columns are queries."""
    values = cost.detach().double().cpu()
    rows, columns = values.shape
    if rows > columns:
        raise ValueError("Hungarian matcher requires at least as many queries as targets")
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_value = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = float(values[row0 - 1, column - 1]) - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    target_for_query = [(p[column] - 1, column - 1) for column in range(1, columns + 1) if p[column]]
    target_for_query.sort()
    target_indices = torch.tensor(
        [entry[0] for entry in target_for_query], dtype=torch.long, device=cost.device
    )
    prediction_indices = torch.tensor(
        [entry[1] for entry in target_for_query], dtype=torch.long, device=cost.device
    )
    return prediction_indices, target_indices


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid().flatten(1)
    target = target.flatten(1)
    numerator = 2.0 * (probability * target).sum(dim=1) + 1.0
    denominator = probability.sum(dim=1) + target.sum(dim=1) + 1.0
    return (1.0 - numerator / denominator).mean()


def _local_keypoint_heatmap_loss(
    logits: torch.Tensor,
    coarse_xy: torch.Tensor,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
    radius: float = 0.10,
) -> torch.Tensor:
    size = logits.shape[-1]
    relative = (target_xy - coarse_xy.detach()) / max(radius, 1e-6)
    grid_xy = ((relative + 1.0) * 0.5 * (size - 1)).round().long()
    inside = valid & (relative.abs() <= 1.0).all(dim=-1)
    if not inside.any():
        return logits.mean() * 0.0
    grid_xy = grid_xy.clamp(0, size - 1)
    target_index = grid_xy[..., 1] * size + grid_xy[..., 0]
    return F.cross_entropy(logits[inside].flatten(1), target_index[inside])


def _matched_multiview_geometry_losses(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    matches: list[list[tuple[torch.Tensor, torch.Tensor]]],
    *,
    image_size: tuple[int, int] | None,
    world_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Triangulate and supervise each persistent instance independently.

    Query indices may change between views. Hungarian matches plus persistent
    ``instance_ids`` establish the correspondence before any rays are mixed.
    The 3D and reprojection terms are averaged per instance first so an animal
    with more visible points cannot dominate another animal in the same frame.
    """
    predicted_xy = predictions["instance_keypoint_xy"]
    zero = predicted_xy.mean() * 0.0
    required = (
        "keypoint_world_xyz",
        "world_valid",
        "intrinsics",
        "extrinsics",
        "camera_available",
        "instance_ids",
    )
    if image_size is None or any(key not in targets for key in required):
        count = zero.detach()
        return zero, zero, {
            "instance_3d_points": count,
            "instance_reprojection_points": count,
            "instance_geometry_instances": count,
        }

    height, width = image_size
    if height <= 0 or width <= 0 or world_scale <= 0:
        raise ValueError(
            f"Invalid instance geometry scale: image_size={image_size}, "
            f"world_scale={world_scale}"
        )
    intrinsics = targets["intrinsics"].to(dtype=predicted_xy.dtype)
    extrinsics = targets["extrinsics"].to(dtype=predicted_xy.dtype)
    camera_available = targets["camera_available"].bool()
    target_world = targets["keypoint_world_xyz"].to(dtype=predicted_xy.dtype)
    world_valid = targets["world_valid"].bool()
    target_xy = targets["keypoint_xy"].to(dtype=predicted_xy.dtype)
    point_valid = targets["keypoint_valid"].bool()
    instance_ids = targets["instance_ids"].long()
    keypoint_count = predicted_xy.shape[-2]

    instance_3d_terms: list[torch.Tensor] = []
    instance_reprojection_terms: list[torch.Tensor] = []
    triangulated_count = 0
    reprojected_count = 0
    geometry_instance_count = 0
    pixel_scale = predicted_xy.new_tensor((float(width), float(height)))

    for batch_index, view_matches in enumerate(matches):
        # identity -> view -> (prediction query, target slot)
        identity_matches: dict[int, dict[int, tuple[int, int]]] = {}
        for view_index, (prediction_indices, target_indices) in enumerate(view_matches):
            for prediction_index, target_index in zip(
                prediction_indices.tolist(), target_indices.tolist(), strict=True
            ):
                identity = int(instance_ids[batch_index, view_index, target_index])
                if identity < 0:
                    continue
                identity_matches.setdefault(identity, {})[view_index] = (
                    prediction_index,
                    target_index,
                )

        for view_lookup in identity_matches.values():
            if len(view_lookup) < 2:
                continue
            world_slot = next(iter(view_lookup.values()))[1]
            point_3d_terms: list[torch.Tensor] = []
            point_reprojection_terms: list[torch.Tensor] = []
            for keypoint_index in range(keypoint_count):
                observations: list[tuple[int, int, int]] = []
                for view_index, (prediction_index, target_index) in view_lookup.items():
                    if (
                        camera_available[batch_index, view_index]
                        and point_valid[
                            batch_index, view_index, target_index, keypoint_index
                        ]
                    ):
                        observations.append(
                            (view_index, prediction_index, target_index)
                        )
                if len(observations) < 2:
                    continue

                rows: list[torch.Tensor] = []
                right_hand_side: list[torch.Tensor] = []
                projection_matrices: list[torch.Tensor] = []
                for view_index, prediction_index, _ in observations:
                    projection = (
                        intrinsics[batch_index, view_index]
                        @ extrinsics[batch_index, view_index, :3, :4]
                    )
                    xy_pixels = (
                        predicted_xy[
                            batch_index,
                            view_index,
                            prediction_index,
                            keypoint_index,
                        ]
                        * pixel_scale
                    )
                    u, v = xy_pixels.unbind(-1)
                    rows.extend(
                        (
                            u * projection[2, :3] - projection[0, :3],
                            v * projection[2, :3] - projection[1, :3],
                        )
                    )
                    right_hand_side.extend(
                        (
                            projection[0, 3] - u * projection[2, 3],
                            projection[1, 3] - v * projection[2, 3],
                        )
                    )
                    projection_matrices.append(projection)
                design = torch.stack(rows)
                rhs = torch.stack(right_hand_side)
                normal = design.transpose(0, 1) @ design
                # Scale-aware Tikhonov regularization prevents degenerate camera
                # layouts from making the differentiable solve singular.
                damping = normal.diagonal().mean().detach().clamp_min(1.0) * 1e-7
                triangulated = torch.linalg.solve(
                    normal + torch.eye(3, device=normal.device, dtype=normal.dtype) * damping,
                    design.transpose(0, 1) @ rhs,
                )
                if not torch.isfinite(triangulated).all():
                    continue
                triangulated_count += 1

                if world_valid[batch_index, world_slot, keypoint_index]:
                    residual = (
                        triangulated
                        - target_world[batch_index, world_slot, keypoint_index]
                    ) / float(world_scale)
                    point_3d_terms.append(
                        F.smooth_l1_loss(
                            residual,
                            torch.zeros_like(residual),
                            beta=0.01,
                        )
                    )

                for projection, (view_index, _, target_index) in zip(
                    projection_matrices, observations, strict=True
                ):
                    homogeneous = (
                        projection[:, :3] @ triangulated + projection[:, 3]
                    )
                    if not torch.isfinite(homogeneous).all():
                        continue
                    reprojection = homogeneous[:2] / homogeneous[2].clamp_min(1e-6)
                    reprojection = reprojection / pixel_scale
                    point_reprojection_terms.append(
                        F.smooth_l1_loss(
                            reprojection,
                            target_xy[
                                batch_index,
                                view_index,
                                target_index,
                                keypoint_index,
                            ],
                            beta=0.01,
                        )
                    )
                    reprojected_count += 1

            if point_3d_terms or point_reprojection_terms:
                geometry_instance_count += 1
            if point_3d_terms:
                instance_3d_terms.append(torch.stack(point_3d_terms).mean())
            if point_reprojection_terms:
                instance_reprojection_terms.append(
                    torch.stack(point_reprojection_terms).mean()
                )

    keypoint_3d_loss = (
        torch.stack(instance_3d_terms).mean() if instance_3d_terms else zero
    )
    reprojection_loss = (
        torch.stack(instance_reprojection_terms).mean()
        if instance_reprojection_terms
        else zero
    )
    return keypoint_3d_loss, reprojection_loss, {
        "instance_3d_points": predicted_xy.new_tensor(float(triangulated_count)),
        "instance_reprojection_points": predicted_xy.new_tensor(
            float(reprojected_count)
        ),
        "instance_geometry_instances": predicted_xy.new_tensor(
            float(geometry_instance_count)
        ),
    }


def _matched_reid_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    matches: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    embeddings = []
    identities = []
    for batch_index, view_matches in enumerate(matches):
        for view_index, (prediction_indices, target_indices) in enumerate(view_matches):
            if prediction_indices.numel() == 0:
                continue
            embeddings.append(
                predictions["instance_reid_embedding"][
                    batch_index, view_index, prediction_indices
                ]
            )
            identity = targets["instance_ids"][batch_index, view_index, target_indices]
            # Identity labels are only comparable inside one batch item.
            identities.append(identity + batch_index * 1_000_000)
    if not embeddings:
        return predictions["instance_reid_embedding"].mean() * 0.0
    embedding = torch.cat(embeddings, dim=0)
    identity = torch.cat(identities, dim=0)
    if embedding.shape[0] < 2:
        return embedding.mean() * 0.0
    similarity = embedding @ embedding.transpose(0, 1)
    upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    same = (identity[:, None] == identity[None, :]) & upper
    different = (identity[:, None] != identity[None, :]) & upper
    same_loss = (1.0 - similarity[same]).mean() if same.any() else similarity.mean() * 0.0
    different_loss = F.relu(similarity[different] - 0.2).mean() if different.any() else similarity.mean() * 0.0
    return same_loss + different_loss


def _boxes_from_masks_or_keypoints(
    masks: torch.Tensor,
    keypoint_xy: torch.Tensor,
    keypoint_valid: torch.Tensor,
) -> torch.Tensor:
    batch, views, height, width = masks.shape
    boxes = masks.new_zeros((batch, views, 4), dtype=torch.float32)
    for batch_index in range(batch):
        for view_index in range(views):
            y, x = torch.where(masks[batch_index, view_index])
            if x.numel():
                x0 = x.min().float() / width
                y0 = y.min().float() / height
                x1 = (x.max().float() + 1.0) / width
                y1 = (y.max().float() + 1.0) / height
            else:
                valid = keypoint_valid[batch_index, view_index]
                if valid.any():
                    points = keypoint_xy[batch_index, view_index, valid]
                    x0, y0 = (points.min(dim=0).values - 0.05).clamp(0.0, 1.0)
                    x1, y1 = (points.max(dim=0).values + 0.05).clamp(0.0, 1.0)
                else:
                    x0 = y0 = torch.tensor(0.25, device=masks.device)
                    x1 = y1 = torch.tensor(0.75, device=masks.device)
            boxes[batch_index, view_index] = torch.stack(
                ((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0)
            )
    return boxes

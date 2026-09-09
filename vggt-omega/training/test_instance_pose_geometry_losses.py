from __future__ import annotations

import torch

from training.instance_pose_losses import InstancePoseLossWeights, instance_pose_loss


def _project(
    world: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    camera = torch.einsum("vij,mkj->vmki", extrinsics[:, :3, :3], world)
    camera = camera + extrinsics[:, None, None, :3, 3]
    homogeneous = torch.einsum("vij,vmkj->vmki", intrinsics, camera)
    pixels = homogeneous[..., :2] / homogeneous[..., 2:3]
    height, width = image_size
    return pixels / pixels.new_tensor((width, height))


def _problem(perturb: float = 0.0):
    image_size = (480, 640)
    views, instances, queries, keypoints = 3, 2, 2, 2
    intrinsics = torch.tensor(
        [[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]]
    ).expand(views, -1, -1).clone()
    extrinsics = torch.eye(4).repeat(views, 1, 1)
    extrinsics[1, 0, 3] = -0.3
    extrinsics[2, 1, 3] = -0.25
    world = torch.tensor(
        [
            [[0.00, 0.00, 2.0], [0.10, 0.05, 2.2]],
            [[0.35, 0.00, 2.5], [0.48, 0.10, 2.8]],
        ]
    )
    target_xy = _project(world, extrinsics, intrinsics, image_size)

    # Query order changes by view; persistent IDs must prevent cross-animal rays.
    query_order = torch.tensor([[0, 1], [1, 0], [0, 1]])
    predicted_xy = torch.stack(
        [target_xy[view, query_order[view]] for view in range(views)], dim=0
    )
    predicted_xy[1, 1, 0, 0] += perturb
    predicted_xy = predicted_xy[None].clone().detach().requires_grad_(True)

    target_boxes = torch.tensor(
        [[[0.35, 0.45, 0.25, 0.25], [0.65, 0.45, 0.25, 0.25]]]
    ).expand(views, -1, -1)
    predicted_boxes = torch.stack(
        [target_boxes[view, query_order[view]] for view in range(views)], dim=0
    )[None]
    objectness = torch.full((1, views, queries), 8.0, requires_grad=True)
    predictions = {
        "instance_objectness_logits": objectness,
        "instance_boxes_cxcywh": predicted_boxes.clone().requires_grad_(True),
        "instance_mask_logits": torch.zeros(1, views, queries, 4, 4),
        "instance_reid_embedding": torch.nn.functional.normalize(
            torch.ones(1, views, queries, 4), dim=-1
        ),
        "instance_keypoint_coarse_xy": predicted_xy,
        "instance_keypoint_xy": predicted_xy,
        "instance_keypoint_visibility_logits": torch.zeros(
            1, views, queries, keypoints
        ),
        "instance_keypoint_local_heatmap_logits": torch.zeros(
            1, views, queries, keypoints, 2, 2
        ),
    }
    targets = {
        "instance_valid": torch.ones(1, views, instances, dtype=torch.bool),
        "instance_ids": torch.arange(instances)[None, None].expand(1, views, -1),
        "boxes_cxcywh": target_boxes[None],
        "box_valid": torch.zeros(1, views, instances, dtype=torch.bool),
        "keypoint_xy": target_xy[None],
        "keypoint_valid": torch.ones(
            1, views, instances, keypoints, dtype=torch.bool
        ),
        "keypoint_visible": torch.zeros(
            1, views, instances, keypoints, dtype=torch.bool
        ),
        "keypoint_visibility_valid": torch.zeros(
            1, views, instances, keypoints, dtype=torch.bool
        ),
        "keypoint_world_xyz": world[None],
        "world_valid": torch.ones(1, instances, keypoints, dtype=torch.bool),
        "intrinsics": intrinsics[None],
        "extrinsics": extrinsics[None],
        "camera_available": torch.ones(1, views, dtype=torch.bool),
    }
    return predictions, targets, predicted_xy, image_size


def test_per_instance_geometry_handles_query_permutations() -> None:
    predictions, targets, _, image_size = _problem()
    weights = InstancePoseLossWeights(
        objectness=0.0,
        box_l1=0.0,
        box_giou=0.0,
        mask_bce=0.0,
        mask_dice=0.0,
        keypoint_l1=0.0,
        keypoint_visibility=0.0,
        local_heatmap=0.0,
        reid=0.0,
        keypoint_3d=1.0,
        reprojection=1.0,
    )
    total, logs = instance_pose_loss(
        predictions, targets, weights, image_size=image_size
    )
    assert logs["instance_geometry_instances"].item() == 2
    assert logs["instance_3d_points"].item() == 4
    assert logs["instance_reprojection_points"].item() == 12
    assert logs["loss_instance_keypoint_3d"].item() < 1e-5
    assert logs["loss_instance_reprojection"].item() < 1e-5
    assert total.item() < 2e-5


def test_per_instance_geometry_is_differentiable() -> None:
    predictions, targets, predicted_xy, image_size = _problem(perturb=0.02)
    weights = InstancePoseLossWeights(
        objectness=0.0,
        box_l1=0.0,
        box_giou=0.0,
        mask_bce=0.0,
        mask_dice=0.0,
        keypoint_l1=0.0,
        keypoint_visibility=0.0,
        local_heatmap=0.0,
        reid=0.0,
        keypoint_3d=1.0,
        reprojection=1.0,
    )
    total, logs = instance_pose_loss(
        predictions, targets, weights, image_size=image_size
    )
    assert logs["loss_instance_keypoint_3d"].item() > 0
    assert logs["loss_instance_reprojection"].item() > 0
    total.backward()
    assert predicted_xy.grad is not None
    assert torch.isfinite(predicted_xy.grad).all()
    assert predicted_xy.grad.abs().sum() > 0

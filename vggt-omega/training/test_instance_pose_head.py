from __future__ import annotations

import torch

from training.instance_pose_losses import (
    _linear_sum_assignment,
    hungarian_match_instances,
    instance_targets_from_batch,
    instance_pose_loss,
)
from vggt_omega.models.heads import MultiInstanceMouseHead


def test_multi_instance_head_shapes_and_gradient() -> None:
    torch.manual_seed(7)
    batch, views, keypoints, queries = 1, 2, 5, 3
    height, width = 64, 96
    patch_count = (height // 16) * (width // 16)
    cached: list[torch.Tensor | None] = [None] * 24
    source_tokens = []
    for layer_index in (4, 11, 17, 23):
        tokens = torch.randn(
            batch,
            views,
            5 + patch_count,
            32,
            requires_grad=True,
        )
        cached[layer_index] = tokens
        source_tokens.append(tokens)
    images = torch.randn(batch, views, 3, height, width)
    head = MultiInstanceMouseHead(
        num_keypoints=keypoints,
        num_queries=queries,
        dim_in=32,
        hidden_dim=32,
        num_heads=4,
        local_heatmap_size=6,
    )

    output = head(cached, images, patch_token_start=5)
    assert output["instance_objectness_logits"].shape == (batch, views, queries)
    assert output["instance_boxes_cxcywh"].shape == (batch, views, queries, 4)
    assert output["instance_mask_logits"].shape == (
        batch,
        views,
        queries,
        height // 4,
        width // 4,
    )
    assert output["instance_keypoint_xy"].shape == (
        batch,
        views,
        queries,
        keypoints,
        2,
    )
    assert output["instance_keypoint_local_heatmap_logits"].shape == (
        batch,
        views,
        queries,
        keypoints,
        6,
        6,
    )
    assert torch.all((output["instance_keypoint_xy"] >= 0.0))
    assert torch.all((output["instance_keypoint_xy"] <= 1.0))
    loss = sum(value.float().square().mean() for value in output.values())
    loss.backward()
    for tokens in source_tokens:
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()


def test_hungarian_algorithm_finds_rectangular_optimum() -> None:
    cost = torch.tensor(
        [
            [8.0, 1.0, 9.0, 7.0],
            [2.0, 8.0, 3.0, 9.0],
        ]
    )
    prediction, target = _linear_sum_assignment(cost)
    assert target.tolist() == [0, 1]
    assert prediction.tolist() == [1, 0]


def test_native_multi_instance_batch_is_not_collapsed() -> None:
    batch = {
        "instance_valid": torch.ones(1, 2, 3, dtype=torch.bool),
        "instance_ids": torch.arange(3).view(1, 1, 3).expand(1, 2, 3),
        "boxes_cxcywh": torch.rand(1, 2, 3, 4),
        "keypoint_xy": torch.rand(1, 2, 3, 5, 2),
        "keypoint_valid": torch.ones(1, 2, 3, 5, dtype=torch.bool),
        "keypoint_visible": torch.ones(1, 2, 3, 5, dtype=torch.bool),
    }
    targets = instance_targets_from_batch(batch)
    assert targets["instance_valid"].shape == (1, 2, 3)
    assert targets["keypoint_xy"].shape == (1, 2, 3, 5, 2)


def test_blendedmvs_instance_aliases_preserve_legacy_keypoint_fields() -> None:
    batch = {
        "instance_valid": torch.ones(1, 2, 2, dtype=torch.bool),
        "instance_ids": torch.tensor([[[0, 1], [0, 1]]]),
        "boxes_cxcywh": torch.rand(1, 2, 2, 4),
        "keypoint_valid": torch.ones(1, 5, dtype=torch.bool),
        "instance_keypoint_xy": torch.rand(1, 2, 2, 5, 2),
        "instance_keypoint_valid": torch.ones(1, 2, 2, 5, dtype=torch.bool),
        "instance_keypoint_visible": torch.ones(1, 2, 2, 5, dtype=torch.bool),
        "instance_keypoint_world_xyz": torch.rand(1, 2, 5, 3),
        "instance_world_valid": torch.ones(1, 2, 5, dtype=torch.bool),
    }
    targets = instance_targets_from_batch(batch)
    assert targets["keypoint_xy"].shape == (1, 2, 2, 5, 2)
    assert targets["keypoint_valid"].shape == (1, 2, 2, 5)
    assert targets["keypoint_world_xyz"].shape == (1, 2, 5, 3)
    assert batch["keypoint_valid"].shape == (1, 5)


def test_matching_is_query_permutation_invariant_and_partial_labels_train() -> None:
    torch.manual_seed(11)
    batch, views, queries, instances, keypoints = 1, 2, 4, 2, 3
    objectness = torch.full((batch, views, queries), -5.0, requires_grad=True)
    boxes = torch.rand(batch, views, queries, 4, requires_grad=True)
    keypoint_xy = torch.rand(
        batch, views, queries, keypoints, 2, requires_grad=True
    )
    target_boxes = torch.tensor(
        [[[[0.25, 0.30, 0.20, 0.25], [0.72, 0.65, 0.18, 0.22]]] * views]
    )
    target_xy = torch.rand(batch, views, instances, keypoints, 2)
    with torch.no_grad():
        objectness[:, :, 1] = 6.0
        objectness[:, :, 3] = 6.0
        boxes[:, :, 1] = target_boxes[:, :, 0]
        boxes[:, :, 3] = target_boxes[:, :, 1]
        keypoint_xy[:, :, 1] = target_xy[:, :, 0]
        keypoint_xy[:, :, 3] = target_xy[:, :, 1]
    predictions = {
        "instance_objectness_logits": objectness,
        "instance_boxes_cxcywh": boxes,
        "instance_mask_logits": torch.randn(
            batch, views, queries, 8, 12, requires_grad=True
        ),
        "instance_reid_embedding": torch.nn.functional.normalize(
            torch.randn(batch, views, queries, 8, requires_grad=True), dim=-1
        ),
        "instance_keypoint_coarse_xy": keypoint_xy,
        "instance_keypoint_xy": keypoint_xy,
        "instance_keypoint_visibility_logits": torch.randn(
            batch, views, queries, keypoints, requires_grad=True
        ),
        "instance_keypoint_sigma": torch.ones(batch, views, queries, keypoints),
        "instance_keypoint_local_heatmap_logits": torch.randn(
            batch, views, queries, keypoints, 6, 6, requires_grad=True
        ),
    }
    valid = torch.ones(batch, views, instances, dtype=torch.bool)
    point_valid = torch.ones(batch, views, instances, keypoints, dtype=torch.bool)
    point_valid[..., -1] = False  # partial-label path
    targets = {
        "instance_valid": valid,
        "instance_ids": torch.tensor([[[0, 1], [0, 1]]]),
        "boxes_cxcywh": target_boxes,
        "keypoint_xy": target_xy,
        "keypoint_valid": point_valid,
        "keypoint_visible": point_valid,
        "masks": torch.rand(batch, views, instances, 32, 48) > 0.5,
        "mask_valid": valid,
    }
    matches = hungarian_match_instances(predictions, targets)
    assert matches[0][0][0].tolist() == [1, 3]
    total, logs = instance_pose_loss(predictions, targets)
    assert torch.isfinite(total)
    assert logs["matched_instances"].item() == views * instances
    total.backward()
    assert objectness.grad is not None
    assert boxes.grad is not None
    assert keypoint_xy.grad is not None

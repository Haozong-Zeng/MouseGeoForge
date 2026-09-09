from __future__ import annotations

import torch

from training.keypoint_head_losses import (
    heatmap_soft_argmax,
    triangulate_gaussian_rays,
)
from vggt_omega.models.heads import KeypointHead


def test_keypoint_head_shape_and_gradient() -> None:
    torch.manual_seed(3)
    batch, views, keypoints = 1, 3, 5
    height, width = 64, 96
    patch_tokens = (height // 16) * (width // 16)
    tokens = torch.randn(
        batch,
        views,
        17 + patch_tokens,
        32,
        requires_grad=True,
    )
    cached = [None] * 24
    cached[-1] = tokens
    images = torch.randn(batch, views, 3, height, width)
    head = KeypointHead(
        num_keypoints=keypoints,
        dim_in=32,
        hidden_dim=32,
    )

    heatmaps, visibility, sigma = head(cached, images, patch_token_start=17)
    assert heatmaps.shape == (batch, views, keypoints, height // 4, width // 4)
    assert visibility.shape == (batch, views, keypoints)
    assert sigma.shape == (batch, views, keypoints)
    assert torch.all(sigma > 1.0)

    loss = heatmaps.square().mean() + visibility.square().mean() + sigma.mean()
    loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_soft_argmax_uses_full_resolution_coordinates() -> None:
    logits = torch.full((1, 1, 1, 4, 6), -20.0)
    logits[0, 0, 0, 2, 3] = 20.0
    xy, peak = heatmap_soft_argmax(logits, image_size=(16, 24))
    torch.testing.assert_close(xy[0, 0, 0], torch.tensor([13.5, 9.5]))
    assert peak.item() > 0.99


def test_gaussian_ray_fusion_recovers_known_world_point() -> None:
    target = torch.tensor([0.0, 0.0, 5.0])
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    extrinsics[0, 0, 0, 3] = 1.0
    extrinsics[0, 1, 0, 3] = -1.0
    intrinsics = torch.tensor(
        [[[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]]
    ).repeat(1, 2, 1, 1)
    xy = torch.tensor([[[[70.0, 50.0]], [[30.0, 50.0]]]])
    visibility = torch.full((1, 2, 1), 8.0)
    sigma = torch.full((1, 2, 1), 4.0)
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    center, covariance, view_count = triangulate_gaussian_rays(
        xy,
        visibility,
        sigma,
        extrinsics,
        intrinsics,
        valid,
    )
    torch.testing.assert_close(center[0, 0], target, atol=2e-3, rtol=2e-3)
    assert torch.linalg.eigvalsh(covariance[0, 0]).min() > 0
    assert view_count.item() == 2


def test_robust_gaussian_fusion_downweights_third_view_outlier() -> None:
    target = torch.tensor([0.0, 0.0, 5.0])
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    extrinsics[0, 0, 0, 3] = 1.0
    extrinsics[0, 1, 0, 3] = -1.0
    extrinsics[0, 2, 1, 3] = 1.0
    intrinsics = torch.tensor(
        [[[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]]
    ).repeat(1, 3, 1, 1)
    xy = torch.tensor(
        [[[[70.0, 50.0]], [[30.0, 50.0]], [[88.0, 18.0]]]]
    )
    visibility = torch.full((1, 3, 1), 8.0)
    sigma = torch.full((1, 3, 1), 4.0)
    valid = torch.ones((1, 3, 1), dtype=torch.bool)

    plain, _, _ = triangulate_gaussian_rays(
        xy,
        visibility,
        sigma,
        extrinsics,
        intrinsics,
        valid,
        robust_iterations=0,
    )
    robust, _, _ = triangulate_gaussian_rays(
        xy,
        visibility,
        sigma,
        extrinsics,
        intrinsics,
        valid,
        robust_iterations=3,
    )
    plain_error = torch.linalg.vector_norm(plain[0, 0] - target)
    robust_error = torch.linalg.vector_norm(robust[0, 0] - target)
    assert robust_error < plain_error

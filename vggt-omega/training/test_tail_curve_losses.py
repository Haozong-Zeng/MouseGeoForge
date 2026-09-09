from __future__ import annotations

import torch

from training.tail_curve_losses import (
    TAIL_KEYPOINT_NAMES,
    TailCurveLossConfig,
    curve_distance,
    fit_weighted_bspline,
    one_sided_soft_curve_distance,
    resample_curve_by_arclength,
    resolve_tail_indices,
    robust_view_reduce,
    strict_point_weights,
    tail_curve_loss,
)


def test_tail_indices_and_strict_interior_ablation_weights() -> None:
    names = ["nose", *TAIL_KEYPOINT_NAMES, "left_ear"]
    assert resolve_tail_indices(names) == tuple(range(1, 8))
    weights = strict_point_weights(
        names,
        0.2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(weights[2:7], torch.full((5,), 0.2))
    assert weights[1].item() == 1.0
    assert weights[7].item() == 1.0
    assert weights[0].item() == 1.0


def test_weighted_bspline_is_differentiable_and_ignores_invalid_outlier() -> None:
    parameter = torch.linspace(0.0, 1.0, 7)
    clean = torch.stack([parameter, parameter.square()], dim=-1)
    points = clean.clone()
    points[3] = torch.tensor([4.0, -3.0])
    points.requires_grad_(True)
    weights = torch.ones(7)
    weights[3] = 0.0

    curve, valid = fit_weighted_bspline(
        points,
        weights,
        samples=31,
        smoothness=1e-2,
        min_points=4,
    )
    clean_curve, _ = fit_weighted_bspline(
        clean,
        torch.ones(7),
        samples=31,
        smoothness=1e-2,
        min_points=4,
    )
    assert valid.item()
    assert torch.linalg.vector_norm(curve - clean_curve, dim=-1).mean() < 0.08
    curve.square().mean().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.all(points.grad[3] == 0)


def test_soft_dtw_is_reparameterization_tolerant_and_direction_sensitive() -> None:
    target_parameter = torch.linspace(0.0, 1.0, 32)
    shifted_parameter = target_parameter.pow(1.7)
    target = torch.stack([target_parameter, target_parameter.square()], dim=-1)[None]
    shifted = torch.stack([shifted_parameter, shifted_parameter.square()], dim=-1)[None]
    reversed_curve = target.flip(dims=[1])

    ordered_shifted = curve_distance(
        shifted,
        target,
        mode="ordered",
        soft_dtw_gamma=0.02,
    )
    soft_shifted = curve_distance(
        shifted,
        target,
        mode="soft_dtw",
        soft_dtw_gamma=0.02,
    )
    soft_reversed = curve_distance(
        reversed_curve,
        target,
        mode="soft_dtw",
        soft_dtw_gamma=0.02,
    )
    assert soft_shifted.item() < soft_reversed.item()
    assert soft_shifted.item() < ordered_shifted.item()


def test_arclength_resampling_equalizes_nonuniform_segments() -> None:
    curve = torch.tensor([[[0.0, 0.0], [0.01, 0.0], [0.1, 0.0], [1.0, 0.0]]])
    resampled = resample_curve_by_arclength(curve, 6)
    segment_lengths = torch.linalg.vector_norm(resampled[:, 1:] - resampled[:, :-1], dim=-1)
    torch.testing.assert_close(segment_lengths, torch.full_like(segment_lengths, 0.2), atol=1e-5, rtol=1e-5)


def test_visible_fragment_uses_one_sided_curve_distance() -> None:
    parameter = torch.linspace(0.0, 1.0, 32)
    curve = torch.stack([parameter, 0.2 * torch.sin(parameter * torch.pi)], dim=-1)[None]
    queries = curve[:, ::4].clone()
    valid = torch.zeros((1, queries.shape[1]), dtype=torch.bool)
    valid[:, 2:6] = True
    close, close_valid = one_sided_soft_curve_distance(
        queries,
        valid,
        curve,
        temperature=1e-3,
    )
    far, _ = one_sided_soft_curve_distance(
        queries,
        valid,
        curve + torch.tensor([0.0, 0.5]),
        temperature=1e-3,
    )
    assert close_valid.item()
    assert close.item() < far.item()


def test_robust_view_reduction_downweights_single_bad_view() -> None:
    residual = torch.tensor([[0.10, 0.11, 0.09, 4.0]])
    valid = torch.ones_like(residual, dtype=torch.bool)
    robust, weights = robust_view_reduce(
        residual,
        valid,
        enabled=True,
        delta=2.5,
    )
    plain, _ = robust_view_reduce(
        residual,
        valid,
        enabled=False,
        delta=2.5,
    )
    assert weights[0, -1] < weights[0, :-1].min()
    assert robust < plain


def test_complete_tail_loss_runs_for_2d_and_both_3d_rigs() -> None:
    parameter = torch.linspace(0.0, 1.0, 7)
    target_world = torch.stack(
        [parameter, 0.1 * torch.sin(parameter * torch.pi), torch.zeros_like(parameter)],
        dim=-1,
    )[None]
    base_xy = torch.stack([30.0 + 120.0 * parameter, 80.0 + 20.0 * parameter], dim=-1)
    target_xy = torch.stack(
        [base_xy + torch.tensor([10.0 * view, 3.0 * view]) for view in range(4)],
        dim=0,
    )[None]
    predicted_xy = target_xy.clone().requires_grad_(True)
    fixed_center = target_world.clone().requires_grad_(True)
    predicted_center = target_world.clone().requires_grad_(True)
    valid_2d = torch.ones((1, 4, 7), dtype=torch.bool)
    valid_3d = torch.ones((1, 7), dtype=torch.bool)
    view_count = torch.full((1, 7), 4)

    loss, logs = tail_curve_loss(
        keypoint_names=list(TAIL_KEYPOINT_NAMES),
        predicted_xy=predicted_xy,
        target_xy=target_xy,
        visibility_logits=torch.full((1, 4, 7), 6.0),
        pixel_sigma=torch.full((1, 4, 7), 3.0),
        fixed_center=fixed_center,
        predicted_center=predicted_center,
        target_world=target_world,
        point_valid=valid_3d,
        view_valid=valid_2d,
        surface_visible=valid_2d,
        fixed_view_count=view_count,
        predicted_view_count=view_count,
        image_size=(200, 240),
        fixed_rig_weight=1.0,
        predicted_rig_weight=0.25,
        config=TailCurveLossConfig(weight=1.0),
    )
    assert torch.isfinite(loss)
    assert logs["keypoint_tail_valid_2d_views"].item() == 4
    assert logs["keypoint_tail_valid_3d_fixed"].item() == 1
    loss.backward()
    assert predicted_xy.grad is not None and torch.isfinite(predicted_xy.grad).all()
    assert fixed_center.grad is not None and torch.isfinite(fixed_center.grad).all()
    assert predicted_center.grad is not None and torch.isfinite(predicted_center.grad).all()

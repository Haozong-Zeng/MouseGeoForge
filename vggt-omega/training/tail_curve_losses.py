"""Differentiable tail-curve supervision for ordered mouse keypoints.

The module is deliberately independent from the VGGT model and trainer so each
curve component can be unit-tested and ablated in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


TAIL_KEYPOINT_NAMES = (
    "tail_base",
    "tail_q1",
    "tail_q2",
    "tail_q3",
    "tail_q4",
    "tail_q5",
    "tail_tip",
)


@dataclass(frozen=True)
class TailCurveLossConfig:
    """Weights and numerical choices for a tail-curve ablation."""

    weight: float = 0.0
    curve_2d_weight: float = 1.0
    curve_3d_weight: float = 1.0
    partial_2d_weight: float = 0.25
    base_anchor_weight: float = 0.5
    tip_anchor_weight: float = 0.25
    length_weight: float = 0.1
    curvature_weight: float = 0.05
    interior_point_weight: float = 1.0
    samples: int = 32
    smoothness: float = 1e-2
    min_points: int = 4
    distance: str = "ordered"
    soft_dtw_gamma: float = 0.05
    partial_temperature: float = 0.01
    robust_views: bool = True
    robust_delta: float = 2.5
    detach_fit_weights: bool = True


def resolve_tail_indices(
    keypoint_names: Sequence[str],
    tail_names: Sequence[str] = TAIL_KEYPOINT_NAMES,
) -> tuple[int, ...] | None:
    """Return ordered tail indices, or None when the full definition is absent."""
    index = {name: i for i, name in enumerate(keypoint_names)}
    if any(name not in index for name in tail_names):
        return None
    return tuple(index[name] for name in tail_names)


def strict_point_weights(
    keypoint_names: Sequence[str],
    interior_weight: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build per-channel weights for legacy pointwise losses.

    The base and tip remain at one. Only q1..q5 are weakened, which keeps the
    tail attached and preserves its endpoint while allowing curve supervision
    to absorb uncertainty in interior landmark parameterization.
    """
    weights = torch.ones(len(keypoint_names), dtype=dtype, device=device)
    indices = resolve_tail_indices(keypoint_names)
    if indices is not None:
        weights[list(indices[1:-1])] = max(float(interior_weight), 0.0)
    return weights


def _open_uniform_bspline_basis(
    parameters: torch.Tensor,
    num_controls: int,
    degree: int = 3,
) -> torch.Tensor:
    """Evaluate a clamped open-uniform B-spline basis."""
    degree = min(int(degree), num_controls - 1)
    interior_count = num_controls - degree - 1
    pieces = [parameters.new_zeros(degree + 1)]
    if interior_count > 0:
        pieces.append(
            torch.arange(
                1,
                interior_count + 1,
                dtype=parameters.dtype,
                device=parameters.device,
            )
            / float(interior_count + 1)
        )
    pieces.append(parameters.new_ones(degree + 1))
    knots = torch.cat(pieces)

    basis = torch.stack(
        [
            ((parameters >= knots[i]) & (parameters < knots[i + 1])).to(parameters.dtype)
            for i in range(knots.numel() - 1)
        ],
        dim=-1,
    )
    for current_degree in range(1, degree + 1):
        next_basis = []
        for i in range(basis.shape[-1] - 1):
            left_denominator = knots[i + current_degree] - knots[i]
            right_denominator = knots[i + current_degree + 1] - knots[i + 1]
            left = basis[..., i] * (parameters - knots[i]) / left_denominator.clamp_min(1e-12)
            right = basis[..., i + 1] * (knots[i + current_degree + 1] - parameters) / right_denominator.clamp_min(1e-12)
            left = torch.where(left_denominator > 0, left, torch.zeros_like(left))
            right = torch.where(right_denominator > 0, right, torch.zeros_like(right))
            next_basis.append(left + right)
        basis = torch.stack(next_basis, dim=-1)

    endpoint = parameters >= 1.0 - 1e-7
    if endpoint.any():
        basis = torch.where(endpoint[..., None], torch.zeros_like(basis), basis)
        basis[..., -1] = torch.where(endpoint, torch.ones_like(parameters), basis[..., -1])
    return basis


def fit_weighted_bspline(
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    samples: int = 32,
    smoothness: float = 1e-2,
    min_points: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a regularized cubic B-spline to ordered points.

    Args:
        points: ``[..., K, D]`` ordered coordinates.
        weights: ``[..., K]`` non-negative confidence/validity weights.
    Returns:
        Sampled curves ``[..., samples, D]`` and a validity mask ``[...]``.
    """
    if points.ndim < 2 or points.shape[-2] < 2:
        raise ValueError("points must have shape [..., K, D] with K >= 2")
    if weights.shape != points.shape[:-1]:
        raise ValueError("weights must match points.shape[:-1]")
    if samples < 2:
        raise ValueError("samples must be at least 2")

    keypoints = points.shape[-2]
    leading_shape = points.shape[:-2]
    flat_points = torch.nan_to_num(points).reshape(-1, keypoints, points.shape[-1])
    flat_weights = torch.nan_to_num(weights).clamp_min(0.0).reshape(-1, keypoints)
    valid = (flat_weights > 0).sum(dim=-1) >= max(min(int(min_points), keypoints), 2)
    flat_weights = flat_weights / flat_weights.amax(dim=-1, keepdim=True).clamp_min(1e-8)

    fit_parameters = torch.linspace(
        0.0,
        1.0,
        keypoints,
        dtype=points.dtype,
        device=points.device,
    )
    sample_parameters = torch.linspace(
        0.0,
        1.0,
        int(samples),
        dtype=points.dtype,
        device=points.device,
    )
    fit_basis = _open_uniform_bspline_basis(fit_parameters, keypoints)
    sample_basis = _open_uniform_bspline_basis(sample_parameters, keypoints)
    weighted_basis = fit_basis[None, :, :] * flat_weights[..., None]
    gram = torch.einsum("ki,nkj->nij", fit_basis, weighted_basis)
    right_hand = torch.einsum(
        "ki,nk,nkd->nid",
        fit_basis,
        flat_weights,
        flat_points,
    )

    second_difference = points.new_zeros((max(keypoints - 2, 1), keypoints))
    if keypoints > 2:
        row = torch.arange(keypoints - 2, device=points.device)
        second_difference[row, row] = 1.0
        second_difference[row, row + 1] = -2.0
        second_difference[row, row + 2] = 1.0
    regularizer = second_difference.transpose(0, 1) @ second_difference
    identity = torch.eye(keypoints, dtype=points.dtype, device=points.device)
    system = gram + max(float(smoothness), 0.0) * regularizer + 1e-5 * identity
    controls = torch.linalg.solve(system, right_hand)
    curves = torch.einsum("sk,nkd->nsd", sample_basis, controls)
    return curves.reshape(*leading_shape, samples, points.shape[-1]), valid.reshape(leading_shape)


def curve_length(curves: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(curves[..., 1:, :] - curves[..., :-1, :], dim=-1).sum(dim=-1)


def resample_curve_by_arclength(
    curves: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    """Resample dense curves at uniform normalized arc-length positions."""
    if curves.shape[-2] < 2 or samples < 2:
        raise ValueError("curves and requested output must contain at least two samples")
    leading_shape = curves.shape[:-2]
    dense_samples = curves.shape[-2]
    dimensions = curves.shape[-1]
    flat = curves.reshape(-1, dense_samples, dimensions)
    segment_length = torch.linalg.vector_norm(flat[:, 1:] - flat[:, :-1], dim=-1)
    cumulative = torch.cat([segment_length.new_zeros((flat.shape[0], 1)), segment_length.cumsum(dim=-1)], dim=-1)
    cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-8)
    target = torch.linspace(
        0.0,
        1.0,
        int(samples),
        dtype=curves.dtype,
        device=curves.device,
    )[None, :].expand(flat.shape[0], -1)
    upper = torch.searchsorted(cumulative.detach().contiguous(), target.contiguous(), right=True)
    upper = upper.clamp(min=1, max=dense_samples - 1)
    lower = upper - 1
    gather_index_lower = lower[..., None].expand(-1, -1, dimensions)
    gather_index_upper = upper[..., None].expand(-1, -1, dimensions)
    point_lower = torch.gather(flat, 1, gather_index_lower)
    point_upper = torch.gather(flat, 1, gather_index_upper)
    length_lower = torch.gather(cumulative, 1, lower)
    length_upper = torch.gather(cumulative, 1, upper)
    alpha = (target - length_lower) / (length_upper - length_lower).clamp_min(1e-8)
    resampled = point_lower + alpha[..., None] * (point_upper - point_lower)
    return resampled.reshape(*leading_shape, samples, dimensions)


def soft_dtw_cost(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Batched differentiable soft-DTW cost for ``[N, S, D]`` curves."""
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0]:
        raise ValueError("soft_dtw_cost expects x/y shaped [N, S, D]")
    gamma = max(float(gamma), 1e-6)
    pairwise_cost = torch.cdist(x, y).square()
    infinity = torch.full(
        (x.shape[0],),
        torch.inf,
        dtype=x.dtype,
        device=x.device,
    )
    previous = [x.new_zeros(x.shape[0])] + [infinity] * y.shape[1]
    for i in range(x.shape[1]):
        current = [infinity]
        for j in range(y.shape[1]):
            predecessors = torch.stack(
                [previous[j + 1], current[j], previous[j]],
                dim=-1,
            )
            soft_minimum = -gamma * torch.logsumexp(-predecessors / gamma, dim=-1)
            current.append(pairwise_cost[:, i, j] + soft_minimum)
        previous = current
    return previous[-1]


def soft_dtw_divergence(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    xy = soft_dtw_cost(x, y, gamma)
    xx = soft_dtw_cost(x, x, gamma)
    yy = soft_dtw_cost(y, y, gamma)
    return (xy - 0.5 * (xx + yy)).clamp_min(0.0)


def curve_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    soft_dtw_gamma: float,
) -> torch.Tensor:
    """Return one ordered/symmetric curve residual per flattened curve."""
    if mode == "ordered":
        return F.smooth_l1_loss(predicted, target, reduction="none", beta=0.02).mean(dim=(-2, -1))
    if mode == "chamfer":
        pairwise = torch.cdist(predicted, target).square()
        return 0.5 * (pairwise.min(dim=-1).values.mean(dim=-1) + pairwise.min(dim=-2).values.mean(dim=-1))
    if mode == "soft_dtw":
        return soft_dtw_divergence(predicted, target, soft_dtw_gamma)
    raise ValueError(f"Unknown tail curve distance: {mode}")


def one_sided_soft_curve_distance(
    query_points: torch.Tensor,
    query_valid: torch.Tensor,
    curves: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance from visible target points to a complete predicted curve."""
    temperature = max(float(temperature), 1e-6)
    squared_distance = torch.cdist(query_points, curves).square()
    soft_minimum = -temperature * torch.logsumexp(-squared_distance / temperature, dim=-1)
    soft_minimum = soft_minimum + temperature * torch.log(
        torch.tensor(curves.shape[-2], dtype=curves.dtype, device=curves.device)
    )
    weights = query_valid.to(dtype=curves.dtype)
    residual = (soft_minimum.clamp_min(0.0) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    return residual, query_valid.any(dim=-1)


def robust_view_reduce(
    residual: torch.Tensor,
    valid: torch.Tensor,
    *,
    enabled: bool,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[B, V]`` residuals with detached median/MAD view weights."""
    base_weights = valid.to(dtype=residual.dtype)
    robust_weights = torch.ones_like(residual)
    if enabled:
        with torch.no_grad():
            for batch_index in range(residual.shape[0]):
                sample_valid = valid[batch_index]
                if int(sample_valid.sum()) < 3:
                    continue
                values = residual[batch_index, sample_valid]
                median = values.median()
                mad = (values - median).abs().median().clamp_min(1e-6)
                normalized = (residual[batch_index] - median).abs() / (1.4826 * mad)
                robust_weights[batch_index] = 1.0 / (
                    1.0 + (normalized / max(float(delta), 1e-3)).square()
                )
    weights = base_weights * robust_weights
    reduced = (residual * weights).sum() / weights.sum().clamp_min(1.0)
    return reduced, weights


def _curve_components(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    distance_mode: str,
    soft_dtw_gamma: float,
) -> dict[str, torch.Tensor]:
    predicted_length = curve_length(predicted).clamp_min(1e-6)
    target_length = curve_length(target).clamp_min(1e-6)
    predicted_shape = (predicted - predicted[..., :1, :]) / predicted_length[..., None, None]
    target_shape = (target - target[..., :1, :]) / target_length[..., None, None]
    shape = curve_distance(
        predicted_shape.reshape(-1, predicted.shape[-2], predicted.shape[-1]),
        target_shape.reshape(-1, target.shape[-2], target.shape[-1]),
        mode=distance_mode,
        soft_dtw_gamma=soft_dtw_gamma,
    ).reshape(predicted.shape[:-2])
    length = (predicted_length.log() - target_length.log()).abs()
    predicted_curvature = predicted_shape[..., 2:, :] - 2.0 * predicted_shape[..., 1:-1, :] + predicted_shape[..., :-2, :]
    target_curvature = target_shape[..., 2:, :] - 2.0 * target_shape[..., 1:-1, :] + target_shape[..., :-2, :]
    curvature = F.smooth_l1_loss(
        predicted_curvature,
        target_curvature,
        reduction="none",
        beta=0.01,
    ).mean(dim=(-2, -1))
    return {"shape": shape, "length": length, "curvature": curvature}


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _fit_pair(
    predicted: torch.Tensor,
    target: torch.Tensor,
    predicted_weights: torch.Tensor,
    target_weights: torch.Tensor,
    config: TailCurveLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_samples = max(int(config.samples) * 4, 64)
    predicted_curve, predicted_valid = fit_weighted_bspline(
        predicted,
        predicted_weights,
        samples=dense_samples,
        smoothness=config.smoothness,
        min_points=config.min_points,
    )
    target_curve, target_valid = fit_weighted_bspline(
        target,
        target_weights,
        samples=dense_samples,
        smoothness=config.smoothness,
        min_points=config.min_points,
    )
    predicted_curve = resample_curve_by_arclength(predicted_curve, config.samples)
    target_curve = resample_curve_by_arclength(target_curve, config.samples)
    return predicted_curve, target_curve, predicted_valid & target_valid


def _spatial_curve_loss(
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    predicted_weights: torch.Tensor,
    target_weights: torch.Tensor,
    config: TailCurveLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    predicted_curve, target_curve, valid = _fit_pair(
        predicted_points,
        target_points,
        predicted_weights,
        target_weights,
        config,
    )
    components = _curve_components(
        predicted_curve,
        target_curve,
        distance_mode=config.distance,
        soft_dtw_gamma=config.soft_dtw_gamma,
    )
    base = torch.linalg.vector_norm(predicted_points[..., 0, :] - target_points[..., 0, :], dim=-1)
    tip = torch.linalg.vector_norm(predicted_points[..., -1, :] - target_points[..., -1, :], dim=-1)
    valid_base = (predicted_weights[..., 0] > 0) & (target_weights[..., 0] > 0)
    valid_tip = (predicted_weights[..., -1] > 0) & (target_weights[..., -1] > 0)
    total_per_curve = (
        components["shape"]
        + config.length_weight * components["length"]
        + config.curvature_weight * components["curvature"]
        + config.base_anchor_weight * base * valid_base.to(base.dtype)
        + config.tip_anchor_weight * tip * valid_tip.to(tip.dtype)
    )
    logs = {
        "shape": components["shape"],
        "length": components["length"],
        "curvature": components["curvature"],
        "base": base,
        "tip": tip,
        "valid_base": valid_base,
        "valid_tip": valid_tip,
    }
    return total_per_curve, logs, valid, predicted_curve


def tail_curve_loss(
    *,
    keypoint_names: Sequence[str],
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    visibility_logits: torch.Tensor,
    pixel_sigma: torch.Tensor,
    fixed_center: torch.Tensor,
    predicted_center: torch.Tensor,
    target_world: torch.Tensor,
    point_valid: torch.Tensor,
    view_valid: torch.Tensor,
    surface_visible: torch.Tensor,
    fixed_view_count: torch.Tensor,
    predicted_view_count: torch.Tensor,
    image_size: tuple[int, int],
    fixed_rig_weight: float,
    predicted_rig_weight: float,
    config: TailCurveLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute modular 2D, partial-visibility, and fused-3D tail losses."""
    zero = predicted_xy.mean() * 0.0
    indices = resolve_tail_indices(keypoint_names)
    if config.weight <= 0 or indices is None:
        return zero, {"loss_keypoint_tail_curve": zero}
    index = torch.tensor(indices, dtype=torch.long, device=predicted_xy.device)

    predicted_xy = predicted_xy.index_select(-2, index)
    target_xy = target_xy.index_select(-2, index)
    visibility_logits = visibility_logits.index_select(-1, index)
    pixel_sigma = pixel_sigma.index_select(-1, index)
    view_valid = view_valid.index_select(-1, index)
    surface_visible = surface_visible.index_select(-1, index) & view_valid
    point_valid = point_valid.index_select(-1, index)
    fixed_center = fixed_center.index_select(-2, index)
    predicted_center = predicted_center.index_select(-2, index)
    target_world = target_world.index_select(-2, index)
    fixed_view_count = fixed_view_count.index_select(-1, index)
    predicted_view_count = predicted_view_count.index_select(-1, index)

    image_height, image_width = image_size
    image_scale = predicted_xy.new_tensor([image_width, image_height])
    predicted_xy_normalized = predicted_xy / image_scale
    target_xy_normalized = target_xy / image_scale
    confidence = torch.sigmoid(visibility_logits) / pixel_sigma.clamp_min(1.0).square()
    if config.detach_fit_weights:
        confidence = confidence.detach()
    predicted_2d_weights = confidence * view_valid.to(confidence.dtype)
    target_2d_weights = view_valid.to(confidence.dtype)
    loss_2d_per_view, logs_2d, valid_2d, predicted_2d_curve = _spatial_curve_loss(
        predicted_xy_normalized,
        target_xy_normalized,
        predicted_2d_weights,
        target_2d_weights,
        config,
    )

    partial_per_view, partial_valid = one_sided_soft_curve_distance(
        target_xy_normalized,
        surface_visible,
        predicted_2d_curve,
        config.partial_temperature,
    )
    loss_2d_per_view = loss_2d_per_view + config.partial_2d_weight * partial_per_view
    loss_2d, view_weights = robust_view_reduce(
        loss_2d_per_view,
        valid_2d,
        enabled=config.robust_views,
        delta=config.robust_delta,
    )
    partial_2d = _masked_mean(partial_per_view, partial_valid & valid_2d)

    def fused_loss(
        centers: torch.Tensor,
        view_count: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        center_valid = point_valid & (view_count >= 2)
        target_weights = point_valid.to(dtype=centers.dtype)
        predicted_weights = center_valid.to(dtype=centers.dtype)
        per_sample, component_logs, valid, _ = _spatial_curve_loss(
            centers,
            target_world,
            predicted_weights,
            target_weights,
            config,
        )
        return _masked_mean(per_sample, valid), component_logs, valid

    fixed_3d, fixed_logs, fixed_valid = fused_loss(fixed_center, fixed_view_count)
    predicted_3d, predicted_logs, predicted_valid = fused_loss(
        predicted_center,
        predicted_view_count,
    )
    loss_3d = fixed_rig_weight * fixed_3d + predicted_rig_weight * predicted_3d
    total = config.curve_2d_weight * loss_2d + config.curve_3d_weight * loss_3d

    logs = {
        "loss_keypoint_tail_curve": total,
        "loss_keypoint_tail_curve_2d": loss_2d,
        "loss_keypoint_tail_curve_2d_partial": partial_2d,
        "loss_keypoint_tail_curve_2d_shape": _masked_mean(logs_2d["shape"], valid_2d),
        "loss_keypoint_tail_curve_2d_length": _masked_mean(logs_2d["length"], valid_2d),
        "loss_keypoint_tail_curve_2d_curvature": _masked_mean(logs_2d["curvature"], valid_2d),
        "loss_keypoint_tail_curve_3d": loss_3d,
        "loss_keypoint_tail_curve_3d_fixed": fixed_3d,
        "loss_keypoint_tail_curve_3d_predicted": predicted_3d,
        "loss_keypoint_tail_curve_3d_fixed_shape": _masked_mean(fixed_logs["shape"], fixed_valid),
        "loss_keypoint_tail_curve_3d_predicted_shape": _masked_mean(predicted_logs["shape"], predicted_valid),
        "keypoint_tail_valid_2d_views": valid_2d.sum().to(dtype=predicted_xy.dtype),
        "keypoint_tail_valid_3d_fixed": fixed_valid.sum().to(dtype=predicted_xy.dtype),
        "keypoint_tail_valid_3d_predicted": predicted_valid.sum().to(dtype=predicted_xy.dtype),
        "keypoint_tail_mean_view_weight": _masked_mean(view_weights, valid_2d),
    }
    return total, logs

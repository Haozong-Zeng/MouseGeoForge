#!/usr/bin/env python3
"""Refine one shared three-camera rig per trial from multiview 2D tracks.

The optimization never uses Anipose 3D labels. Per-frame/keypoint 3D locations
are eliminated analytically by weighted ray intersection, leaving only the
shared CamB/C relative poses as trainable variables. Anipose 3D is loaded after
optimization for evaluation with one trial-level similarity transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CAMERAS = ("A", "B", "C")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "water-reaching-dataset"
    / "vggt-omega-eval-geometryaligned6000-624x416-global-full-perframe-min2"
)
DEFAULT_INIT_ROOT = (
    REPO_ROOT
    / "water-reaching-dataset"
    / "vggt-omega-eval-geometryaligned6000-624x416-global-full-fixedrel-pairwise-top50-min2"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "shared_rig_refinement_geometryaligned6000"


@dataclass
class TrackData:
    frames: np.ndarray
    keypoints: list[str]
    uv: np.ndarray
    weights: np.ndarray
    valid: np.ndarray
    gt_world: np.ndarray
    intrinsics: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--init-root", type=Path, default=DEFAULT_INIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trial", default="FJ_L2_2024-06-24_1_Trial_7")
    parser.add_argument("--all-trials", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--val-stride", type=int, default=5)
    parser.add_argument("--train-min-views", type=int, default=3)
    parser.add_argument("--eval-min-views", type=int, default=2)
    parser.add_argument("--huber-beta-px", type=float, default=3.0)
    parser.add_argument("--rotation-prior-weight", type=float, default=0.02)
    parser.add_argument("--translation-prior-weight", type=float, default=0.002)
    parser.add_argument("--baseline-ratio-prior-weight", type=float, default=0.02)
    parser.add_argument("--cheirality-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def load_initial_extrinsics(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("fixed_relative_extrinsics")
    if values is None:
        values = payload.get("camera_stabilization", {}).get("fixed_relative_extrinsics")
    if values is None:
        raise KeyError(f"No fixed_relative_extrinsics in {path}")
    output = np.stack([np.asarray(values[name], dtype=np.float64) for name in CAMERAS])
    for view in range(len(CAMERAS)):
        output[view, :3, :3] = project_rotation(output[view, :3, :3])
    output[0, :3, :3] = np.eye(3)
    output[0, :3, 3] = 0.0
    return output


def load_tracks(sample_csv: Path, camera_csv: Path) -> TrackData:
    sample_df = pd.read_csv(sample_csv)
    camera_df = pd.read_csv(camera_csv)
    required = {
        "frame",
        "keypoint",
        "view_idx",
        "x_model",
        "y_model",
        "likelihood",
        "gt_x",
        "gt_y",
        "gt_z",
    }
    missing = required.difference(sample_df.columns)
    if missing:
        raise KeyError(f"{sample_csv} is missing columns: {sorted(missing)}")

    groups = list(sample_df.groupby(["frame", "keypoint"], sort=True))
    count = len(groups)
    uv = np.zeros((count, len(CAMERAS), 2), dtype=np.float64)
    weights = np.zeros((count, len(CAMERAS)), dtype=np.float64)
    valid = np.zeros((count, len(CAMERAS)), dtype=bool)
    frames = np.zeros(count, dtype=np.int64)
    gt_world = np.zeros((count, 3), dtype=np.float64)
    keypoints: list[str] = []
    for point_idx, ((frame, keypoint), rows) in enumerate(groups):
        frames[point_idx] = int(frame)
        keypoints.append(str(keypoint))
        first = rows.iloc[0]
        gt_world[point_idx] = [
            float(first["gt_x"]),
            float(first["gt_y"]),
            float(first["gt_z"]),
        ]
        for row in rows.itertuples(index=False):
            view = int(row.view_idx)
            if view < 0 or view >= len(CAMERAS):
                continue
            xy = np.asarray([row.x_model, row.y_model], dtype=np.float64)
            likelihood = float(row.likelihood)
            if not np.isfinite(xy).all() or not np.isfinite(likelihood):
                continue
            uv[point_idx, view] = xy
            weights[point_idx, view] = max(likelihood, 1e-3) ** 2
            valid[point_idx, view] = True

    intrinsics = np.zeros((len(CAMERAS), 3, 3), dtype=np.float64)
    for view, name in enumerate(CAMERAS):
        rows = camera_df[camera_df["cam"].astype(str) == name]
        if rows.empty:
            raise ValueError(f"No camera rows for {name} in {camera_csv}")
        intrinsics[view] = np.asarray(
            [
                [float(rows["fx"].median()), 0.0, float(rows["cx"].median())],
                [0.0, float(rows["fy"].median()), float(rows["cy"].median())],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return TrackData(
        frames=frames,
        keypoints=keypoints,
        uv=uv,
        weights=weights,
        valid=valid,
        gt_world=gt_world,
        intrinsics=intrinsics,
    )


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    return torch.matrix_exp(skew(vector))


class SharedRig(torch.nn.Module):
    def __init__(self, initial: torch.Tensor):
        super().__init__()
        self.register_buffer("initial_rotations", initial[:, :3, :3].clone())
        self.register_buffer("initial_translations", initial[:, :3, 3].clone())
        self.rotation_delta = torch.nn.Parameter(torch.zeros(2, 3, dtype=initial.dtype))
        self.translation_raw = torch.nn.Parameter(initial[1:, :3, 3].clone())
        baseline = torch.linalg.vector_norm(initial[1, :3, 3]).clamp_min(1e-6)
        self.register_buffer("anchor_baseline", baseline)
        initial_ratio = torch.linalg.vector_norm(initial[2, :3, 3]) / baseline
        self.register_buffer("initial_baseline_ratio", initial_ratio)

    def forward(self) -> torch.Tensor:
        rotations = torch.cat(
            [
                self.initial_rotations[:1],
                so3_exp(self.rotation_delta) @ self.initial_rotations[1:],
            ],
            dim=0,
        )
        scale = self.anchor_baseline / torch.linalg.vector_norm(
            self.translation_raw[0]
        ).clamp_min(1e-6)
        translations = torch.cat(
            [
                torch.zeros_like(self.initial_translations[:1]),
                self.translation_raw * scale,
            ],
            dim=0,
        )
        return torch.cat([rotations, translations[..., None]], dim=-1)

    def prior_loss(
        self,
        *,
        rotation_weight: float,
        translation_weight: float,
        baseline_ratio_weight: float,
    ) -> torch.Tensor:
        rotation = self.rotation_delta.square().mean()
        scale = self.anchor_baseline / torch.linalg.vector_norm(
            self.translation_raw[0]
        ).clamp_min(1e-6)
        translations = self.translation_raw * scale
        translation = (
            translations - self.initial_translations[1:]
        ).square().mean() / self.anchor_baseline.square().clamp_min(1e-8)
        ratio = (
            torch.linalg.vector_norm(translations[1]) / self.anchor_baseline
        )
        baseline = (ratio - self.initial_baseline_ratio).square()
        return (
            float(rotation_weight) * rotation
            + float(translation_weight) * translation
            + float(baseline_ratio_weight) * baseline
        )


def triangulate_rays(
    uv: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fx = intrinsics[:, 0, 0][None, :]
    fy = intrinsics[:, 1, 1][None, :]
    cx = intrinsics[:, 0, 2][None, :]
    cy = intrinsics[:, 1, 2][None, :]
    camera_directions = torch.stack(
        [
            (uv[..., 0] - cx) / fx.clamp_min(1e-6),
            (uv[..., 1] - cy) / fy.clamp_min(1e-6),
            torch.ones_like(uv[..., 0]),
        ],
        dim=-1,
    )
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    world_directions = torch.einsum(
        "vij,nvj->nvi",
        rotations.transpose(-1, -2),
        camera_directions,
    )
    world_directions = F.normalize(world_directions, dim=-1, eps=1e-8)
    centers = -torch.einsum(
        "vij,vj->vi",
        rotations.transpose(-1, -2),
        translations,
    )
    identity = torch.eye(3, dtype=uv.dtype, device=uv.device)
    projector = identity - (
        world_directions[..., :, None] * world_directions[..., None, :]
    )
    effective_weights = weights * valid.to(dtype=weights.dtype)
    precision = (effective_weights[..., None, None] * projector).sum(dim=1)
    rhs = torch.einsum(
        "nvij,vj,nv->ni",
        projector,
        centers,
        effective_weights,
    )
    diagonal = precision.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    regularizer = (diagonal * 1e-6).clamp_min(1e-7)
    regularized = precision + regularizer[:, None, None] * identity
    points = torch.linalg.solve(regularized, rhs[..., None]).squeeze(-1)
    covariance = torch.linalg.inv(regularized)
    return points, covariance


def project_points(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = torch.einsum("vij,nj->nvi", extrinsics[:, :3, :3], points)
    camera = camera + extrinsics[:, :3, 3][None, :, :]
    homogeneous = torch.einsum("vij,nvj->nvi", intrinsics, camera)
    uv = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-6)
    return uv, camera[..., 2]


def reprojection_stats(
    uv: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    points, covariance = triangulate_rays(
        uv,
        weights,
        valid,
        extrinsics,
        intrinsics,
    )
    projected, depth = project_points(points, extrinsics, intrinsics)
    error = torch.linalg.vector_norm(projected - uv, dim=-1)
    mask = valid & torch.isfinite(error) & (depth > 1e-6)
    values = error[mask]
    stats = {
        "count": int(values.numel()),
        "mean_px": float(values.mean().item()) if values.numel() else float("nan"),
        "median_px": float(values.median().item()) if values.numel() else float("nan"),
        "p90_px": (
            float(torch.quantile(values, 0.9).item())
            if values.numel()
            else float("nan")
        ),
        "rmse_px": (
            float(torch.sqrt((values * values).mean()).item())
            if values.numel()
            else float("nan")
        ),
    }
    return stats, points, covariance, mask


def similarity_transform(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if len(source) < 3:
        raise ValueError("Need at least three points for similarity alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "fit_count": int(len(source)),
    }


def apply_similarity(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    return (
        float(transform["scale"])
        * (np.asarray(transform["rotation"]) @ points.T).T
        + np.asarray(transform["translation"])[None, :]
    )


def error_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "mean_mm": float(np.mean(values)) if len(values) else float("nan"),
        "median_mm": float(np.median(values)) if len(values) else float("nan"),
        "p90_mm": float(np.quantile(values, 0.9)) if len(values) else float("nan"),
        "rmse_mm": (
            float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")
        ),
    }


def evaluate_3d(
    points: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, Any]:
    all_transform = similarity_transform(points, target)
    all_aligned = apply_similarity(points, all_transform)
    all_error = np.linalg.norm(all_aligned - target, axis=1)
    train_transform = similarity_transform(points[train_mask], target[train_mask])
    train_aligned = apply_similarity(points, train_transform)
    train_error = np.linalg.norm(train_aligned - target, axis=1)
    return {
        "all_fit_all_frames": error_stats(all_error),
        "train_fit_train_frames": error_stats(train_error[train_mask]),
        "train_fit_validation_frames": error_stats(train_error[val_mask]),
        "all_fit_transform": {
            "scale": float(all_transform["scale"]),
            "rotation": np.asarray(all_transform["rotation"]).tolist(),
            "translation": np.asarray(all_transform["translation"]).tolist(),
            "fit_count": int(all_transform["fit_count"]),
        },
        "train_fit_transform": {
            "scale": float(train_transform["scale"]),
            "rotation": np.asarray(train_transform["rotation"]).tolist(),
            "translation": np.asarray(train_transform["translation"]).tolist(),
            "fit_count": int(train_transform["fit_count"]),
        },
        "all_error": all_error,
        "train_fit_error": train_error,
    }


def rotation_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    delta = first @ second.T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rig_delta(initial: np.ndarray, refined: np.ndarray) -> dict[str, Any]:
    initial_centers = np.stack(
        [-initial[v, :3, :3].T @ initial[v, :3, 3] for v in range(3)]
    )
    refined_centers = np.stack(
        [-refined[v, :3, :3].T @ refined[v, :3, 3] for v in range(3)]
    )
    return {
        "rotation_change_deg": {
            CAMERAS[v]: rotation_angle_degrees(
                refined[v, :3, :3],
                initial[v, :3, :3],
            )
            for v in range(3)
        },
        "camera_center_change": {
            CAMERAS[v]: float(
                np.linalg.norm(refined_centers[v] - initial_centers[v])
            )
            for v in range(3)
        },
        "initial_camera_centers": initial_centers.tolist(),
        "refined_camera_centers": refined_centers.tolist(),
    }


def optimize_trial(
    trial: str,
    source_dir: Path,
    init_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(
        source_dir / "keypoint_view_samples.csv",
        source_dir / "camera_extrinsics.csv",
    )
    initial_np = load_initial_extrinsics(init_dir / "fixed_relative_extrinsics.json")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float32
    initial = torch.as_tensor(initial_np, dtype=dtype, device=device)
    intrinsics = torch.as_tensor(tracks.intrinsics, dtype=dtype, device=device)
    uv_all = torch.as_tensor(tracks.uv, dtype=dtype, device=device)
    weights_all = torch.as_tensor(tracks.weights, dtype=dtype, device=device)
    valid_all = torch.as_tensor(tracks.valid, dtype=torch.bool, device=device)

    point_view_count = tracks.valid.sum(axis=1)
    val_mask_np = (
        (tracks.frames % max(int(args.val_stride), 2)) == 0
    ) & (point_view_count >= int(args.eval_min_views))
    train_mask_np = (
        ~val_mask_np
    ) & (point_view_count >= int(args.train_min_views))
    eval_mask_np = point_view_count >= int(args.eval_min_views)
    if train_mask_np.sum() < 10 or val_mask_np.sum() < 3:
        raise RuntimeError(
            f"{trial}: insufficient train/validation tracks "
            f"({train_mask_np.sum()}/{val_mask_np.sum()})"
        )

    train_indices = torch.as_tensor(
        np.flatnonzero(train_mask_np),
        dtype=torch.long,
        device=device,
    )
    uv = uv_all[train_indices]
    weights = weights_all[train_indices]
    valid = valid_all[train_indices]

    torch.manual_seed(int(args.seed))
    model = SharedRig(initial).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.steps), 1),
        eta_min=float(args.lr) * 0.02,
    )
    history: list[dict[str, float]] = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        extrinsics = model()
        points, _ = triangulate_rays(
            uv,
            weights,
            valid,
            extrinsics,
            intrinsics,
        )
        projected, depth = project_points(points, extrinsics, intrinsics)
        reprojection_valid = (
            valid
            & torch.isfinite(projected).all(dim=-1)
            & torch.isfinite(depth)
            & (depth > 1e-3)
        )
        residual = projected - uv
        robust = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=float(args.huber_beta_px),
            reduction="none",
        ).sum(dim=-1)
        weighted = weights * reprojection_valid.to(dtype=dtype)
        reprojection_loss = (robust * weighted).sum() / weighted.sum().clamp_min(1e-6)
        cheirality = F.relu(1e-3 - depth)
        cheirality_loss = (
            cheirality * valid.to(dtype=dtype)
        ).sum() / valid.sum().clamp_min(1)
        prior = model.prior_loss(
            rotation_weight=args.rotation_prior_weight,
            translation_weight=args.translation_prior_weight,
            baseline_ratio_weight=args.baseline_ratio_prior_weight,
        )
        loss = (
            reprojection_loss
            + float(args.cheirality_weight) * cheirality_loss
            + prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        row = {
            "step": float(step + 1),
            "loss": float(loss.detach().item()),
            "reprojection_loss": float(reprojection_loss.detach().item()),
            "cheirality_loss": float(cheirality_loss.detach().item()),
            "prior_loss": float(prior.detach().item()),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        if step == 0 or (step + 1) % max(1, int(args.log_every)) == 0:
            print(
                f"{trial} step={step + 1}/{args.steps} "
                f"loss={row['loss']:.5f} reproj={row['reprojection_loss']:.5f}",
                flush=True,
            )

    with torch.no_grad():
        refined = model()
        initial_reprojection, initial_points, _, _ = reprojection_stats(
            uv_all[torch.as_tensor(eval_mask_np, device=device)],
            weights_all[torch.as_tensor(eval_mask_np, device=device)],
            valid_all[torch.as_tensor(eval_mask_np, device=device)],
            initial,
            intrinsics,
        )
        refined_reprojection, refined_points, refined_cov, refined_obs_mask = (
            reprojection_stats(
                uv_all[torch.as_tensor(eval_mask_np, device=device)],
                weights_all[torch.as_tensor(eval_mask_np, device=device)],
                valid_all[torch.as_tensor(eval_mask_np, device=device)],
                refined,
                intrinsics,
            )
        )

    eval_indices = np.flatnonzero(eval_mask_np)
    eval_train_mask = train_mask_np[eval_indices]
    eval_val_mask = val_mask_np[eval_indices]
    target = tracks.gt_world[eval_indices]
    initial_points_np = initial_points.detach().cpu().numpy()
    refined_points_np = refined_points.detach().cpu().numpy()
    initial_3d = evaluate_3d(
        initial_points_np,
        target,
        eval_train_mask,
        eval_val_mask,
    )
    refined_3d = evaluate_3d(
        refined_points_np,
        target,
        eval_train_mask,
        eval_val_mask,
    )
    refined_np = refined.detach().cpu().numpy()

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "optimization_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(history_df["step"], history_df["reprojection_loss"], label="reprojection")
    ax.plot(history_df["step"], history_df["loss"], label="total", alpha=0.8)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Robust loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "optimization_curve.png", dpi=180)
    plt.close(fig)

    rows = []
    for local_idx, global_idx in enumerate(eval_indices):
        rows.append(
            {
                "frame": int(tracks.frames[global_idx]),
                "keypoint": tracks.keypoints[global_idx],
                "split": "validation" if eval_val_mask[local_idx] else "train",
                "initial_allfit_error_mm": float(
                    initial_3d["all_error"][local_idx]
                ),
                "refined_allfit_error_mm": float(
                    refined_3d["all_error"][local_idx]
                ),
                "initial_trainfit_error_mm": float(
                    initial_3d["train_fit_error"][local_idx]
                ),
                "refined_trainfit_error_mm": float(
                    refined_3d["train_fit_error"][local_idx]
                ),
                "refined_cov_trace": float(
                    refined_cov[local_idx]
                    .diagonal(dim1=-2, dim2=-1)
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                ),
                "refined_valid_observations": int(
                    refined_obs_mask[local_idx].sum().detach().cpu().item()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "point_errors.csv", index=False)

    refined_payload = {
        "mode": "shared-video-bundle-adjustment",
        "anchor_camera": CAMERAS[0],
        "intrinsics_source": "per-trial median VGGT prediction",
        "optimization_supervision": "2D multiview tracks only; Anipose 3D excluded",
        "fixed_relative_extrinsics": {
            name: refined_np[index].tolist()
            for index, name in enumerate(CAMERAS)
        },
        "intrinsics": {
            name: tracks.intrinsics[index].tolist()
            for index, name in enumerate(CAMERAS)
        },
    }
    (output_dir / "refined_relative_extrinsics.json").write_text(
        json.dumps(refined_payload, indent=2),
        encoding="utf-8",
    )

    def without_arrays(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metrics.items()
            if key not in {"all_error", "train_fit_error"}
        }

    summary = {
        "trial": trial,
        "source_dir": str(source_dir),
        "init_dir": str(init_dir),
        "device": str(device),
        "track_count": int(len(tracks.frames)),
        "train_track_count": int(train_mask_np.sum()),
        "validation_track_count": int(val_mask_np.sum()),
        "eval_track_count": int(eval_mask_np.sum()),
        "train_min_views": int(args.train_min_views),
        "eval_min_views": int(args.eval_min_views),
        "val_stride": int(args.val_stride),
        "steps": int(args.steps),
        "initial": {
            "reprojection": initial_reprojection,
            "keypoint_3d": without_arrays(initial_3d),
        },
        "refined": {
            "reprojection": refined_reprojection,
            "keypoint_3d": without_arrays(refined_3d),
        },
        "rig_delta": rig_delta(initial_np, refined_np),
        "outputs": {
            "refined_rig": str(output_dir / "refined_relative_extrinsics.json"),
            "curve_csv": str(output_dir / "optimization_curve.csv"),
            "curve_png": str(output_dir / "optimization_curve.png"),
            "point_errors": str(output_dir / "point_errors.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    initial_3d = summary["initial"]["keypoint_3d"]
    refined_3d = summary["refined"]["keypoint_3d"]
    return {
        "trial": summary["trial"],
        "train_tracks": summary["train_track_count"],
        "validation_tracks": summary["validation_track_count"],
        "initial_reproj_median_px": summary["initial"]["reprojection"]["median_px"],
        "refined_reproj_median_px": summary["refined"]["reprojection"]["median_px"],
        "initial_allfit_median_mm": initial_3d["all_fit_all_frames"]["median_mm"],
        "refined_allfit_median_mm": refined_3d["all_fit_all_frames"]["median_mm"],
        "initial_val_median_mm": initial_3d["train_fit_validation_frames"]["median_mm"],
        "refined_val_median_mm": refined_3d["train_fit_validation_frames"]["median_mm"],
        "initial_val_rmse_mm": initial_3d["train_fit_validation_frames"]["rmse_mm"],
        "refined_val_rmse_mm": refined_3d["train_fit_validation_frames"]["rmse_mm"],
        "camB_rotation_change_deg": summary["rig_delta"]["rotation_change_deg"]["B"],
        "camC_rotation_change_deg": summary["rig_delta"]["rotation_change_deg"]["C"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [
        key
        for key in rows[0]
        if key != "trial" and isinstance(rows[0][key], (int, float))
    ]
    output: dict[str, Any] = {"trial_count": len(rows)}
    for key in numeric:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[f"{key}_median"] = float(np.nanmedian(values))
        output[f"{key}_mean"] = float(np.nanmean(values))
    output["improved_validation_trials"] = int(
        sum(row["refined_val_median_mm"] < row["initial_val_median_mm"] for row in rows)
    )
    output["improved_reprojection_trials"] = int(
        sum(
            row["refined_reproj_median_px"] < row["initial_reproj_median_px"]
            for row in rows
        )
    )
    return output


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    init_root = args.init_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.all_trials:
        trials = sorted(
            path.name
            for path in source_root.iterdir()
            if path.is_dir()
            and (path / "keypoint_view_samples.csv").exists()
            and (init_root / path.name / "fixed_relative_extrinsics.json").exists()
        )
    else:
        trials = [args.trial]
    if not trials:
        raise RuntimeError("No matching trials found")

    rows = []
    summaries = []
    for index, trial in enumerate(trials, start=1):
        print(f"[{index}/{len(trials)}] {trial}", flush=True)
        summary = optimize_trial(
            trial,
            source_root / trial,
            init_root / trial,
            output_root / trial,
            args,
        )
        summaries.append(summary)
        rows.append(flatten_summary(summary))
        pd.DataFrame(rows).to_csv(output_root / "trial_summary.csv", index=False)

    aggregate_summary = aggregate(rows)
    payload = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "aggregate": aggregate_summary,
        "trials": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

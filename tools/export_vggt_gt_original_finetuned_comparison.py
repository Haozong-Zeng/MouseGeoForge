#!/usr/bin/env python3
"""Export GT, original, and finetuned VGGT-Omega geometry for one Blender scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
VGGT_ROOT = PROJECT_ROOT / "vggt-omega"
for path in (TOOLS_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_blendedmvs_depth_consistency import load_scene, read_pfm  # noqa: E402
from evaluate_vggt_omega_blendedmvs import align_predicted_to_gt_world, load_model  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


VIEW_COLORS = np.array(
    [[220, 58, 66], [43, 157, 92], [48, 112, 205], [214, 151, 38], [126, 81, 163]],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=240000)
    parser.add_argument("--pfm-flip-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    return -extrinsic[:3, :3].T @ extrinsic[:3, 3]


def rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = predicted[:3, :3] @ target[:3, :3].T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def scale_intrinsics_to_input(
    intrinsic: np.ndarray,
    original_hw: tuple[int, int],
    input_hw: tuple[int, int],
) -> np.ndarray:
    original_h, original_w = original_hw
    input_h, input_w = input_hw
    scaled = intrinsic.copy()
    scaled[0, :] *= input_w / float(original_w)
    scaled[1, :] *= input_h / float(original_h)
    return scaled


def infer(
    checkpoint: Path,
    image_paths: list[str],
    image_resolution: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model = load_model(checkpoint.resolve(), device)
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )
    result = {
        "images": images.detach().float().cpu().numpy(),
        "depth": predictions["depth"][0, ..., 0].detach().float().cpu().numpy(),
        "depth_conf": predictions["depth_conf"][0].detach().float().cpu().numpy(),
        "extrinsics": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
    }
    del predictions, extrinsics, intrinsics, images, model
    torch.cuda.empty_cache()
    return result


def cloud_from_depth(
    images: np.ndarray,
    depths: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    valid_masks: np.ndarray,
    stride: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_all = []
    colors_all = []
    views_all = []
    for view in range(depths.shape[0]):
        height, width = depths[view].shape
        yy, xx = np.mgrid[0:height:stride, 0:width:stride]
        z = depths[view, yy, xx]
        valid = valid_masks[view, yy, xx] & np.isfinite(z) & (z > 1e-8)
        x = (xx - intrinsics[view, 0, 2]) * z / intrinsics[view, 0, 0]
        y = (yy - intrinsics[view, 1, 2]) * z / intrinsics[view, 1, 1]
        camera = np.stack([x, y, z], axis=-1)
        rotation = extrinsics[view, :3, :3]
        translation = extrinsics[view, :3, 3]
        world = (rotation.T @ (camera.reshape(-1, 3) - translation).T).T
        rgb = np.clip(images[view].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        points_all.append(world[valid.reshape(-1)])
        colors_all.append(rgb[yy, xx][valid])
        views_all.append(np.full(int(valid.sum()), view, dtype=np.int16))
    points = np.concatenate(points_all)
    colors = np.concatenate(colors_all)
    views = np.concatenate(views_all)
    finite = np.isfinite(points).all(axis=1)
    points, colors, views = points[finite], colors[finite], views[finite]
    if len(points) > max_points:
        keep = rng.choice(len(points), size=max_points, replace=False)
        points, colors, views = points[keep], colors[keep], views[keep]
    return points.astype(np.float32), colors, views


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    values = np.empty(len(points), dtype=dtype)
    values["x"], values["y"], values["z"] = points[:, 0], points[:, 1], points[:, 2]
    values["red"], values["green"], values["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        values.tofile(handle)


def shared_bounds(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sampled = []
    rng = np.random.default_rng(17)
    for points in clouds:
        if len(points) > 80000:
            points = points[rng.choice(len(points), 80000, replace=False)]
        sampled.append(points)
    merged = np.concatenate(sampled)
    low = np.percentile(merged, 1.0, axis=0)
    high = np.percentile(merged, 99.0, axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low) / 2.0), 1e-6)
    return center - radius, center + radius


def set_bounds(ax, low: np.ndarray, high: np.ndarray) -> None:
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect((1, 1, 1))


def plot_clouds(
    path: Path,
    records: list[tuple[str, np.ndarray, np.ndarray]],
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(15.2, 9.4), dpi=180)
    views = [(24, -58, "Perspective"), (82, -90, "Top")]
    rng = np.random.default_rng(23)
    for row, (elev, azim, view_name) in enumerate(views):
        for column, (name, points, colors) in enumerate(records):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            if len(points) > 65000:
                keep = rng.choice(len(points), 65000, replace=False)
                draw_points, draw_colors = points[keep], colors[keep]
            else:
                draw_points, draw_colors = points, colors
            ax.scatter(
                draw_points[:, 0],
                draw_points[:, 1],
                draw_points[:, 2],
                c=draw_colors.astype(np.float32) / 255.0,
                s=0.28,
                alpha=0.65,
                linewidths=0,
            )
            set_bounds(ax, low, high)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{name} | {view_name}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
    fig.suptitle("Dense point cloud reconstruction in the shared GT world gauge", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_camera_axes(ax, extrinsics: np.ndarray, label: str) -> None:
    centers = np.stack([camera_center(item) for item in extrinsics])
    span = max(float(np.max(np.ptp(centers, axis=0))), 1e-6)
    axis_length = span * 0.12
    for index, (extrinsic, center) in enumerate(zip(extrinsics, centers)):
        color = VIEW_COLORS[index % len(VIEW_COLORS)].astype(np.float32) / 255.0
        ax.scatter(*center, c=[color], s=55, marker="^", edgecolors="black", linewidths=0.5)
        world_axes = extrinsic[:3, :3].T
        forward = world_axes[:, 2]
        ax.quiver(*center, *forward, length=axis_length, color=color, linewidth=1.5)
        ax.text(*center, f"cam{index}", fontsize=8)
    ax.set_title(label)


def plot_cameras(
    path: Path,
    gt: np.ndarray,
    original: np.ndarray,
    finetuned: np.ndarray,
) -> None:
    all_centers = np.concatenate(
        [np.stack([camera_center(item) for item in rig]) for rig in (gt, original, finetuned)]
    )
    center = (all_centers.min(axis=0) + all_centers.max(axis=0)) / 2.0
    radius = max(float(np.max(np.ptp(all_centers, axis=0)) / 2.0), 1e-6) * 1.15
    fig = plt.figure(figsize=(15.0, 5.2), dpi=180)
    for index, (name, rig) in enumerate(
        (("Ground truth", gt), ("Original VGGT-Omega", original), ("Finetuned", finetuned)),
        start=1,
    ):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        draw_camera_axes(ax, rig, name)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-58)
    fig.suptitle("Camera poses after one shared-rig Sim(3) gauge alignment per model", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def depth_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    residual = np.abs(prediction[valid] - target[valid])
    relative = residual / np.maximum(np.abs(target[valid]), 1e-6)
    return {
        "depth_mae": float(np.mean(residual)),
        "depth_median_ae": float(np.median(residual)),
        "depth_abs_rel": float(np.mean(relative)),
        "depth_median_abs_rel": float(np.median(relative)),
    }


def camera_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | list[float]]:
    center_errors = [
        float(np.linalg.norm(camera_center(pred) - camera_center(gt)))
        for pred, gt in zip(predicted, target)
    ]
    rotation_errors = [rotation_error_deg(pred, gt) for pred, gt in zip(predicted, target)]
    return {
        "camera_center_errors": center_errors,
        "camera_center_rmse": float(np.sqrt(np.mean(np.square(center_errors)))),
        "camera_rotation_errors_deg": rotation_errors,
        "camera_rotation_median_deg": float(np.median(rotation_errors)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this comparison.")

    frames_by_id, _ = load_scene(args.scene.resolve())
    view_ids = sorted(frames_by_id)[: args.max_views]
    frames = [frames_by_id[index] for index in view_ids]
    image_paths = [str(frame.image_path) for frame in frames]

    original = infer(args.original_checkpoint, image_paths, args.image_resolution, device)
    finetuned = infer(args.finetuned_checkpoint, image_paths, args.image_resolution, device)
    if original["images"].shape != finetuned["images"].shape:
        raise ValueError("Original and finetuned inputs have different shapes.")
    images = original["images"]
    if images.ndim == 5:
        images = images[0]
    input_hw = tuple(int(value) for value in images.shape[-2:])

    gt_extrinsics = np.stack([frame.extrinsic[:3, :4] for frame in frames])
    gt_intrinsics = []
    gt_depths = []
    for frame in frames:
        rgb = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(frame.image_path)
        gt_intrinsics.append(scale_intrinsics_to_input(frame.intrinsic, rgb.shape[:2], input_hw))
        depth = read_pfm(frame.depth_path, flip_y=args.pfm_flip_y)
        gt_depths.append(cv2.resize(depth, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_NEAREST))
    gt_intrinsics = np.stack(gt_intrinsics)
    gt_depths = np.stack(gt_depths)
    valid_masks = np.isfinite(gt_depths) & (gt_depths > 1e-8)

    original_aligned, original_align = align_predicted_to_gt_world(
        original["extrinsics"],
        gt_extrinsics,
    )
    finetuned_aligned, finetuned_align = align_predicted_to_gt_world(
        finetuned["extrinsics"],
        gt_extrinsics,
    )
    original_depth = original["depth"] / float(original_align["align_scale_gt_to_pred"])
    finetuned_depth = finetuned["depth"] / float(finetuned_align["align_scale_gt_to_pred"])

    rng = np.random.default_rng(5)
    gt_points, gt_colors, _ = cloud_from_depth(
        images,
        gt_depths,
        gt_extrinsics,
        gt_intrinsics,
        valid_masks,
        args.stride,
        args.max_points,
        rng,
    )
    original_points, original_colors, _ = cloud_from_depth(
        images,
        original_depth,
        original_aligned,
        original["intrinsics"],
        valid_masks,
        args.stride,
        args.max_points,
        rng,
    )
    finetuned_points, finetuned_colors, _ = cloud_from_depth(
        images,
        finetuned_depth,
        finetuned_aligned,
        finetuned["intrinsics"],
        valid_masks,
        args.stride,
        args.max_points,
        rng,
    )

    records = [
        ("Ground truth", gt_points, gt_colors),
        ("Original VGGT-Omega", original_points, original_colors),
        ("Finetuned step 6000", finetuned_points, finetuned_colors),
    ]
    for filename, (_, points, colors) in zip(
        ("groundtruth.ply", "original.ply", "finetuned_step6000.ply"),
        records,
    ):
        write_ply(output_dir / filename, points, colors)

    low, high = shared_bounds([gt_points, original_points, finetuned_points])
    plot_clouds(output_dir / "pointcloud_comparison.png", records, low, high)
    plot_cameras(
        output_dir / "camera_pose_comparison.png",
        gt_extrinsics,
        original_aligned,
        finetuned_aligned,
    )

    metrics = {
        "scene": str(args.scene.resolve()),
        "view_ids": view_ids,
        "input_hw": list(input_hw),
        "gauge_alignment_note": (
            "Each prediction uses one Sim(3) fitted jointly to all selected GT camera centers. "
            "The same fitted scale is applied to the model depth."
        ),
        "original": {
            "checkpoint": str(args.original_checkpoint.resolve()),
            "alignment": original_align,
            **camera_metrics(original_aligned, gt_extrinsics),
            **depth_metrics(original_depth, gt_depths, valid_masks),
        },
        "finetuned": {
            "checkpoint": str(args.finetuned_checkpoint.resolve()),
            "alignment": finetuned_align,
            **camera_metrics(finetuned_aligned, gt_extrinsics),
            **depth_metrics(finetuned_depth, gt_depths, valid_masks),
        },
        "outputs": {
            "groundtruth_ply": str(output_dir / "groundtruth.ply"),
            "original_ply": str(output_dir / "original.ply"),
            "finetuned_ply": str(output_dir / "finetuned_step6000.ply"),
            "pointcloud_figure": str(output_dir / "pointcloud_comparison.png"),
            "camera_figure": str(output_dir / "camera_pose_comparison.png"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

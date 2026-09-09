#!/usr/bin/env python3
"""Export a real-frame GT/keypoint, original, and finetuned VGGT comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
WATER_EVAL_PATH = PROJECT_ROOT / "water-reaching-dataset" / "evaluate_vggt_omega_keypoints.py"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from export_real_vggt_omega_frame_pointcloud import read_anipose_extrinsics  # noqa: E402
from export_vggt_gt_original_finetuned_comparison import (  # noqa: E402
    camera_center,
    camera_metrics,
    cloud_from_depth,
    plot_cameras,
    set_bounds,
    shared_bounds,
    write_ply,
)
from evaluate_vggt_omega_blendedmvs import align_predicted_to_gt_world  # noqa: E402


CAMERAS = ("A", "B", "C")
FINGER_PREFIXES = ("1", "2", "3", "4")


def load_water_eval():
    spec = importlib.util.spec_from_file_location("water_eval_real_compare", WATER_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {WATER_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


water_eval = load_water_eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--original-summary", type=Path, required=True)
    parser.add_argument("--finetuned-summary", type=Path, required=True)
    parser.add_argument("--original-fixedrel", type=Path, default=None)
    parser.add_argument("--finetuned-fixedrel", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--roi-margin", type=float, default=55.0)
    parser.add_argument("--likelihood-threshold", type=float, default=0.7)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def infer(
    checkpoint: Path,
    rgb_frames: list[np.ndarray],
    image_resolution: int,
    preprocess_mode: str,
    device: torch.device,
) -> dict:
    image_tensor, transforms = water_eval.preprocess_rgb_frames(
        rgb_frames,
        mode=preprocess_mode,
        image_resolution=image_resolution,
        patch_size=16,
    )
    images = image_tensor.unsqueeze(0).to(device)
    model = water_eval.VGGTOmega().to(device).eval()
    water_eval.load_checkpoint_chain(model, checkpoint)
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = water_eval.encoding_to_camera(
            predictions["pose_enc"],
            images.shape[-2:],
        )
    result = {
        "images": image_tensor.detach().float().cpu().numpy(),
        "depth": predictions["depth"][0, ..., 0].detach().float().cpu().numpy(),
        "extrinsics": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
        "transforms": transforms,
    }
    del predictions, extrinsics, intrinsics, images, model
    torch.cuda.empty_cache()
    return result


def load_global_similarity(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    transforms = summary.get("alignment_transforms", [])
    if len(transforms) != 1 or transforms[0].get("scope") != "global":
        raise ValueError(f"Expected one global sequence transform in {path}")
    item = transforms[0]
    return (
        float(item["scale"]),
        np.asarray(item["rotation"], dtype=np.float64),
        np.asarray(item["translation"], dtype=np.float64),
    )


def load_fixed_relative_extrinsics(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["fixed_relative_extrinsics"]
    return np.stack([np.asarray(values[cam], dtype=np.float64) for cam in CAMERAS])


def apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def transform_cameras_to_gt_world(
    extrinsics: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transformed = np.zeros_like(extrinsics, dtype=np.float64)
    for index, extrinsic in enumerate(extrinsics):
        camera_rotation = extrinsic[:3, :3] @ rotation.T
        transformed[index, :3, :3] = camera_rotation
        transformed[index, :3, 3] = (
            scale * extrinsic[:3, 3] - camera_rotation @ translation
        )
    return transformed


def roi_masks(
    trial: Path,
    frame: int,
    transforms: list,
    model_hw: tuple[int, int],
    margin: float,
    likelihood_threshold: float,
) -> np.ndarray:
    keypoints_2d = water_eval.load_2d_keypoints(trial)
    gt_3d = water_eval.load_3d_keypoints(trial)
    names = water_eval.select_keypoints(
        keypoints_2d,
        gt_3d,
        include_regex=None,
        exclude_regex=water_eval.DEFAULT_EXCLUDE_RE,
    )
    height, width = model_hw
    masks = np.zeros((len(CAMERAS), height, width), dtype=bool)
    for view, cam in enumerate(CAMERAS):
        values = water_eval.get_2d_for_frame(keypoints_2d[cam], frame, names)
        mapped = []
        for x, y, likelihood in values.values():
            if likelihood < likelihood_threshold or not np.isfinite([x, y]).all():
                continue
            point = transforms[view].map_xy(np.asarray([x, y], dtype=np.float64))
            if np.isfinite(point).all():
                mapped.append(point)
        if not mapped:
            masks[view] = True
            continue
        points = np.stack(mapped)
        x0 = max(0, int(np.floor(points[:, 0].min() - margin)))
        x1 = min(width, int(np.ceil(points[:, 0].max() + margin + 1)))
        y0 = max(0, int(np.floor(points[:, 1].min() - margin)))
        y1 = min(height, int(np.ceil(points[:, 1].max() + margin + 1)))
        masks[view, y0:y1, x0:x1] = True
    return masks


def gt_keypoints(trial: Path, frame: int) -> dict[str, np.ndarray]:
    points_2d = water_eval.load_2d_keypoints(trial)
    points_3d = water_eval.load_3d_keypoints(trial)
    names = water_eval.select_keypoints(
        points_2d,
        points_3d,
        include_regex=None,
        exclude_regex=water_eval.DEFAULT_EXCLUDE_RE,
    )
    return water_eval.get_3d_for_frame(points_3d, frame, names)


def skeleton_edges() -> list[tuple[str, str]]:
    edges = [("snout_A", "mouth_tip_A")]
    for side in ("l", "r"):
        palm = f"{side}_mid_A"
        for finger in FINGER_PREFIXES:
            joint = f"{side}{finger}_joint_A"
            mid = f"{side}{finger}_mid_A"
            tip = f"{side}{finger}_tip_A"
            edges.extend([(palm, joint), (joint, mid), (mid, tip)])
    return edges


def draw_skeleton(ax, keypoints: dict[str, np.ndarray], color: str = "#f4c542") -> None:
    if not keypoints:
        return
    values = np.stack(list(keypoints.values()))
    ax.scatter(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        c=color,
        s=34,
        edgecolors="black",
        linewidths=0.35,
        depthshade=False,
    )
    for start, end in skeleton_edges():
        if start in keypoints and end in keypoints:
            a, b = keypoints[start], keypoints[end]
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color=color,
                linewidth=1.8,
            )


def plot_real_clouds(
    path: Path,
    keypoints: dict[str, np.ndarray],
    original: tuple[np.ndarray, np.ndarray],
    finetuned: tuple[np.ndarray, np.ndarray],
) -> None:
    original_points, original_colors = original
    finetuned_points, finetuned_colors = finetuned
    gt_points = np.stack(list(keypoints.values()))
    low, high = shared_bounds([gt_points, original_points, finetuned_points])
    rng = np.random.default_rng(31)
    fig = plt.figure(figsize=(15.2, 8.8), dpi=180)
    views = [(22, -58, "Perspective"), (82, -90, "Top")]
    for row, (elev, azim, view_name) in enumerate(views):
        for column, (name, cloud) in enumerate(
            (
                ("Anipose GT keypoints", None),
                ("Original dense depth", original),
                ("Finetuned dense depth", finetuned),
            )
        ):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            if cloud is not None:
                points, colors = cloud
                if len(points) > 60000:
                    keep = rng.choice(len(points), 60000, replace=False)
                    points, colors = points[keep], colors[keep]
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=colors.astype(np.float32) / 255.0,
                    s=0.32,
                    alpha=0.62,
                    linewidths=0,
                )
            draw_skeleton(ax, keypoints)
            set_bounds(ax, low, high)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{name} | {view_name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
    fig.suptitle(
        "Real frame in the Anipose world gauge; yellow is the measured 3D skeleton",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_camera_rigs(
    path: Path,
    gt: np.ndarray,
    original: np.ndarray,
    finetuned: np.ndarray,
) -> None:
    colors = ("#d1495b", "#2878b5", "#2a9d6f")
    labels = ("Calibration GT", "Original", "Finetuned")
    rigs = (gt, original, finetuned)
    centers = [np.stack([camera_center(item) for item in rig]) for rig in rigs]
    merged = np.concatenate(centers)
    center = (merged.min(axis=0) + merged.max(axis=0)) / 2.0
    radius = max(float(np.max(np.ptp(merged, axis=0)) / 2.0), 1e-6) * 1.2
    fig = plt.figure(figsize=(14.5, 4.9), dpi=180)
    for index, (label, rig, rig_centers, color) in enumerate(
        zip(labels, rigs, centers, colors),
        start=1,
    ):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.plot(rig_centers[:, 0], rig_centers[:, 1], rig_centers[:, 2], color=color)
        for camera_index, (extrinsic, camera_position) in enumerate(zip(rig, rig_centers)):
            forward = extrinsic[:3, :3].T[:, 2]
            ax.scatter(*camera_position, c=color, s=60, marker="^")
            ax.quiver(*camera_position, *forward, length=radius * 0.18, color=color)
            ax.text(*camera_position, f"cam{CAMERAS[camera_index]}", fontsize=8)
        ax.set_title(label)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
    fig.suptitle("Real camera rig after sequence-level keypoint gauge alignment", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trial = args.trial.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this comparison.")

    videos = {cam: trial / "videos-raw" / f"pawfilt_cam{cam}.mp4" for cam in CAMERAS}
    readers = {cam: water_eval.VideoFrameReader(videos[cam]) for cam in CAMERAS}
    try:
        rgb_frames = [readers[cam].read(args.frame) for cam in CAMERAS]
    finally:
        for reader in readers.values():
            reader.close()
    original = infer(
        args.original_checkpoint,
        rgb_frames,
        args.image_resolution,
        args.preprocess_mode,
        device,
    )
    finetuned = infer(
        args.finetuned_checkpoint,
        rgb_frames,
        args.image_resolution,
        args.preprocess_mode,
        device,
    )
    masks = roi_masks(
        trial,
        args.frame,
        original["transforms"],
        tuple(original["images"].shape[-2:]),
        args.roi_margin,
        args.likelihood_threshold,
    )
    rng = np.random.default_rng(7)

    outputs = {}
    aligned_clouds = {}
    aligned_cameras = {}
    for label, prediction, summary_path in (
        ("original", original, args.original_summary),
        ("finetuned", finetuned, args.finetuned_summary),
    ):
        points, colors, _ = cloud_from_depth(
            prediction["images"],
            prediction["depth"],
            prediction["extrinsics"],
            prediction["intrinsics"],
            masks,
            args.stride,
            args.max_points,
            rng,
        )
        scale, rotation, translation = load_global_similarity(summary_path.resolve())
        points = apply_similarity(points, scale, rotation, translation).astype(np.float32)
        cameras = transform_cameras_to_gt_world(
            prediction["extrinsics"],
            scale,
            rotation,
            translation,
        )
        ply = output_dir / f"{label}_dense_pointcloud.ply"
        write_ply(ply, points, colors)
        outputs[f"{label}_ply"] = str(ply)
        aligned_clouds[label] = (points, colors)
        aligned_cameras[label] = cameras

    keypoints = gt_keypoints(trial, args.frame)
    gt_points = np.stack(list(keypoints.values())).astype(np.float32)
    gt_colors = np.tile(np.asarray([[244, 197, 66]], dtype=np.uint8), (len(gt_points), 1))
    gt_ply = output_dir / "anipose_gt_keypoints.ply"
    write_ply(gt_ply, gt_points, gt_colors)

    gt_cameras = read_anipose_extrinsics(trial / "calibration" / "calibration.toml")
    cloud_figure = output_dir / "real_pointcloud_comparison.png"
    camera_figure = output_dir / "real_camera_pose_comparison.png"
    plot_real_clouds(
        cloud_figure,
        keypoints,
        aligned_clouds["original"],
        aligned_clouds["finetuned"],
    )
    plot_camera_rigs(
        camera_figure,
        gt_cameras,
        aligned_cameras["original"],
        aligned_cameras["finetuned"],
    )

    fixedrig_metrics = None
    fixedrig_figure = None
    if args.original_fixedrel is not None and args.finetuned_fixedrel is not None:
        gt_relative = water_eval.relative_extrinsics_anchor_first(gt_cameras)
        original_fixed = load_fixed_relative_extrinsics(args.original_fixedrel.resolve())
        finetuned_fixed = load_fixed_relative_extrinsics(args.finetuned_fixedrel.resolve())
        original_fixed_aligned, original_fixed_alignment = align_predicted_to_gt_world(
            original_fixed,
            gt_relative,
        )
        finetuned_fixed_aligned, finetuned_fixed_alignment = align_predicted_to_gt_world(
            finetuned_fixed,
            gt_relative,
        )
        fixedrig_figure = output_dir / "fixedrig_camera_pose_comparison.png"
        plot_cameras(
            fixedrig_figure,
            gt_relative,
            original_fixed_aligned,
            finetuned_fixed_aligned,
        )
        fixedrig_metrics = {
            "note": "Each fixed relative rig is aligned to the calibration rig using camera centers only.",
            "original": {
                "alignment": original_fixed_alignment,
                **camera_metrics(original_fixed_aligned, gt_relative),
            },
            "finetuned": {
                "alignment": finetuned_fixed_alignment,
                **camera_metrics(finetuned_fixed_aligned, gt_relative),
            },
        }

    metrics = {
        "trial": str(trial),
        "frame": args.frame,
        "groundtruth_note": (
            "Real data has calibration and sparse Anipose 3D keypoints, but no dense GT depth. "
            "Dense original/finetuned clouds are therefore overlaid with the sparse measured skeleton."
        ),
        "alignment_note": (
            "Each model uses the single global similarity fitted over the complete trial by scoring; "
            "no per-frame alignment is used."
        ),
        "original_camera": camera_metrics(aligned_cameras["original"], gt_cameras),
        "finetuned_camera": camera_metrics(aligned_cameras["finetuned"], gt_cameras),
        "fixed_relative_rig": fixedrig_metrics,
        "outputs": {
            "gt_keypoints_ply": str(gt_ply),
            **outputs,
            "pointcloud_figure": str(cloud_figure),
            "camera_figure": str(camera_figure),
            "fixedrig_camera_figure": str(fixedrig_figure) if fixedrig_figure else None,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

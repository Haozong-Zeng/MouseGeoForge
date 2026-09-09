"""Export one real water-reaching frame as VGGT-Omega diagnostic point clouds.

This script runs the unmodified/local VGGT-Omega checkpoint on one synchronized
camA/camB/camC frame, then exports point clouds twice:

1. using VGGT-Omega's predicted relative camera rig;
2. using the Anipose calibration rig, scaled to the VGGT camera baseline.

The second export is a quick control for deciding whether poor overlap mostly
comes from VGGT camera extrinsics or from depth/domain/background effects.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
WATER_EVAL_PATH = REPO_ROOT / "water-reaching-dataset" / "evaluate_vggt_omega_keypoints.py"


def load_water_eval_module():
    spec = importlib.util.spec_from_file_location("water_vggt_eval", WATER_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {WATER_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


water_eval = load_water_eval_module()


CAMERAS = ("A", "B", "C")
VIEW_COLORS = np.array(
    [
        [230, 60, 60],
        [45, 180, 80],
        [70, 120, 255],
    ],
    dtype=np.uint8,
)


@dataclass
class Cloud:
    points: np.ndarray
    colors: np.ndarray
    view_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial",
        type=Path,
        default=REPO_ROOT / "water-reaching-dataset" / "FJ_L2_2024-06-24_1_Trial_7",
    )
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "vggt-omega" / "vggt_omega_1b_512.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "real_vggt_omega_unfinetuned_frame_pointcloud",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dense-stride", type=int, default=4)
    parser.add_argument("--roi-margin", type=float, default=55.0, help="Margin in model-input pixels around valid 2D keypoints.")
    parser.add_argument("--likelihood-threshold", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_ply(path: Path, cloud: Cloud) -> None:
    valid = np.isfinite(cloud.points).all(axis=1)
    points = cloud.points[valid]
    colors = np.clip(cloud.colors[valid], 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    centers = []
    for extrinsic in extrinsics:
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        centers.append(-rotation.T @ translation)
    return np.stack(centers)


def rotation_angle_degrees(rotation_delta: np.ndarray) -> float:
    value = (float(np.trace(rotation_delta)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def camera_pair_rows(name: str, extrinsics: np.ndarray) -> list[dict[str, float | str]]:
    centers = camera_centers(extrinsics)
    rows = []
    for i, cam_i in enumerate(CAMERAS):
        for j, cam_j in enumerate(CAMERAS):
            if j <= i:
                continue
            rows.append(
                {
                    "rig": name,
                    "pair": f"{cam_i}{cam_j}",
                    "center_distance": float(np.linalg.norm(centers[i] - centers[j])),
                    "relative_rotation_deg": rotation_angle_degrees(extrinsics[j, :3, :3] @ extrinsics[i, :3, :3].T),
                }
            )
    return rows


def direction_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (an * bn), -1.0, 1.0))))


def read_anipose_extrinsics(calibration_path: Path) -> np.ndarray:
    data = tomllib.loads(calibration_path.read_text(encoding="utf-8"))
    extrinsics = []
    for idx in range(3):
        cam = data[f"cam_{idx}"]
        if cam["name"] != CAMERAS[idx]:
            raise ValueError(f"Expected cam_{idx} to be {CAMERAS[idx]}, got {cam['name']}")
        rotation, _ = cv2.Rodrigues(np.asarray(cam["rotation"], dtype=np.float64))
        translation = np.asarray(cam["translation"], dtype=np.float64)
        extrinsic = np.zeros((3, 4), dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        extrinsics.append(extrinsic)
    return np.stack(extrinsics)


def scale_calibration_to_vggt(calib_rel: np.ndarray, vggt_rel: np.ndarray) -> tuple[np.ndarray, float]:
    calib_centers = camera_centers(calib_rel)
    vggt_centers = camera_centers(vggt_rel)
    ratios = []
    for i in range(len(CAMERAS)):
        for j in range(i + 1, len(CAMERAS)):
            denom = np.linalg.norm(calib_centers[i] - calib_centers[j])
            numer = np.linalg.norm(vggt_centers[i] - vggt_centers[j])
            if denom > 1e-12 and np.isfinite(numer):
                ratios.append(numer / denom)
    scale = float(np.median(ratios))
    scaled = calib_rel.copy()
    scaled[:, :3, 3] *= scale
    return scaled, scale


def make_roi_boxes(
    trial: Path,
    frame: int,
    transforms: list,
    model_hw: tuple[int, int],
    margin: float,
    likelihood_threshold: float,
) -> dict[int, tuple[float, float, float, float]]:
    keypoints_2d = water_eval.load_2d_keypoints(trial)
    gt_3d = water_eval.load_3d_keypoints(trial)
    keypoint_names = water_eval.select_keypoints(
        keypoints_2d,
        gt_3d,
        include_regex=None,
        exclude_regex=water_eval.DEFAULT_EXCLUDE_RE,
    )
    boxes = {}
    height, width = model_hw
    for view_idx, cam in enumerate(CAMERAS):
        points = []
        two_d = water_eval.get_2d_for_frame(keypoints_2d[cam], frame, keypoint_names)
        for x, y, likelihood in two_d.values():
            if likelihood < likelihood_threshold or not np.isfinite([x, y]).all():
                continue
            mapped = transforms[view_idx].map_xy(np.array([x, y], dtype=np.float64))
            if np.isfinite(mapped).all():
                points.append(mapped)
        if not points:
            continue
        arr = np.stack(points)
        x0 = max(0.0, float(np.min(arr[:, 0]) - margin))
        x1 = min(float(width - 1), float(np.max(arr[:, 0]) + margin))
        y0 = max(0.0, float(np.min(arr[:, 1]) - margin))
        y1 = min(float(height - 1), float(np.max(arr[:, 1]) + margin))
        boxes[view_idx] = (x0, y0, x1, y1)
    return boxes


def cloud_from_depth(
    images: np.ndarray,
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    stride: int,
    roi_boxes: dict[int, tuple[float, float, float, float]] | None,
    color_mode: str,
) -> Cloud:
    all_points = []
    all_colors = []
    all_view_ids = []
    for view_idx in range(depth.shape[0]):
        height, width = depth[view_idx].shape
        ys, xs = np.mgrid[0:height:stride, 0:width:stride]
        z = depth[view_idx, ys, xs]
        fx, fy = intrinsics[view_idx, 0, 0], intrinsics[view_idx, 1, 1]
        cx, cy = intrinsics[view_idx, 0, 2], intrinsics[view_idx, 1, 2]
        cam = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=-1).reshape(-1, 3)
        rotation = extrinsics[view_idx, :3, :3]
        translation = extrinsics[view_idx, :3, 3]
        world = (rotation.T @ (cam - translation).T).T
        valid = np.isfinite(world).all(axis=1) & np.isfinite(z.reshape(-1)) & (z.reshape(-1) > 0)
        if roi_boxes and view_idx in roi_boxes:
            x0, y0, x1, y1 = roi_boxes[view_idx]
            valid &= (
                (xs.reshape(-1) >= x0)
                & (xs.reshape(-1) <= x1)
                & (ys.reshape(-1) >= y0)
                & (ys.reshape(-1) <= y1)
            )
        if color_mode == "rgb":
            colors = (images[view_idx, :, ys, xs].transpose(1, 2, 0).reshape(-1, 3) * 255.0).astype(np.uint8)
        elif color_mode == "view":
            colors = np.repeat(VIEW_COLORS[view_idx][None, :], world.shape[0], axis=0)
        else:
            raise ValueError(f"Unknown color mode: {color_mode}")
        all_points.append(world[valid])
        all_colors.append(colors[valid])
        all_view_ids.append(np.full(int(valid.sum()), view_idx, dtype=np.int16))
    return Cloud(
        points=np.concatenate(all_points, axis=0),
        colors=np.concatenate(all_colors, axis=0),
        view_ids=np.concatenate(all_view_ids, axis=0),
    )


def set_axes_equal(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) / 2.0)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_cloud_preview(path: Path, title: str, cloud: Cloud, extrinsics: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    keep = min(len(cloud.points), 45000)
    if keep < len(cloud.points):
        indices = rng.choice(len(cloud.points), size=keep, replace=False)
        points = cloud.points[indices]
        colors = cloud.colors[indices].astype(np.float32) / 255.0
    else:
        points = cloud.points
        colors = cloud.colors.astype(np.float32) / 255.0

    centers = camera_centers(extrinsics)
    fig = plt.figure(figsize=(14, 10), dpi=160)
    views = [
        ("iso", 22, -55),
        ("top", 88, -90),
        ("front-ish", 8, -90),
        ("side-ish", 8, 0),
    ]
    for idx, (label, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.35, alpha=0.6, linewidths=0)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=VIEW_COLORS / 255.0, s=48, marker="^", edgecolors="k")
        for cam, center in zip(CAMERAS, centers):
            ax.text(center[0], center[1], center[2], f"cam{cam}", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_axes_equal(ax, points)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_input_montage(path: Path, images: np.ndarray, roi_boxes: dict[int, tuple[float, float, float, float]]) -> None:
    panels = []
    for view_idx, cam in enumerate(CAMERAS):
        image = (images[view_idx].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if view_idx in roi_boxes:
            x0, y0, x1, y1 = roi_boxes[view_idx]
            cv2.rectangle(bgr, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), (0, 255, 255), 2)
        cv2.putText(bgr, f"cam{cam}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        panels.append(bgr)
    montage = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), montage)


def main() -> None:
    args = parse_args()
    trial = args.trial.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This diagnostic uses the released VGGT-Omega path and expects CUDA.")

    videos = {cam: trial / "videos-raw" / f"pawfilt_cam{cam}.mp4" for cam in CAMERAS}
    rgb_frames = [water_eval.read_video_frame(videos[cam], args.frame) for cam in CAMERAS]
    image_tensor, transforms = water_eval.preprocess_rgb_frames(
        rgb_frames,
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
        patch_size=args.patch_size,
    )
    images = image_tensor.unsqueeze(0).to(device)

    model = water_eval.VGGTOmega().to(device).eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)

    with torch.inference_mode():
        predictions = model(images)
        depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        depth_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
        extrinsics_t, intrinsics_t = water_eval.encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
        raw_extrinsics = extrinsics_t[0].detach().float().cpu().numpy()
        intrinsics = intrinsics_t[0].detach().float().cpu().numpy()

    vggt_rel = water_eval.relative_extrinsics_anchor_first(raw_extrinsics).astype(np.float64)
    calib_raw = read_anipose_extrinsics(trial / "calibration" / "calibration.toml")
    calib_rel = water_eval.relative_extrinsics_anchor_first(calib_raw).astype(np.float64)
    calib_scaled, calib_scale = scale_calibration_to_vggt(calib_rel, vggt_rel)

    roi_boxes = make_roi_boxes(
        trial=trial,
        frame=args.frame,
        transforms=transforms,
        model_hw=tuple(images.shape[-2:]),
        margin=args.roi_margin,
        likelihood_threshold=args.likelihood_threshold,
    )
    images_np = image_tensor.numpy()
    save_input_montage(output_dir / f"frame_{args.frame:04d}_model_inputs_roi.png", images_np, roi_boxes)

    clouds: dict[str, Cloud] = {}
    for rig_name, rig_extrinsics in (("vggt_pose", vggt_rel), ("anipose_calib_scaled", calib_scaled)):
        for roi_name, boxes in (("all", None), ("keypoint_roi", roi_boxes)):
            for color_mode in ("view", "rgb"):
                key = f"{rig_name}_{color_mode}_{roi_name}"
                clouds[key] = cloud_from_depth(
                    images_np,
                    depth,
                    rig_extrinsics,
                    intrinsics,
                    stride=max(1, args.dense_stride),
                    roi_boxes=boxes,
                    color_mode=color_mode,
                )
                write_ply(output_dir / f"frame_{args.frame:04d}_{key}.ply", clouds[key])
            plot_cloud_preview(
                output_dir / f"frame_{args.frame:04d}_{rig_name}_view_{roi_name}_preview.png",
                title=f"frame {args.frame} {rig_name} {roi_name}",
                cloud=clouds[f"{rig_name}_view_{roi_name}"],
                extrinsics=rig_extrinsics,
            )

    rows = []
    rows.extend(camera_pair_rows("vggt_relative_to_camA", vggt_rel))
    rows.extend(camera_pair_rows("anipose_relative_to_camA", calib_rel))
    rows.extend(camera_pair_rows("anipose_scaled_to_vggt", calib_scaled))

    vggt_centers = camera_centers(vggt_rel)
    calib_centers = camera_centers(calib_scaled)
    for view_idx, cam in enumerate(CAMERAS):
        rows.append(
            {
                "rig": "vggt_vs_scaled_anipose",
                "pair": f"cam{cam}_from_camA",
                "center_distance": float(np.linalg.norm(vggt_centers[view_idx] - calib_centers[view_idx])),
                "relative_rotation_deg": rotation_angle_degrees(vggt_rel[view_idx, :3, :3] @ calib_scaled[view_idx, :3, :3].T),
                "translation_direction_delta_deg": direction_angle_degrees(vggt_centers[view_idx], calib_centers[view_idx]),
            }
        )

    camera_csv = output_dir / "camera_rig_diagnostics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with camera_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    depth_stats = {}
    for view_idx, cam in enumerate(CAMERAS):
        z = depth[view_idx]
        conf = depth_conf[view_idx]
        depth_stats[cam] = {
            "depth_min": float(np.nanmin(z)),
            "depth_median": float(np.nanmedian(z)),
            "depth_max": float(np.nanmax(z)),
            "depth_conf_mean": float(np.nanmean(conf)),
            "depth_conf_median": float(np.nanmedian(conf)),
        }

    summary = {
        "trial": str(trial),
        "frame": int(args.frame),
        "checkpoint": str(checkpoint),
        "checkpoint_note": "unmodified local VGGT-Omega checkpoint; no fine-tuned adapter loaded",
        "model_input_hw": list(images.shape[-2:]),
        "dense_stride": int(args.dense_stride),
        "roi_margin": float(args.roi_margin),
        "anipose_to_vggt_translation_scale": float(calib_scale),
        "camera_diagnostics_csv": str(camera_csv),
        "depth_stats": depth_stats,
        "roi_boxes_model_pixels": {CAMERAS[idx]: list(box) for idx, box in roi_boxes.items()},
        "outputs": sorted(str(path) for path in output_dir.iterdir()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

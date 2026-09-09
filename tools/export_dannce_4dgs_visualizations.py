#!/usr/bin/env python3
"""Export compact visual diagnostics from a trained official 4D-GS model.

The output includes representative six-camera comparisons, raw metric depth,
an observed-camera time sweep, and a fixed-time free-view orbit.  Alpha is
recovered exactly from paired white/black-background renders.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-camera", type=int, default=80)
    parser.add_argument("--representative-frames", type=int, nargs="+", default=[0, 27, 40, 72, 79])
    parser.add_argument("--orbit-frame", type=int, default=27)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--annotation-output",
        type=Path,
        help="Optionally export all views as uint16 0.1-mm depth and uint8 alpha PNGs.",
    )
    return parser.parse_args()


def tensor_rgb(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    )


def label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 260), 29), (15, 15, 15), -1)
    cv2.putText(
        output,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def depth_visual(depth: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    valid = np.isfinite(depth) & (alpha > 0.05) & (depth > 0)
    if np.count_nonzero(valid) < 16:
        return np.zeros((*depth.shape, 3), dtype=np.uint8), (0.0, 0.0)
    near, far = np.percentile(depth[valid], [2.0, 98.0])
    if far <= near:
        far = near + 1e-6
    normalized = np.clip((depth - near) / (far - near), 0, 1)
    # Turbo maps near to blue and far to red after this inversion choice.
    color = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color, (float(near), float(far))


def render_products(camera, gaussians, pipeline, renderer, cam_type: str):
    white = torch.ones(3, dtype=torch.float32, device="cuda")
    black = torch.zeros(3, dtype=torch.float32, device="cuda")
    white_result = renderer(camera, gaussians, pipeline, white, stage="fine", cam_type=cam_type)
    black_result = renderer(camera, gaussians, pipeline, black, stage="fine", cam_type=cam_type)
    rgb = tensor_rgb(white_result["render"])
    rgb_black = black_result["render"].detach().clamp(0, 1)
    rgb_white = white_result["render"].detach().clamp(0, 1)
    alpha = (1.0 - (rgb_white - rgb_black).mean(dim=0)).clamp(0, 1).cpu().numpy()
    depth_tensor = white_result["depth"].detach().float().cpu()
    weighted_depth = depth_tensor.squeeze().numpy()
    # The official depth rasterizer accumulates sum(z * alpha * transmittance),
    # not an already-normalized expected z. Normalize by accumulated alpha.
    depth = np.zeros_like(weighted_depth, dtype=np.float32)
    valid_alpha = alpha > 1e-4
    depth[valid_alpha] = weighted_depth[valid_alpha] / alpha[valid_alpha]
    depth_rgb, limits = depth_visual(depth, alpha)
    return rgb, depth, alpha, depth_rgb, limits


def metric_summary(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    gt_f = gt.astype(np.float32) / 255.0
    pred_f = prediction.astype(np.float32) / 255.0
    mse = float(np.mean((gt_f - pred_f) ** 2))
    return {
        "mae": float(np.mean(np.abs(gt_f - pred_f))),
        "mse": mse,
        "psnr_db": float(-10.0 * np.log10(max(mse, 1e-12))),
    }


def silhouette_metrics(reference: np.ndarray, alpha: np.ndarray) -> dict[str, float]:
    target = reference > 127
    prediction = alpha > 0.5
    intersection = int(np.count_nonzero(target & prediction))
    union = int(np.count_nonzero(target | prediction))
    denominator = int(np.count_nonzero(target)) + int(np.count_nonzero(prediction))
    return {
        "silhouette_iou": float(intersection / max(union, 1)),
        "silhouette_dice": float(2 * intersection / max(denominator, 1)),
    }


def interpolated_rig_orbit(train, frame_count: int, frame: int, count: int = 120):
    """Build an upright closed orbit through the six physical camera positions."""
    from scene.cameras import Camera

    reference_cameras = [train[index * frame_count + frame] for index in range(len(train) // frame_count)]
    centers = np.stack([camera.camera_center.detach().cpu().numpy() for camera in reference_cameras])
    angles = np.arctan2(centers[:, 1], centers[:, 0])
    order = np.argsort(angles)
    angles = np.unwrap(angles[order])
    centers = centers[order]
    # Close the orbit without taking the long way around the angle discontinuity.
    angles_closed = np.concatenate([angles, [angles[0] + 2.0 * np.pi]])
    centers_closed = np.concatenate([centers, centers[:1]], axis=0)
    sample_angles = np.linspace(angles_closed[0], angles_closed[-1], count, endpoint=False)
    radius = np.linalg.norm(centers_closed[:, :2], axis=1)
    radius_interp = np.interp(sample_angles, angles_closed, radius)
    height_interp = np.interp(sample_angles, angles_closed, centers_closed[:, 2])
    fovx = reference_cameras[0].FoVx
    fovy = reference_cameras[0].FoVy
    height = reference_cameras[0].image_height
    width = reference_cameras[0].image_width
    dummy = torch.zeros((3, height, width), dtype=torch.float32)
    fixed_time = float(frame / frame_count)
    target = np.zeros(3, dtype=np.float64)
    world_up = np.asarray([0.0, 0.0, 1.0])
    cameras = []
    for index, (angle, camera_radius, camera_height) in enumerate(
        zip(sample_angles, radius_interp, height_interp, strict=True)
    ):
        center = np.asarray(
            [camera_radius * np.cos(angle), camera_radius * np.sin(angle), camera_height]
        )
        forward = target - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        rotation_world_to_camera = np.stack((right, down, forward))
        translation = -rotation_world_to_camera @ center
        cameras.append(
            Camera(
                colmap_id=index,
                R=rotation_world_to_camera.T,
                T=translation,
                FoVx=fovx,
                FoVy=fovy,
                image=dummy,
                gt_alpha_mask=None,
                image_name=f"orbit_{index:04d}",
                uid=index,
                data_device="cuda",
                time=fixed_time,
            )
        )
    return cameras


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    third_party = repo_root / "third_party" / "4DGaussians"
    sys.path.insert(0, str(third_party))

    # Imports must happen after official 4D-GS is placed first on sys.path.
    from arguments import ModelHiddenParams, PipelineParams  # noqa: PLC0415
    from gaussian_renderer import GaussianModel, render  # noqa: PLC0415
    from scene import Scene  # noqa: PLC0415
    from utils.system_utils import searchForMaxIteration  # noqa: PLC0415

    cfg_path = model_path / "cfg_args"
    cfg = eval(cfg_path.read_text(encoding="utf-8"), {"Namespace": Namespace})  # noqa: S307
    cfg.source_path = str(Path(cfg.source_path).resolve())
    cfg.model_path = str(model_path)
    cfg.data_device = "cuda"
    iteration = args.iteration
    if iteration < 0:
        iteration = searchForMaxIteration(str(model_path / "point_cloud"))

    # Extract only the official parameter groups so this remains compatible with
    # models whose cfg_args contains training-only keys.
    dummy_parser = argparse.ArgumentParser(add_help=False)
    hyper_group = ModelHiddenParams(dummy_parser)
    pipe_group = PipelineParams(dummy_parser)
    hyper = hyper_group.extract(cfg)
    pipeline = pipe_group.extract(cfg)

    torch.set_grad_enabled(False)
    gaussians = GaussianModel(cfg.sh_degree, hyper)
    scene = Scene(cfg, gaussians, load_iteration=iteration, shuffle=False)
    train = scene.getTrainCameras()
    cam_type = scene.dataset_type
    frame_count = args.frames_per_camera
    camera_count = len(train) // frame_count
    if camera_count * frame_count != len(train):
        raise ValueError(f"{len(train)} views is not divisible by {frame_count} frames")

    selected_root = output / "selected"
    selected_root.mkdir(exist_ok=True)
    records: list[dict] = []
    all_metrics: list[dict[str, float]] = []

    for frame in args.representative_frames:
        row_tiles = []
        for camera_index in range(camera_count):
            camera = train[camera_index * frame_count + frame]
            gt = tensor_rgb(camera.original_image[:3])
            prediction, depth, alpha, depth_rgb, limits = render_products(
                camera, gaussians, pipeline, render, cam_type
            )
            error = np.uint8(np.clip(np.abs(gt.astype(np.int16) - prediction.astype(np.int16)) * 3, 0, 255))
            alpha_rgb = np.repeat(np.uint8(alpha[..., None] * 255), 3, axis=2)
            panels = [
                label(gt, f"cam{camera_index + 1:02d} GT"),
                label(prediction, "4D-GS RGB"),
                label(error, "3x abs error"),
                label(depth_rgb, f"depth {limits[0]:.2f}-{limits[1]:.2f}"),
                label(alpha_rgb, "alpha"),
            ]
            row = np.concatenate(panels, axis=1)
            row_tiles.append(cv2.resize(row, (1280, 256), interpolation=cv2.INTER_AREA))
            stem = f"frame{frame:04d}_cam{camera_index + 1:02d}"
            np.save(selected_root / f"{stem}_depth.npy", depth.astype(np.float32))
            imageio.imwrite(selected_root / f"{stem}_alpha.png", alpha_rgb)
            metrics = metric_summary(gt, prediction)
            mask_path = (
                Path(cfg.source_path)
                / "masks"
                / f"cam{camera_index + 1:02d}"
                / f"frame_{frame + 1:05d}.png"
            )
            if mask_path.exists():
                reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                metrics.update(silhouette_metrics(reference_mask, alpha))
            metrics.update({"frame": frame, "camera": camera_index + 1})
            all_metrics.append(metrics)
            records.append(
                {
                    "frame": frame,
                    "camera": camera_index + 1,
                    "time": float(camera.time),
                    "depth_display_near_far": limits,
                    **metrics,
                }
            )
        montage = np.concatenate(row_tiles, axis=0)
        imageio.imwrite(output / f"frame{frame:04d}_sixview_diagnostics.jpg", montage, quality=94)

    # Observed-camera temporal sweep: verifies dynamics independently from novel view.
    observed_path = output / "cam01_gt_render_depth.mp4"
    with imageio.get_writer(observed_path, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in range(frame_count):
            camera = train[frame]
            gt = tensor_rgb(camera.original_image[:3])
            prediction, _, alpha, depth_rgb, _ = render_products(
                camera, gaussians, pipeline, render, cam_type
            )
            alpha_rgb = np.repeat(np.uint8(alpha[..., None] * 255), 3, axis=2)
            panels = [
                label(gt, "cam01 GT"),
                label(prediction, "4D-GS RGB"),
                label(depth_rgb, "rendered depth"),
                label(alpha_rgb, "predicted alpha"),
            ]
            canvas = np.concatenate(panels, axis=1)
            writer.append_data(cv2.resize(canvas, (2048, 512), interpolation=cv2.INTER_AREA))

    # Interpolate the calibrated rig positions while holding time fixed. The
    # repository's generic LLFF spiral is not appropriate for this cylindrical
    # DANNCE rig and can pass through scene geometry.
    fixed_time = float(args.orbit_frame / frame_count)
    orbit_cameras = interpolated_rig_orbit(train, frame_count, args.orbit_frame)
    orbit_path = output / f"fixed_time_orbit_frame{args.orbit_frame:04d}.mp4"
    with imageio.get_writer(orbit_path, fps=30.0, codec="libx264", quality=8) as writer:
        for camera in orbit_cameras:
            prediction, _, _, depth_rgb, _ = render_products(
                camera, gaussians, pipeline, render, cam_type
            )
            canvas = np.concatenate(
                [label(prediction, f"novel view, t={fixed_time:.3f}"), label(depth_rgb, "rendered depth")],
                axis=1,
            )
            writer.append_data(cv2.resize(canvas, (1024, 512), interpolation=cv2.INTER_AREA))

    if args.annotation_output is not None:
        annotation_output = args.annotation_output.resolve()
        source_manifest_path = Path(cfg.source_path) / "pilot_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        world_scale_mm = float(source_manifest["world_scale_mm"])
        for camera_index in range(camera_count):
            depth_root = annotation_output / f"cam{camera_index + 1:02d}" / "depth_mm_x10"
            alpha_root = annotation_output / f"cam{camera_index + 1:02d}" / "alpha"
            valid_root = annotation_output / f"cam{camera_index + 1:02d}" / "valid_mask"
            depth_root.mkdir(parents=True, exist_ok=True)
            alpha_root.mkdir(parents=True, exist_ok=True)
            valid_root.mkdir(parents=True, exist_ok=True)
            for frame in range(frame_count):
                camera = train[camera_index * frame_count + frame]
                _, depth, alpha, _, _ = render_products(
                    camera, gaussians, pipeline, render, cam_type
                )
                depth_mm_x10 = np.uint16(
                    np.clip(np.rint(depth * world_scale_mm * 10.0), 0, 65535)
                )
                valid = alpha > 0.5
                source_mask_path = (
                    Path(cfg.source_path)
                    / "masks"
                    / f"cam{camera_index + 1:02d}"
                    / f"frame_{frame + 1:05d}.png"
                )
                if source_mask_path.exists():
                    source_mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
                    if source_mask is None or source_mask.shape != valid.shape:
                        raise ValueError(f"Invalid source mask: {source_mask_path}")
                    valid &= source_mask > 127
                depth_mm_x10[~valid] = 0
                alpha_u8 = np.uint8(np.clip(np.rint(alpha * 255.0), 0, 255))
                valid_u8 = np.uint8(valid) * 255
                cv2.imwrite(
                    str(depth_root / f"frame_{frame + 1:05d}.png"), depth_mm_x10
                )
                cv2.imwrite(str(alpha_root / f"frame_{frame + 1:05d}.png"), alpha_u8)
                cv2.imwrite(
                    str(valid_root / f"frame_{frame + 1:05d}.png"), valid_u8
                )
        annotation_manifest = {
            "format": "mouse_4dgs_pseudo_depth_v1",
            "source_dataset": cfg.source_path,
            "source_model": str(model_path),
            "iteration": iteration,
            "camera_count": camera_count,
            "frame_count": frame_count,
            "depth_encoding": "uint16 PNG, camera-z millimetres multiplied by 10; 0 is invalid/background",
            "depth_scale_to_mm": 0.1,
            "alpha_encoding": "uint8 PNG, 0-255",
            "valid_mask_encoding": "uint8 PNG, 0 or 255",
            "valid_depth_rule": "predicted alpha > 0.5 intersected with the source SAM2 foreground mask when available",
            "supervision_status": "pseudo-label/weak supervision, not sensor ground truth",
        }
        (annotation_output / "annotation_manifest.json").write_text(
            json.dumps(annotation_manifest, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "model": str(model_path),
        "iteration": iteration,
        "source_path": cfg.source_path,
        "camera_count": camera_count,
        "frames_per_camera": frame_count,
        "representative_frames": args.representative_frames,
        "mean_mae": float(np.mean([row["mae"] for row in all_metrics])),
        "mean_psnr_db": float(np.mean([row["psnr_db"] for row in all_metrics])),
        "selected_records": records,
        "notes": {
            "depth_units": "alpha-normalized expected camera-z in scene units; preparation uses 1 unit = 100 mm",
            "alpha": "recovered from paired white/black-background renders",
            "orbit": "upright closed interpolation through calibrated rig centers with time fixed",
        },
    }
    silhouette_rows = [row for row in all_metrics if "silhouette_iou" in row]
    if silhouette_rows:
        summary["mean_silhouette_iou"] = float(
            np.mean([row["silhouette_iou"] for row in silhouette_rows])
        )
        summary["mean_silhouette_dice"] = float(
            np.mean([row["silhouette_dice"] for row in silhouette_rows])
        )
    (output / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "selected_records"}, indent=2))


if __name__ == "__main__":
    main()

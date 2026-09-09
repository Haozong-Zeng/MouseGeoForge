#!/usr/bin/env python3
"""Estimate approximate dual-mouse body heights from CRIM13 top-view images.

The method follows the PyMouseLifter principle: monocular depth is sampled at
tracked anatomy, while a scene-derived scale converts relative depth to useful
metric geometry.  Here the known Homecage floor scale (mm/pixel) and VGGT's
predicted focal length replace multi-camera static 3D anchors.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATER_ROOT = PROJECT_ROOT / "water-reaching-dataset"
if str(WATER_ROOT) not in sys.path:
    sys.path.insert(0, str(WATER_ROOT))

from evaluate_vggt_omega_keypoints import (  # noqa: E402
    VGGTOmega,
    encoding_to_camera,
    load_checkpoint_chain,
    preprocess_rgb_frames,
)


DEFAULT_PLAN = PROJECT_ROOT / "dataset/Homecage/LightningPose-TwoColor-DualMouse-Pilot.plan.json"
DEFAULT_LABELS = PROJECT_ROOT / "dataset/LightningPose-CRIM13/crim13/labels_InD.csv"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "output/blenderpymouse_vggt_omega_all/"
    "run_20260718_real_palm_dense_lr5e7_e20_v2/checkpoints/"
    "vggt_omega_real_palm_dense_latest_best_real_palm.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset/Homecage/depth_height/vggt_crim13_dual_mouse"
IDENTITIES = ("black_mouse", "white_mouse")
BODY_PARTS = {
    "nose": "nose",
    "right_ear": "right_ear",
    "left_ear": "left_ear",
    "neck": "top_of_neck",
    "right_knee": "right_rear_knee",
    "left_knee": "left_rear_knee",
    "tail_base": "base_of_tail",
}
NEUTRAL_HEIGHTS_MM = np.asarray((14.0, 20.0, 24.0, 21.0, 16.0), dtype=np.float64)
HEIGHT_LIMITS_MM = np.asarray(
    ((6.0, 21.0), (10.0, 30.0), (12.0, 38.0), (10.0, 45.0), (8.0, 52.0)),
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", default="real_palm_best")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--sample-radius", type=int, default=4)
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        help=(
            "Reuse matching candidate reports and depth maps from an earlier plan; "
            "only source frames absent there are inferred."
        ),
    )
    return parser.parse_args()


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def read_labels(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    table.index = table.index.astype(str).str.replace("\\\\", "/", regex=False)
    return table


def label_key(image_path: Path) -> str:
    parts = image_path.as_posix().split("/crim13/")
    if len(parts) != 2:
        raise ValueError(f"Could not derive labels_InD key from {image_path}")
    return parts[1]


def keypoints_for_row(row: pd.Series, identity: str) -> dict[str, np.ndarray]:
    result = {}
    for output_name, source_name in BODY_PARTS.items():
        bodypart = f"{identity}_{source_name}"
        matches = [column for column in row.index if column[1] == bodypart]
        if len(matches) != 2:
            raise KeyError(f"Expected x/y for {bodypart}, got {matches}")
        values = {column[2]: float(row[column]) for column in matches}
        result[output_name] = np.asarray((values["x"], values["y"]), dtype=np.float64)
    return result


def centerline_points(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    knee_center = (keypoints["left_knee"] + keypoints["right_knee"]) * 0.5
    root = keypoints["tail_base"]
    return np.stack((
        root,
        root * 0.52 + knee_center * 0.48,
        knee_center * 0.58 + keypoints["neck"] * 0.42,
        keypoints["neck"],
        keypoints["nose"],
    ))


def body_mask(shape: tuple[int, int], keypoints: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    body_names = ("tail_base", "left_knee", "right_knee", "neck", "left_ear", "right_ear", "nose")
    hull = cv2.convexHull(np.round(np.stack([keypoints[name] for name in body_names])).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    centerline = np.round(centerline_points(keypoints)).astype(np.int32)
    cv2.polylines(mask, [centerline], False, 255, thickness=26, lineType=cv2.LINE_AA)
    cv2.circle(mask, tuple(centerline[2]), 24, 255, -1, lineType=cv2.LINE_AA)
    return mask


def map_mask(mask: np.ndarray, transform, output_shape: tuple[int, int]) -> np.ndarray:
    cropped = mask[
        transform.crop_top:transform.crop_top + transform.crop_height,
        transform.crop_left:transform.crop_left + transform.crop_width,
    ]
    resized = cv2.resize(
        cropped,
        (transform.target_width, transform.target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros(output_shape, dtype=np.uint8)
    y0, x0 = transform.pad_top, transform.pad_left
    output[y0:y0 + transform.target_height, x0:x0 + transform.target_width] = resized
    return output


def patch_values(array: np.ndarray, xy: np.ndarray, radius: int) -> np.ndarray:
    x, y = (int(round(float(value))) for value in xy)
    y0, y1 = max(0, y - radius), min(array.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(array.shape[1], x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return np.empty(0, dtype=np.float64)
    values = np.asarray(array[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    return values[np.isfinite(values) & (values > 0)]


def sample_patch(array: np.ndarray, xy: np.ndarray, radius: int) -> float:
    values = patch_values(array, xy, radius)
    return float(np.median(values)) if len(values) else float("nan")


def floor_depth_for_mouse(
    depth: np.ndarray,
    confidence: np.ndarray,
    own_mask: np.ndarray,
    union_mask: np.ndarray,
) -> tuple[float, int]:
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39))
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    inner = cv2.dilate(own_mask, kernel_inner)
    outer = cv2.dilate(own_mask, kernel_outer)
    occupied = cv2.dilate(union_mask, kernel_inner)
    ring = (outer > 0) & (inner == 0) & (occupied == 0)
    valid = ring & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if np.count_nonzero(valid) < 80:
        valid = (outer > 0) & (occupied == 0) & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    conf = confidence[valid]
    if len(values) < 20:
        raise RuntimeError("Insufficient local floor pixels around mouse mask")
    conf_cutoff = float(np.quantile(conf, 0.35)) if len(conf) else -np.inf
    values = values[conf >= conf_cutoff]
    # Bedding and paws are closer than the support floor in camera depth. The
    # upper-middle quantile is more stable than the maximum and rejects the
    # metal tube while preserving the local floor level.
    return float(np.quantile(values, 0.68)), int(len(values))


def reliability_weight(confidence_samples: np.ndarray, local_floor_confidence: np.ndarray) -> float:
    object_conf = float(np.nanmedian(confidence_samples))
    floor_conf = float(np.nanmedian(local_floor_confidence))
    if not np.isfinite(object_conf) or not np.isfinite(floor_conf) or floor_conf <= 0:
        return 0.25
    ratio = object_conf / floor_conf
    return float(np.clip(0.30 + 0.45 * ratio, 0.25, 0.82))


def estimate_instance(
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    transform,
    keypoints: dict[str, np.ndarray],
    own_mask_model: np.ndarray,
    union_mask_model: np.ndarray,
    pixel_to_mm: float,
    sample_radius: int,
) -> dict:
    points_original = centerline_points(keypoints)
    points_model = transform.map_xy(points_original)
    floor_depth, floor_samples = floor_depth_for_mouse(
        depth, confidence, own_mask_model, union_mask_model
    )
    scale_x = transform.target_width / transform.crop_width
    scale_y = transform.target_height / transform.crop_height
    model_pixel_to_mm = pixel_to_mm / ((scale_x + scale_y) * 0.5)
    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) * 0.5)
    metric_scale = model_pixel_to_mm * focal / max(floor_depth, 1.0e-9)
    sampled_depth = np.asarray([
        sample_patch(depth, point, sample_radius) for point in points_model
    ])
    sampled_conf = np.asarray([
        sample_patch(confidence, point, sample_radius) for point in points_model
    ])
    raw_height = (floor_depth - sampled_depth) * metric_scale
    clipped = np.clip(raw_height, HEIGHT_LIMITS_MM[:, 0], HEIGHT_LIMITS_MM[:, 1])
    ring_conf = confidence[(cv2.dilate(own_mask_model, np.ones((41, 41), np.uint8)) > 0) & (union_mask_model == 0)]
    weight = reliability_weight(sampled_conf, ring_conf)
    height = NEUTRAL_HEIGHTS_MM * (1.0 - weight) + clipped * weight
    # A tail base cannot rise with an occluding tube, and adjacent spline
    # points should remain reachable by the neutral rig.
    height[0] = float(np.clip(height[0], 8.0, 18.0))
    for index in range(1, len(height)):
        height[index] = float(np.clip(height[index], height[index - 1] - 12.0, height[index - 1] + 18.0))
    return {
        "centerline_xy_original": points_original.tolist(),
        "centerline_xy_model": points_model.tolist(),
        "sampled_depth": sampled_depth.tolist(),
        "sampled_depth_conf": sampled_conf.tolist(),
        "floor_depth": floor_depth,
        "floor_samples": floor_samples,
        "predicted_focal_px": focal,
        "model_pixel_to_mm": model_pixel_to_mm,
        "depth_metric_scale_mm_per_model_unit": metric_scale,
        "raw_height_mm": raw_height.tolist(),
        "clipped_height_mm": clipped.tolist(),
        "reliability_weight": weight,
        "height_mm": height.tolist(),
    }


def depth_preview(
    rgb: np.ndarray,
    depth: np.ndarray,
    transform,
    instances: dict[str, dict],
    output: Path,
) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0)]
    low, high = np.quantile(finite, (0.03, 0.97))
    normalized = np.clip((depth - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resized_original = cv2.resize(original, (colored.shape[1], colored.shape[0]))
    overlay = cv2.addWeighted(colored, 0.72, resized_original, 0.28, 0.0)
    colors = {"black_mouse": (70, 220, 255), "white_mouse": (255, 180, 60)}
    for identity, record in instances.items():
        points = np.round(np.asarray(record["centerline_xy_model"])).astype(int)
        heights = record["height_mm"]
        cv2.polylines(overlay, [points], False, colors[identity], 2, cv2.LINE_AA)
        for index, (point, height) in enumerate(zip(points, heights, strict=True)):
            cv2.circle(overlay, tuple(point), 4, colors[identity], -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"{index}:{height:.1f}mm",
                (int(point[0]) + 5, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                colors[identity],
                1,
                cv2.LINE_AA,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve()
    labels_path = args.labels.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    labels = read_labels(labels_path)
    pixel_to_mm = float(plan["homecage_mapping"]["pixel_to_mm"])
    records = []
    for keyframe in plan["keyframes"]:
        image_path = (PROJECT_ROOT / keyframe["source_image"]).resolve()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        row = labels.loc[label_key(image_path)]
        keypoints = {identity: keypoints_for_row(row, identity) for identity in IDENTITIES}
        records.append({
            "scene_frame": int(keyframe["scene_frame"]),
            "source_image_frame": int(keyframe["source_image_frame"]),
            "source_image": str(image_path),
            "rgb": rgb,
            "keypoints": keypoints,
        })

    reused_records: dict[int, dict] = {}
    depth_by_source: dict[int, np.ndarray] = {}
    confidence_by_source: dict[int, np.ndarray] = {}
    if args.reuse_dir is not None:
        reuse_dir = args.reuse_dir.resolve()
        reuse_report_path = reuse_dir / f"{args.candidate}_body_height.json"
        reuse_arrays_path = reuse_dir / f"{args.candidate}_depth_maps.npz"
        reuse_report = json.loads(reuse_report_path.read_text(encoding="utf-8"))
        if reuse_report.get("candidate") != args.candidate:
            raise ValueError(
                f"Reuse candidate mismatch: {reuse_report.get('candidate')!r} != {args.candidate!r}"
            )
        report_by_source = {
            int(row["source_image_frame"]): row for row in reuse_report["keyframes"]
        }
        with np.load(reuse_arrays_path) as arrays:
            reuse_frames = arrays["source_image_frames"].astype(int)
            reuse_depth = arrays["depth"]
            reuse_confidence = arrays["depth_conf"]
            for index, source_frame in enumerate(reuse_frames):
                source_frame = int(source_frame)
                if source_frame not in report_by_source:
                    continue
                reused_records[source_frame] = report_by_source[source_frame]
                depth_by_source[source_frame] = reuse_depth[index].copy()
                confidence_by_source[source_frame] = reuse_confidence[index].copy()

    inference_records = [
        record for record in records
        if record["source_image_frame"] not in reused_records
    ]
    device = torch.device(args.device)
    model = None
    if inference_records:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        model = VGGTOmega().to(device).eval()
        load_checkpoint_chain(model, checkpoint)
    with torch.inference_mode():
        for batch_index, batch in enumerate(chunks(inference_records, args.batch_size), start=1):
            tensor, transforms = preprocess_rgb_frames(
                [record["rgb"] for record in batch],
                mode=args.preprocess_mode,
                image_resolution=args.image_resolution,
                patch_size=args.patch_size,
            )
            images = tensor[:, None].to(device)
            predictions = model(images)
            depth = predictions["depth"][:, 0, ..., 0].detach().float().cpu().numpy()
            confidence = predictions["depth_conf"][:, 0].detach().float().cpu().numpy()
            _extrinsics, intrinsics_t = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
            intrinsics = intrinsics_t[:, 0].detach().float().cpu().numpy()
            for sample_index, (record, transform) in enumerate(zip(batch, transforms, strict=True)):
                model_shape = depth[sample_index].shape
                original_shape = record["rgb"].shape[:2]
                masks_original = {
                    identity: body_mask(original_shape, record["keypoints"][identity])
                    for identity in IDENTITIES
                }
                masks_model = {
                    identity: map_mask(mask, transform, model_shape)
                    for identity, mask in masks_original.items()
                }
                union_mask = np.maximum.reduce(list(masks_model.values()))
                instances = {
                    identity: estimate_instance(
                        depth[sample_index],
                        confidence[sample_index],
                        intrinsics[sample_index],
                        transform,
                        record["keypoints"][identity],
                        masks_model[identity],
                        union_mask,
                        pixel_to_mm,
                        args.sample_radius,
                    )
                    for identity in IDENTITIES
                }
                record["model_input_hw"] = list(model_shape)
                record["intrinsics"] = intrinsics[sample_index].tolist()
                record["instances"] = instances
                record["transform"] = {
                    key: int(getattr(transform, key))
                    for key in (
                        "original_width", "original_height", "crop_left", "crop_top",
                        "crop_width", "crop_height", "target_width", "target_height",
                        "pad_left", "pad_top",
                    )
                }
                preview_path = output_dir / f"frame_{record['source_image_frame']:08d}_{args.candidate}_depth_height.png"
                depth_preview(record["rgb"], depth[sample_index], transform, instances, preview_path)
                record["preview"] = str(preview_path)
                source_frame = int(record["source_image_frame"])
                depth_by_source[source_frame] = depth[sample_index]
                confidence_by_source[source_frame] = confidence[sample_index]
            print(f"batch {batch_index}: processed {len(batch)} images", flush=True)
            del images, predictions
            torch.cuda.empty_cache()

    serializable = []
    for record in records:
        source_frame = int(record["source_image_frame"])
        if source_frame in reused_records:
            reused = dict(reused_records[source_frame])
            reused["scene_frame"] = int(record["scene_frame"])
            reused["source_image_frame"] = source_frame
            reused["source_image"] = str(record["source_image"])
            serializable.append(reused)
        else:
            serializable.append({
                key: value for key, value in record.items()
                if key not in {"rgb", "keypoints"}
            })
    report = {
        "schema": "mousegeoforge.crim13_vggt_monocular_body_height.v1",
        "plan": str(plan_path),
        "labels": str(labels_path),
        "candidate": args.candidate,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "pixel_to_mm": pixel_to_mm,
        "scale_method": "known floor mm/pixel + VGGT predicted focal/depth",
        "floor_method": "confidence-filtered local annulus upper-middle depth quantile",
        "height_policy": "depth estimate blended with neutral physiology and per-control bounds",
        "spine_controls": ["tail_base", "rear_mid", "body_mid", "neck", "nose"],
        "reuse": {
            "source_dir": str(args.reuse_dir.resolve()) if args.reuse_dir is not None else None,
            "reused_source_frame_count": sum(
                int(record["source_image_frame"]) in reused_records
                for record in records
            ),
            "inferred_source_frame_count": len(inference_records),
        },
        "keyframes": serializable,
    }
    report_path = output_dir / f"{args.candidate}_body_height.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / f"{args.candidate}_depth_maps.npz",
        source_image_frames=np.asarray([record["source_image_frame"] for record in records]),
        depth=np.stack([
            depth_by_source[int(record["source_image_frame"])] for record in records
        ]),
        depth_conf=np.stack([
            confidence_by_source[int(record["source_image_frame"])] for record in records
        ]),
    )
    summary = {
        identity: {
            "minimum_height_mm": float(min(min(row["instances"][identity]["height_mm"]) for row in serializable)),
            "maximum_height_mm": float(max(max(row["instances"][identity]["height_mm"]) for row in serializable)),
            "median_reliability": float(np.median([
                row["instances"][identity]["reliability_weight"] for row in serializable
            ])),
        }
        for identity in IDENTITIES
    }
    print(json.dumps({"report": str(report_path), "summary": summary}, indent=2), flush=True)
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

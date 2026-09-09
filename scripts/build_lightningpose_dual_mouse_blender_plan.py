#!/usr/bin/env python3
"""Build a sparse CRIM13/LightningPose plan for a two-mouse Blender pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE23 = (
    "nose", "head_center", "left_ear", "right_ear", "neck", "spine_center",
    "lumbar_spine", "tail_base", "tail_q1", "tail_q2", "tail_q3", "tail_q4",
    "tail_q5", "tail_tip", "left_forepaw_center", "right_forepaw_center",
    "left_hindpaw_center", "right_hindpaw_center", "left_elbow", "right_elbow",
    "left_knee", "right_knee", "abdomen_center",
)
SOURCE_TO_CORE23 = {
    "nose": "nose",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "top_of_neck": "neck",
    "base_of_tail": "tail_base",
    "left_rear_knee": "left_knee",
    "right_rear_knee": "right_knee",
}
CONDITIONING = ("nose", "left_ear", "right_ear", "neck", "left_knee", "right_knee", "tail_base")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def descriptor(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    return centered / max(float(np.linalg.norm(centered)), 1.0e-8)


def similarity_residual(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    variance = float(np.square(source_centered).sum() / len(source))
    scale = float(singular.sum() / max(variance, 1.0e-8))
    predicted = scale * (source_centered @ rotation.T) + target_mean
    return float(np.linalg.norm(predicted - target, axis=1).mean()), scale


def load_pose_bank(hybrid_root: Path) -> list[dict]:
    bank = []
    for split in ("train", "val", "test"):
        for record in load_jsonl(hybrid_root / f"{split}.jsonl"):
            if record["source"] != "homecage_blackfur_blender":
                continue
            if record.get("keypoint_world_xyz") is None:
                continue
            if 50.0 <= record_body_length_mm(record) <= 100.0:
                bank.append(record)
    return sorted(bank, key=lambda record: record["sample_id"])


def source_pose(record: dict) -> np.ndarray:
    width, height = record["image_size"]
    points = np.asarray(record["keypoint_xy"][0][0], dtype=np.float64)
    return points * np.asarray((width, height), dtype=np.float64)


def record_body_length_mm(record: dict) -> float:
    points = np.asarray(record["keypoint_world_xyz"][0], dtype=np.float64)
    return float(np.linalg.norm(points[CORE23.index("nose")] - points[CORE23.index("tail_base")]))


def best_pose(
    target_by_name: dict[str, np.ndarray],
    bank: list[dict],
    target_body_length_mm: float,
) -> tuple[dict, float, float]:
    indices = [CORE23.index(name) for name in CONDITIONING]
    target = np.asarray([target_by_name[name] for name in CONDITIONING])
    target_descriptor = descriptor(target)
    best = None
    for record in bank:
        source = source_pose(record)[indices]
        shape_error = float(np.square(descriptor(source) - target_descriptor).mean())
        residual, scale = similarity_residual(source, target)
        length_error = abs(math.log(record_body_length_mm(record) / target_body_length_mm))
        candidate = (shape_error + 0.02 * length_error, residual, record, scale)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("empty Blender pose bank")
    return best[2], best[1], best[3]


def load_actual_pose_bank(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload["poses"]
    if not poses:
        raise RuntimeError(f"empty actual Blender pose bank: {path}")
    return poses


def best_actual_pose(
    target_by_name: dict[str, np.ndarray],
    bank: list[dict],
) -> tuple[dict, float, float]:
    target = np.asarray([target_by_name[name] for name in CONDITIONING], dtype=np.float64)
    target_length = float(np.linalg.norm(target_by_name["nose"] - target_by_name["tail_base"]))
    best = None
    for record in bank:
        source = np.asarray([
            record["keypoints_world_xyz"][name][:2] for name in CONDITIONING
        ], dtype=np.float64)
        residual_mm, fitted_scale = similarity_residual(source, target)
        normalized_residual = residual_mm / max(target_length, 1.0e-6)
        length_error = abs(math.log(record["body_length_mm"] / target_length))
        candidate = (
            normalized_residual + 0.02 * length_error,
            residual_mm,
            record,
            fitted_scale,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2], best[1], best[3]


def load_source_rows(csv_path: Path, session: str, start: int, count: int) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    bodyparts = rows[1][1:]
    coords = rows[2][1:]
    candidates = []
    for row in rows[3:]:
        if f"/{session}/" not in f"/{row[0]}":
            continue
        match = re.search(r"img(\d+)\.png$", row[0])
        if match is None:
            continue
        values: dict[str, dict[str, float]] = {}
        for bodypart, coord, value in zip(bodyparts, coords, row[1:], strict=True):
            values.setdefault(bodypart, {})[coord] = float(value)
        candidates.append({"path": row[0], "image_frame": int(match.group(1)), "values": values})
    candidates.sort(key=lambda item: item["image_frame"])
    selected = [item for item in candidates if item["image_frame"] >= start][:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but found {len(selected)}")
    return selected


def instance_points(row: dict, identity: str) -> dict[str, np.ndarray]:
    output = {}
    prefix = f"{identity}_mouse_"
    for source_suffix, target_name in SOURCE_TO_CORE23.items():
        values = row["values"][prefix + source_suffix]
        output[target_name] = np.asarray((values["x"], values["y"]), dtype=np.float64)
    return output


def image_to_homecage(
    point: np.ndarray,
    segment_center_px: np.ndarray,
    pixel_to_mm: float,
    scene_center_mm: np.ndarray,
) -> np.ndarray:
    # Preserve the local interaction scale, recenter the selected fragment, and
    # invert image Y into Blender world Y.  Mapping the entire 640x480 image into
    # the cage made the mice physically half-size because the image includes a
    # large border outside the bedding.
    centered = np.asarray(point, dtype=np.float64) - segment_center_px
    return scene_center_mm + np.asarray((centered[0], -centered[1])) * pixel_to_mm


def separate_close_centers(instances: list[dict], minimum_mm: float) -> float:
    first = np.asarray(instances[0]["target_center_xy_mm"], dtype=np.float64)
    second = np.asarray(instances[1]["target_center_xy_mm"], dtype=np.float64)
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance >= minimum_mm:
        return 0.0
    if distance < 1.0e-6:
        delta = np.asarray((1.0, 0.0))
        distance = 1.0
    correction = minimum_mm - distance
    direction = delta / distance
    for sign, instance in ((-1.0, instances[0]), (1.0, instances[1])):
        shift = sign * direction * correction * 0.5
        instance["target_tail_base_xy_mm"] = (
            np.asarray(instance["target_tail_base_xy_mm"]) + shift
        ).tolist()
        instance["target_center_xy_mm"] = (
            np.asarray(instance["target_center_xy_mm"]) + shift
        ).tolist()
        instance["body_contact_shift_xy_mm"] = shift.tolist()
    return correction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="021709_A25_Block22_BCfe1_t")
    parser.add_argument("--source-start", type=int, default=5306)
    parser.add_argument("--keyframes", type=int, default=8)
    parser.add_argument("--scene-step", type=int, default=8)
    parser.add_argument("--pixel-to-mm", type=float, default=0.55)
    parser.add_argument("--scene-center-x-mm", type=float, default=-15.0)
    parser.add_argument("--scene-center-y-mm", type=float, default=-15.0)
    parser.add_argument("--minimum-body-center-mm", type=float, default=38.0)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "dataset/LightningPose-CRIM13/crim13/labels_InD.csv",
    )
    parser.add_argument(
        "--hybrid-root",
        type=Path,
        default=ROOT / "dataset/Homecage-LightningPose-Core23-Hybrid-v1",
    )
    parser.add_argument(
        "--actual-pose-bank",
        type=Path,
        default=(
            ROOT
            / "dataset/Homecage/Homecage-BlackFur-KeypointMoseq-Actual7-PoseBank.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset/Homecage/LightningPose-TwoColor-DualMouse-Pilot.plan.json",
    )
    args = parser.parse_args()
    actual_bank = load_actual_pose_bank(args.actual_pose_bank)
    rows = load_source_rows(args.labels, args.session, args.source_start, args.keyframes)
    parsed_points = [
        [instance_points(row, identity) for identity in ("black", "white")]
        for row in rows
    ]
    segment_center_px = np.mean(
        np.asarray([
            point
            for frame in parsed_points
            for instance in frame
            for point in instance.values()
        ]),
        axis=0,
    )
    scene_center_mm = np.asarray(
        (args.scene_center_x_mm, args.scene_center_y_mm), dtype=np.float64
    )

    keyframes = []
    for index, (row, frame_points) in enumerate(zip(rows, parsed_points, strict=True)):
        instances = []
        for instance_index, (identity, appearance) in enumerate(
            (("black", "black_c57"), ("white", "brown_agouti"))
        ):
            points = frame_points[instance_index]
            world = {
                name: image_to_homecage(
                    point,
                    segment_center_px,
                    args.pixel_to_mm,
                    scene_center_mm,
                )
                for name, point in points.items()
            }
            matched, residual_mm, fitted_scale = best_actual_pose(world, actual_bank)
            center = np.mean(np.asarray(list(world.values())), axis=0)
            heading = world["nose"] - world["tail_base"]
            instances.append({
                "source_identity": f"{identity}_mouse",
                "appearance": appearance,
                "source_pose_sample_id": f"actual_blender_frame:{matched['frame']:06d}",
                "source_pose_frame": int(matched["frame"]),
                "conditioning_residual_mm": residual_mm,
                "conditioning_residual_px": residual_mm / args.pixel_to_mm,
                "fitted_source_scale": fitted_scale,
                "target_tail_base_xy_mm": world["tail_base"].tolist(),
                "target_center_xy_mm": center.tolist(),
                "target_heading_xy": (heading / max(np.linalg.norm(heading), 1.0e-8)).tolist(),
                "target_keypoints_xy_mm": {name: value.tolist() for name, value in world.items()},
                "body_contact_shift_xy_mm": [0.0, 0.0],
            })
        correction = separate_close_centers(instances, args.minimum_body_center_mm)
        keyframes.append({
            "scene_frame": 1 + index * args.scene_step,
            "source_image_frame": row["image_frame"],
            "source_image": f"dataset/LightningPose-CRIM13/crim13/{row['path']}",
            "minimum_center_correction_mm": correction,
            "instances": instances,
        })

    plan = {
        "schema": "mousegeoforge.lightningpose_dual_mouse_blender_plan.v1",
        "source_dataset": "Lightning Pose CRIM13",
        "source_session": args.session,
        "source_label_frames": [row["image_frame"] for row in rows],
        "source_label_stride_note": "sparse labels; interpolated Blender frames are synthetic, not real frame labels",
        "scene_frame_range": [1, keyframes[-1]["scene_frame"]],
        "scene_fps": 8,
        "appearance_mapping": {"black_mouse": "black_c57", "white_mouse": "brown_agouti"},
        "homecage_mapping": {
            "segment_center_px": segment_center_px.tolist(),
            "pixel_to_mm": args.pixel_to_mm,
            "scene_center_mm": scene_center_mm.tolist(),
        },
        "minimum_body_center_mm": args.minimum_body_center_mm,
        "conditioning_keypoints": list(CONDITIONING),
        "actual_pose_bank": str(args.actual_pose_bank),
        "keyframes": keyframes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "keyframes": len(keyframes),
        "scene_frames": plan["scene_frame_range"],
        "mean_residual_px": float(np.mean([
            instance["conditioning_residual_px"]
            for keyframe in keyframes for instance in keyframe["instances"]
        ])),
        "mean_residual_mm": float(np.mean([
            instance["conditioning_residual_mm"]
            for keyframe in keyframes for instance in keyframe["instances"]
        ])),
        "contact_corrected_keyframes": sum(k["minimum_center_correction_mm"] > 0 for k in keyframes),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select diverse labeled CRIM13 poses and build a discrete Blender plan.

The output is deliberately not an interpolation schedule: every Blender frame
maps to one real, labeled source frame.  Greedy farthest-point sampling covers
single-mouse shape, relative heading/distance, and arena location jointly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRIM_ROOT = PROJECT_ROOT / "dataset/LightningPose-CRIM13/crim13"
DEFAULT_LABELS = CRIM_ROOT / "labels_InD.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset/Homecage/LightningPose-CRIM13-Diverse200.plan.json"
IDENTITIES = ("black_mouse", "white_mouse")
PARTS = (
    "nose", "right_ear", "left_ear", "top_of_neck",
    "right_rear_knee", "left_rear_knee", "base_of_tail",
)
OUTPUT_NAMES = {
    "nose": "nose",
    "right_ear": "right_ear",
    "left_ear": "left_ear",
    "top_of_neck": "neck",
    "right_rear_knee": "right_knee",
    "left_rear_knee": "left_knee",
    "base_of_tail": "tail_base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--session", help="Force one labeled-data session instead of automatic ranking")
    parser.add_argument("--pixel-to-mm", type=float, default=0.55)
    parser.add_argument("--minimum-center-mm", type=float, default=38.0)
    parser.add_argument(
        "--pair-midpoint-scale",
        type=float,
        default=0.20,
        help="Map source pair midpoint pixels into safe synthetic-arena millimetres.",
    )
    parser.add_argument(
        "--social-scale",
        type=float,
        default=0.12,
        help="Scale only the portion of inter-mouse distance beyond 90 mm.",
    )
    parser.add_argument("--seed", type=int, default=5706)
    parser.add_argument(
        "--exclude-source-frames",
        default="",
        help="Comma-separated source frames rejected by geometry QA.",
    )
    parser.add_argument(
        "--base-plan",
        type=Path,
        help="Keep the base plan's accepted frames and replace only excluded sources.",
    )
    return parser.parse_args()


def read_labels(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    table.index = table.index.astype(str).str.replace("\\", "/", regex=False)
    return table


def session_from_key(key: str) -> str:
    match = re.search(r"labeled-data_InD/([^/]+)/", key)
    if not match:
        raise ValueError(f"Unexpected label path: {key}")
    return match.group(1)


def frame_from_key(key: str) -> int:
    match = re.search(r"img(\d+)\.png$", key)
    if not match:
        raise ValueError(f"Unexpected image filename: {key}")
    return int(match.group(1))


def xy_from_row(row: pd.Series, identity: str) -> np.ndarray:
    points = []
    for part in PARTS:
        bodypart = f"{identity}_{part}"
        values = {}
        for column in row.index:
            if column[1] == bodypart:
                values[column[2]] = float(row[column])
        points.append((values.get("x", np.nan), values.get("y", np.nan)))
    return np.asarray(points, dtype=np.float64)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1.0e-8)


def descriptor(black: np.ndarray, white: np.ndarray) -> np.ndarray:
    per_mouse = []
    centers = []
    headings = []
    lengths = []
    for points in (black, white):
        nose, right_ear, left_ear, neck, right_knee, left_knee, tail = points
        heading = unit(nose - tail)
        lateral = np.asarray((-heading[1], heading[0]))
        length = float(np.linalg.norm(nose - tail))
        center = np.mean(points, axis=0)
        local = points - center
        normalized = np.stack((local @ heading, local @ lateral), axis=1) / max(length, 1.0)
        knee_width = abs(float((left_knee - right_knee) @ lateral)) / max(length, 1.0)
        ear_width = abs(float((left_ear - right_ear) @ lateral)) / max(length, 1.0)
        neck_bend = float((neck - (tail + 0.67 * (nose - tail))) @ lateral) / max(length, 1.0)
        per_mouse.extend(normalized.reshape(-1))
        per_mouse.extend((np.log(max(length, 1.0)), knee_width, ear_width, neck_bend))
        centers.append(center)
        headings.append(heading)
        lengths.append(length)
    pair_mid = (centers[0] + centers[1]) * 0.5
    separation = centers[1] - centers[0]
    scale = max(float(np.mean(lengths)), 1.0)
    bearing = np.asarray((separation @ headings[0], separation @ np.asarray((-headings[0][1], headings[0][0])))) / scale
    relative_heading = (
        float(headings[0] @ headings[1]),
        float(headings[0][0] * headings[1][1] - headings[0][1] * headings[1][0]),
    )
    arena = np.asarray((pair_mid[0] / 640.0, pair_mid[1] / 480.0))
    return np.concatenate((np.asarray(per_mouse), bearing, relative_heading, arena))


def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    low, high = np.quantile(values, (0.10, 0.90), axis=0)
    scale = np.maximum(high - low, 1.0e-4)
    return np.clip((values - median) / scale, -4.0, 4.0)


def farthest_point_indices(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) < count:
        raise ValueError(f"Need {count} frames but only {len(values)} are valid")
    center = np.median(values, axis=0)
    first = int(np.argmax(np.sum((values - center) ** 2, axis=1)))
    selected = [first]
    minimum_distance = np.sum((values - values[first]) ** 2, axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.sum((values - values[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def session_metrics(values: np.ndarray, selection: np.ndarray) -> dict:
    chosen = values[selection]
    spread = np.quantile(chosen, 0.95, axis=0) - np.quantile(chosen, 0.05, axis=0)
    # Coverage is the mean distance from every labeled frame to its nearest
    # selected representative; diversity measures selected-set extent.
    nearest = np.full(len(values), np.inf)
    for row in chosen:
        nearest = np.minimum(nearest, np.sqrt(np.sum((values - row) ** 2, axis=1)))
    return {
        "valid_frames": int(len(values)),
        "selected_frames": int(len(selection)),
        "mean_feature_spread": float(np.mean(spread)),
        "median_feature_spread": float(np.median(spread)),
        "mean_coverage_distance": float(np.mean(nearest)),
        "p95_coverage_distance": float(np.quantile(nearest, 0.95)),
        "selection_score": float(np.mean(spread) - 0.18 * np.mean(nearest)),
    }


def converted_points(
    points_px: np.ndarray,
    pair_mid_px: np.ndarray,
    pixel_to_mm: float,
    pair_midpoint_scale: float,
    social_scale: float,
) -> np.ndarray:
    # Keep true local/social scale, but normalize the pair midpoint into the
    # synthetic arena so full-video coverage does not put mice through walls.
    mouse_center_px = np.mean(points_px, axis=0)
    local = points_px - mouse_center_px
    social_offset = mouse_center_px - pair_mid_px
    target_pair_mid = np.asarray((
        (pair_mid_px[0] - 320.0) * pair_midpoint_scale,
        (240.0 - pair_mid_px[1]) * pair_midpoint_scale,
    ))
    local_scene = np.stack((local[:, 0], -local[:, 1]), axis=1) * pixel_to_mm
    social_scene = np.asarray((social_offset[0], -social_offset[1])) * social_scale
    return target_pair_mid + social_scene + local_scene


def fit_pair_with_tail_room(target_points: list[np.ndarray]) -> list[np.ndarray]:
    """Translate both mice together, reserving room behind each tail base."""
    preferred_tail_ends = []
    for points in target_points:
        heading = unit(points[0] - points[-1])
        preferred_tail_ends.append(points[-1] - heading * 65.0)
    body_geometry = np.concatenate(target_points, axis=0)
    geometry = np.concatenate((body_geometry, np.stack(preferred_tail_ends)), axis=0)
    body_low = np.min(body_geometry, axis=0)
    body_high = np.max(body_geometry, axis=0)
    preferred_low = np.min(geometry, axis=0)
    preferred_high = np.max(geometry, axis=0)
    shift = np.zeros(2)
    for axis, bounds in enumerate(((-112.0, 112.0), (-78.0, 78.0))):
        body_span = body_high[axis] - body_low[axis]
        if body_span > bounds[1] - bounds[0]:
            shift[axis] = (
                bounds[0] + bounds[1] - body_low[axis] - body_high[axis]
            ) * 0.5
            continue
        feasible_low = bounds[0] - body_low[axis]
        feasible_high = bounds[1] - body_high[axis]
        preferred = (
            bounds[0] + bounds[1] - preferred_low[axis] - preferred_high[axis]
        ) * 0.5
        shift[axis] = float(np.clip(preferred, feasible_low, feasible_high))
    return [points + shift for points in target_points]


def make_plan(
    table: pd.DataFrame,
    session: str,
    selected_keys: list[str],
    pixel_to_mm: float,
    minimum_center_mm: float,
    pair_midpoint_scale: float,
    far_social_excess_scale: float,
    ranking: list[dict],
    preserve_input_order: bool = False,
) -> dict:
    if not preserve_input_order:
        selected_keys = sorted(selected_keys, key=frame_from_key)
    keyframes = []
    for scene_frame, key in enumerate(selected_keys, start=1):
        row = table.loc[key]
        source_points = [xy_from_row(row, identity) for identity in IDENTITIES]
        source_centers = [np.mean(points, axis=0) for points in source_points]
        physical_social_distance = float(
            np.linalg.norm(source_centers[1] - source_centers[0]) * pixel_to_mm
        )
        if physical_social_distance <= 90.0:
            frame_social_scale = pixel_to_mm
        else:
            mapped_distance = 90.0 + (
                physical_social_distance - 90.0
            ) * far_social_excess_scale
            frame_social_scale = pixel_to_mm * mapped_distance / physical_social_distance
        pair_mid = np.mean(np.concatenate(source_points, axis=0), axis=0)
        target_points = [
            converted_points(
                points, pair_mid, pixel_to_mm, pair_midpoint_scale, frame_social_scale
            )
            for points in source_points
        ]
        target_points = fit_pair_with_tail_room(target_points)
        centers = [np.mean(points, axis=0) for points in target_points]
        delta = centers[0] - centers[1]
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-8:
            delta = np.asarray((1.0, 0.0))
            distance = 0.0
        direction = delta / max(float(np.linalg.norm(delta)), 1.0e-8)
        correction = max(0.0, minimum_center_mm - distance) * 0.5
        shifts = (direction * correction, -direction * correction)
        instances = []
        for identity, points, shift in zip(IDENTITIES, target_points, shifts, strict=True):
            named = {OUTPUT_NAMES[part]: points[index].tolist() for index, part in enumerate(PARTS)}
            heading = unit(points[0] - points[-1])
            center = np.mean(points, axis=0) + shift
            instances.append({
                "source_identity": identity,
                "appearance": "black_c57" if identity == "black_mouse" else "brown_agouti",
                "target_tail_base_xy_mm": (points[-1] + shift).tolist(),
                "target_center_xy_mm": center.tolist(),
                "target_heading_xy": heading.tolist(),
                "target_keypoints_xy_mm": named,
                "body_contact_shift_xy_mm": shift.tolist(),
            })
        keyframes.append({
            "scene_frame": scene_frame,
            "source_image_frame": frame_from_key(key),
            "source_image": f"dataset/LightningPose-CRIM13/crim13/{key}",
            "minimum_center_correction_mm": correction * 2.0,
            "inter_mouse_center_scale_mm_per_px": frame_social_scale,
            "instances": instances,
        })
    return {
        "schema": "mousegeoforge.lightningpose_dual_mouse_diverse_discrete_plan.v1",
        "source_dataset": "Lightning Pose CRIM13",
        "source_sessions": [session],
        "source_label_frames": [row["source_image_frame"] for row in keyframes],
        "source_label_stride_note": (
            "greedy diverse-pose downsampling; every Blender frame is one real labeled source frame; no interpolation frames"
        ),
        "selection_policy": "diverse_discrete_frames",
        "scene_frame_range": [1, len(keyframes)],
        "scene_fps": 8,
        "appearance_mapping": {"black_mouse": "black_c57", "white_mouse": "brown_agouti"},
        "homecage_mapping": {
            "pixel_to_mm": pixel_to_mm,
            "pair_midpoint_scene_scale_mm_per_px": pair_midpoint_scale,
            "local_pose_scale_mm_per_px": pixel_to_mm,
            "inter_mouse_distance_mapping": {
                "physical_until_mm": 90.0,
                "far_excess_scale": far_social_excess_scale,
            },
            "scene_center_px": [320.0, 240.0],
        },
        "minimum_body_center_mm": minimum_center_mm,
        "conditioning_keypoints": [OUTPUT_NAMES[part] for part in PARTS],
        "video_ranking": ranking,
        "keyframes": keyframes,
    }


def main() -> None:
    args = parse_args()
    table = read_labels(args.labels.resolve())
    sessions = np.asarray([session_from_key(key) for key in table.index])
    valid_rows = []
    descriptors = []
    for row_index, (_key, row) in enumerate(table.iterrows()):
        black = xy_from_row(row, IDENTITIES[0])
        white = xy_from_row(row, IDENTITIES[1])
        if not np.all(np.isfinite(black)) or not np.all(np.isfinite(white)):
            continue
        valid_rows.append(row_index)
        descriptors.append(descriptor(black, white))
    valid_rows = np.asarray(valid_rows, dtype=int)
    descriptors = robust_standardize(np.asarray(descriptors, dtype=np.float64))
    valid_sessions = sessions[valid_rows]

    base_plan = None
    if args.base_plan is not None:
        base_plan = json.loads(args.base_plan.resolve().read_text(encoding="utf-8"))
    base_count = len(base_plan["keyframes"]) if base_plan is not None else 0
    extension_count = args.count - base_count
    extending = base_plan is not None and extension_count > 0
    selection_count = extension_count if extending else args.count

    candidates = []
    session_selections = {}
    for session in sorted(set(valid_sessions)):
        indices = np.flatnonzero(valid_sessions == session)
        if len(indices) < selection_count:
            continue
        selection_local = farthest_point_indices(descriptors[indices], selection_count)
        metrics = session_metrics(descriptors[indices], selection_local)
        metrics["session"] = session
        candidates.append(metrics)
        session_selections[session] = indices[selection_local]
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    if not candidates:
        raise RuntimeError(
            f"No session contains {selection_count} fully labeled frames"
        )
    if extending:
        base_sessions = set(base_plan["source_sessions"])
        base_source_frames = {
            int(value) for value in base_plan["source_label_frames"]
        }
        compatible_candidates = []
        for row in candidates:
            candidate_session = row["session"]
            if candidate_session in base_sessions:
                continue
            candidate_indices = session_selections[candidate_session]
            candidate_frames = {
                frame_from_key(table.index[valid_rows[index]])
                for index in candidate_indices
            }
            # The current depth-height reports use numeric source frame IDs.
            # Prefer a second video whose selected IDs do not collide with the
            # base video, while source_image still preserves full provenance.
            if not (candidate_frames & base_source_frames):
                compatible_candidates.append(row)
        if args.session:
            session = args.session
            if session not in session_selections:
                raise ValueError(
                    f"Append session {session!r} has fewer than {selection_count} valid frames"
                )
        elif compatible_candidates:
            session = compatible_candidates[0]["session"]
        else:
            raise RuntimeError(
                "No second session provides enough frames without numeric source-frame collisions"
            )
    else:
        session = (
            args.session
            or (base_plan["source_sessions"][0] if base_plan is not None else None)
            or candidates[0]["session"]
        )
    if session not in session_selections:
        raise ValueError(
            f"Session {session!r} is unavailable or has fewer than {selection_count} valid frames"
        )
    excluded_source_frames = {
        int(value.strip())
        for value in args.exclude_source_frames.split(",")
        if value.strip()
    }
    if base_plan is not None:
        excluded_source_frames.update(
            int(value) for value in base_plan.get("excluded_source_frames_after_geometry_qa", [])
        )
    session_indices = np.flatnonzero(valid_sessions == session)
    if extending:
        base_keys = [
            row["source_image"].replace("\\", "/").split("/crim13/", 1)[1]
            for row in base_plan["keyframes"]
        ]
        appended_indices = session_selections[session]
        appended_keys = table.index[valid_rows[appended_indices]].tolist()
        selected_keys = base_keys + appended_keys
    elif base_plan is None:
        available_indices = np.asarray([
            index for index in session_indices
            if frame_from_key(table.index[valid_rows[index]]) not in excluded_source_frames
        ], dtype=int)
        selected_local = farthest_point_indices(descriptors[available_indices], args.count)
        selected_valid_indices = available_indices[selected_local]
    else:
        index_by_source = {
            frame_from_key(table.index[valid_rows[index]]): int(index)
            for index in session_indices
        }
        retained_sources = [
            int(value) for value in base_plan["source_label_frames"]
            if int(value) not in excluded_source_frames
        ]
        selected = [index_by_source[value] for value in retained_sources]
        available = [
            int(index) for index in session_indices
            if frame_from_key(table.index[valid_rows[index]]) not in excluded_source_frames
            and int(index) not in selected
        ]
        while len(selected) < args.count:
            candidate_values = descriptors[np.asarray(available)]
            selected_values = descriptors[np.asarray(selected)]
            minimum_distance = np.min(
                np.sum(
                    (candidate_values[:, None, :] - selected_values[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
            best_position = int(np.argmax(minimum_distance))
            selected.append(available.pop(best_position))
        selected_valid_indices = np.asarray(selected, dtype=int)
    if not extending:
        selected_table_rows = valid_rows[selected_valid_indices]
        selected_keys = table.index[selected_table_rows].tolist()
    plan = make_plan(
        table,
        session,
        selected_keys,
        args.pixel_to_mm,
        args.minimum_center_mm,
        args.pair_midpoint_scale,
        args.social_scale,
        candidates,
        preserve_input_order=(
            extending
            and not excluded_source_frames.intersection(
                int(value) for value in base_plan["source_label_frames"]
            )
        ),
    )
    plan["excluded_source_frames_after_geometry_qa"] = sorted(excluded_source_frames)
    if base_plan is not None:
        if extending:
            plan["source_sessions"] = [*base_plan["source_sessions"], session]
        previous_sources = {int(value) for value in base_plan["source_label_frames"]}
        current_sources = {int(value) for value in plan["source_label_frames"]}
        plan["incremental_replacement"] = {
            "removed_source_frames": sorted(previous_sources - current_sources),
            "added_source_frames": sorted(current_sources - previous_sources),
        }
        if extending and not (
            previous_sources & excluded_source_frames
        ):
            base_count = len(base_plan["source_label_frames"])
            plan["extension"] = {
                "base_plan": str(args.base_plan.resolve()),
                "base_frame_count": base_count,
                "appended_frame_count": len(plan["keyframes"]) - base_count,
                "preserved_scene_frame_range": [1, base_count],
                "appended_scene_frame_range": [base_count + 1, len(plan["keyframes"])],
                "appended_source_session": session,
            }
            plan["frame_order_policy"] = (
                "base plan order preserved exactly; newly selected diverse poses appended"
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "selected_session": session,
        "selected_count": len(plan["keyframes"]),
        "source_frame_range": [min(plan["source_label_frames"]), max(plan["source_label_frames"])],
        "top_video_ranking": candidates[:10],
    }, indent=2))


if __name__ == "__main__":
    main()

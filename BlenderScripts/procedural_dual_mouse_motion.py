#!/usr/bin/env python3
"""Pure NumPy motion synthesis for sparse dual-mouse top-view labels.

The model deliberately has no donor-video motion.  A neutral quadruped layout
defines only body length, stance width, and joint-side semantics.  Translation,
heading and bend come from the current LightningPose segment; paw motion comes
from a deterministic contact-aware gait clock driven by that segment's motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1.0e-9


@dataclass(frozen=True)
class MotionConfig:
    body_length_min_mm: float = 54.0
    body_length_max_mm: float = 64.0
    # Sparse top-view labels often overlap during social contact.  The 3D
    # meshes and grounded tails need more room than their projected axes do.
    minimum_body_axis_clearance_mm: float = 20.0
    stride_distance_mm: float = 14.0
    stance_fraction: float = 0.64
    turn_motion_radius_mm: float = 12.0
    active_motion_threshold_mm: float = 0.22
    fore_forward_mm: float = 43.0
    hind_forward_mm: float = 7.0
    fore_lateral_mm: float = 11.5
    hind_lateral_mm: float = 11.0
    swing_lift_mm: float = 4.0
    body_smoothing_passes: int = 1
    heading_smoothing_passes: int = 2
    curvature_smoothing_passes: int = 2


def unit2(vector: np.ndarray, fallback=(0.0, 1.0)) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < EPS:
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


def smooth_series(values: np.ndarray, passes: int = 2) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if len(output) < 3:
        return output
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    for _ in range(passes):
        pad = [(2, 2)] + [(0, 0)] * (output.ndim - 1)
        padded = np.pad(output, pad, mode="edge")
        output = sum(kernel[index] * padded[index:index + len(output)] for index in range(5))
    return output


def interpolate(frames: np.ndarray, keyframes: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(frames, keyframes, values)
    columns = [np.interp(frames, keyframes, values[:, index]) for index in range(values.shape[1])]
    return np.stack(columns, axis=1)


def segment_closest_points_2d(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closest points on two finite 2D segments."""
    u = first_end - first_start
    v = second_end - second_start
    w = first_start - second_start
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    if a < EPS and c < EPS:
        return first_start.copy(), second_start.copy()
    if a < EPS:
        first_parameter = 0.0
        second_parameter = float(np.clip(e / max(c, EPS), 0.0, 1.0))
    elif c < EPS:
        second_parameter = 0.0
        first_parameter = float(np.clip(-d / max(a, EPS), 0.0, 1.0))
    else:
        first_parameter = float(np.clip((b * e - c * d) / max(denominator, EPS), 0.0, 1.0))
        second_parameter = float(np.clip((a * e - b * d) / max(denominator, EPS), 0.0, 1.0))
        first_parameter = float(np.clip((b * second_parameter - d) / a, 0.0, 1.0))
        second_parameter = float(np.clip((b * first_parameter + e) / c, 0.0, 1.0))
    return first_start + u * first_parameter, second_start + v * second_parameter


def extract_body_motion(plan: dict, instance_index: int, frames: np.ndarray, config: MotionConfig) -> dict:
    keyframes = np.asarray([row["scene_frame"] for row in plan["keyframes"]], dtype=np.float64)
    targets = [row["instances"][instance_index] for row in plan["keyframes"]]
    roots = np.asarray([target["target_tail_base_xy_mm"] for target in targets], dtype=np.float64)
    headings = np.asarray([unit2(target["target_heading_xy"]) for target in targets])
    heading_angles = np.unwrap(np.arctan2(headings[:, 1], headings[:, 0]))
    noses = []
    necks = []
    lengths = []
    curvature = []
    for target, root, heading in zip(targets, roots, headings, strict=True):
        shift = np.asarray(target.get("body_contact_shift_xy_mm", (0.0, 0.0)), dtype=np.float64)
        keypoints = target["target_keypoints_xy_mm"]
        nose = np.asarray(keypoints["nose"], dtype=np.float64) + shift
        neck = np.asarray(keypoints["neck"], dtype=np.float64) + shift
        noses.append(nose)
        necks.append(neck)
        length = float(np.linalg.norm(nose - root))
        lengths.append(length)
        lateral = np.asarray((-heading[1], heading[0]))
        curvature.append(float(np.dot(neck - (root + heading * length * 0.68), lateral)))

    body_length = float(np.clip(np.median(lengths), config.body_length_min_mm, config.body_length_max_mm))
    dense_root = smooth_series(
        interpolate(frames, keyframes, roots), passes=config.body_smoothing_passes
    )
    dense_angle = smooth_series(
        interpolate(frames, keyframes, heading_angles), passes=config.heading_smoothing_passes
    )
    dense_curvature = np.clip(
        smooth_series(
            interpolate(frames, keyframes, np.asarray(curvature)),
            passes=config.curvature_smoothing_passes,
        ),
        -5.0,
        5.0,
    )
    dense_heading = np.stack((np.cos(dense_angle), np.sin(dense_angle)), axis=1)
    fractions = np.linspace(0.0, 1.0, 5)
    spines = []
    for root, heading, bend in zip(dense_root, dense_heading, dense_curvature, strict=True):
        lateral = np.asarray((-heading[1], heading[0]))
        points = np.stack([
            root + heading * body_length * fraction + lateral * bend * math.sin(math.pi * fraction)
            for fraction in fractions
        ])
        spines.append(points)
    return {
        "root_xy": dense_root,
        "heading_xy": dense_heading,
        "heading_angle": dense_angle,
        "spine_xy": np.asarray(spines),
        "body_length_mm": body_length,
        "source_lengths_mm": lengths,
        "source_nose_xy": np.asarray(noses),
        "source_neck_xy": np.asarray(necks),
        "curvature_mm": dense_curvature,
    }


def separate_body_axes(motions: list[dict], config: MotionConfig) -> dict:
    first, second = motions
    corrections = np.zeros((len(first["root_xy"]), 2), dtype=np.float64)
    minimum_before = float("inf")
    minimum_after = float("inf")
    corrected_frames = 0
    previous_shift = np.zeros(2, dtype=np.float64)

    def shifted_distance(frame: int, direction: np.ndarray, amount: float) -> float:
        shift = direction * amount
        closest_first, closest_second = segment_closest_points_2d(
            first["spine_xy"][frame, 0] + shift,
            first["spine_xy"][frame, -1] + shift,
            second["spine_xy"][frame, 0] - shift,
            second["spine_xy"][frame, -1] - shift,
        )
        return float(np.linalg.norm(closest_first - closest_second))

    for frame in range(len(first["root_xy"])):
        first_start, first_end = first["spine_xy"][frame, 0], first["spine_xy"][frame, -1]
        second_start, second_end = second["spine_xy"][frame, 0], second["spine_xy"][frame, -1]
        closest_first, closest_second = segment_closest_points_2d(
            first_start, first_end, second_start, second_end
        )
        delta = closest_first - closest_second
        distance = float(np.linalg.norm(delta))
        minimum_before = min(minimum_before, distance)
        if distance < config.minimum_body_axis_clearance_mm:
            # Segment intersections cannot be resolved reliably by moving only
            # along their coincident closest-point normal. Search translation
            # directions and choose the smallest temporally coherent shift that
            # produces the requested *re-evaluated* clearance.
            directions = [
                np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
                for angle in np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
            ]
            candidates = []
            for direction in directions:
                if shifted_distance(frame, direction, 70.0) < config.minimum_body_axis_clearance_mm:
                    continue
                low, high = 0.0, 70.0
                for _ in range(24):
                    middle = (low + high) * 0.5
                    if shifted_distance(frame, direction, middle) >= config.minimum_body_axis_clearance_mm:
                        high = middle
                    else:
                        low = middle
                shift = direction * high
                # Both signs are represented geometrically by [0, pi), while
                # this continuity term prevents arbitrary frame-to-frame flips.
                continuity = np.linalg.norm(shift - previous_shift)
                candidates.append((high + 1.3 * continuity, shift))
            if not candidates:
                raise RuntimeError(f"Could not separate body axes at dense frame index {frame}")
            shift = min(candidates, key=lambda item: item[0])[1]
            first["root_xy"][frame] += shift
            first["spine_xy"][frame] += shift
            second["root_xy"][frame] -= shift
            second["spine_xy"][frame] -= shift
            corrections[frame] = shift
            previous_shift = shift
            corrected_frames += 1
        elif np.linalg.norm(previous_shift) > EPS:
            # Fade an existing avoidance offset instead of snapping both mice
            # back to their raw 2D tracks on the first naturally clear frame.
            low, high = 0.0, 1.0
            direction = unit2(previous_shift)
            previous_amount = float(np.linalg.norm(previous_shift))
            if shifted_distance(frame, direction, previous_amount * 0.82) >= config.minimum_body_axis_clearance_mm:
                shift = previous_shift * 0.82
            else:
                # Zero is known to be safe here. Find the largest fraction of
                # the previous offset that remains safe on this frame.
                for _ in range(24):
                    middle = (low + high) * 0.5
                    if shifted_distance(frame, direction, previous_amount * middle) >= config.minimum_body_axis_clearance_mm:
                        low = middle
                    else:
                        high = middle
                shift = previous_shift * low
            if np.linalg.norm(shift) > 0.05:
                first["root_xy"][frame] += shift
                first["spine_xy"][frame] += shift
                second["root_xy"][frame] -= shift
                second["spine_xy"][frame] -= shift
                corrections[frame] = shift
                previous_shift = shift
                corrected_frames += 1
            else:
                previous_shift = np.zeros(2, dtype=np.float64)
        closest_first, closest_second = segment_closest_points_2d(
            first["spine_xy"][frame, 0], first["spine_xy"][frame, -1],
            second["spine_xy"][frame, 0], second["spine_xy"][frame, -1],
        )
        minimum_after = min(minimum_after, float(np.linalg.norm(closest_first - closest_second)))
    return {
        "minimum_axis_clearance_before_mm": minimum_before,
        "minimum_axis_clearance_after_mm": minimum_after,
        "corrected_frames": corrected_frames,
        "maximum_per_mouse_shift_mm": float(np.max(np.linalg.norm(corrections, axis=1))),
        "maximum_frame_to_frame_shift_delta_mm": float(
            np.max(np.linalg.norm(np.diff(corrections, axis=0), axis=1))
        ),
    }


def nominal_feet(motion: dict, config: MotionConfig) -> dict[str, np.ndarray]:
    root = motion["root_xy"]
    heading = motion["heading_xy"]
    lateral = np.stack((-heading[:, 1], heading[:, 0]), axis=1)
    return {
        "wrist_l": root + heading * config.fore_forward_mm + lateral * config.fore_lateral_mm,
        "wrist_r": root + heading * config.fore_forward_mm - lateral * config.fore_lateral_mm,
        "ankle_l": root + heading * config.hind_forward_mm + lateral * config.hind_lateral_mm,
        "ankle_r": root + heading * config.hind_forward_mm - lateral * config.hind_lateral_mm,
    }


def motion_clock(motion: dict, config: MotionConfig) -> tuple[np.ndarray, np.ndarray]:
    root = motion["root_xy"]
    angles = motion["heading_angle"]
    translation = np.linalg.norm(np.diff(root, axis=0), axis=1)
    rotation = np.abs(np.diff(angles)) * config.turn_motion_radius_mm
    increments = np.concatenate(([0.0], translation + rotation))
    local_activity = smooth_series(increments, passes=1)
    return np.cumsum(increments), local_activity


def contact_aware_foot(
    nominal: np.ndarray,
    clock: np.ndarray,
    activity: np.ndarray,
    phase_offset: float,
    config: MotionConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    phase = np.mod(clock / config.stride_distance_mm + phase_offset, 1.0)
    swinging = (phase >= config.stance_fraction) & (activity >= config.active_motion_threshold_mm)
    output = np.zeros_like(nominal)
    lift = np.zeros(len(nominal), dtype=np.float64)
    anchor = nominal[0].copy()
    frame = 0
    swing_segments = 0
    while frame < len(nominal):
        if not swinging[frame]:
            output[frame] = anchor
            frame += 1
            continue
        start = frame
        end = start
        while end < len(nominal) and swinging[end]:
            end += 1
        target_index = min(end, len(nominal) - 1)
        target = nominal[target_index].copy()
        count = max(1, end - start)
        for index in range(start, end):
            raw = (index - start + 1) / (count + (1 if end < len(nominal) else 0))
            blend = raw * raw * (3.0 - 2.0 * raw)
            output[index] = anchor * (1.0 - blend) + target * blend
            lift[index] = config.swing_lift_mm * math.sin(math.pi * raw)
        anchor = target
        if end < len(nominal):
            output[end] = anchor
        swing_segments += 1
        frame = max(end + 1, frame + 1)
    return output, lift, {
        "swing_segments": swing_segments,
        "swing_frames": int(np.count_nonzero(lift > 0.0)),
        "maximum_lift_mm": float(np.max(lift)),
    }


def enforce_foot_side_and_reach(
    feet: dict[str, np.ndarray],
    motion: dict,
) -> dict:
    root = motion["root_xy"]
    heading = motion["heading_xy"]
    lateral = np.stack((-heading[:, 1], heading[:, 0]), axis=1)
    corrections = 0
    side_failures_before = 0
    ranges = {
        "wrist_l": (28.0, 53.0, 7.5, 17.0, 1.0),
        "wrist_r": (28.0, 53.0, 7.5, 17.0, -1.0),
        "ankle_l": (-8.0, 22.0, 7.0, 16.0, 1.0),
        "ankle_r": (-8.0, 22.0, 7.0, 16.0, -1.0),
    }
    for name, positions in feet.items():
        low_forward, high_forward, low_lateral, high_lateral, side = ranges[name]
        relative = positions - root
        forward_value = np.sum(relative * heading, axis=1)
        lateral_value = np.sum(relative * lateral, axis=1)
        side_failures_before += int(np.count_nonzero(lateral_value * side <= 0.0))
        clamped_forward = np.clip(forward_value, low_forward, high_forward)
        clamped_lateral = side * np.clip(lateral_value * side, low_lateral, high_lateral)
        corrected = root + heading * clamped_forward[:, None] + lateral * clamped_lateral[:, None]
        changed = np.linalg.norm(corrected - positions, axis=1) > 1.0e-5
        corrections += int(np.count_nonzero(changed))
        positions[:] = corrected
    return {
        "corrected_foot_frames": corrections,
        "side_failures_before": side_failures_before,
        "side_failures_after": 0,
    }


def synthesize(plan: dict, config: MotionConfig | None = None) -> dict:
    config = config or MotionConfig()
    frame_start, frame_end = (int(value) for value in plan["scene_frame_range"])
    frames = np.arange(frame_start, frame_end + 1, dtype=np.float64)
    motions = [extract_body_motion(plan, index, frames, config) for index in range(2)]
    spacing = separate_body_axes(motions, config)
    foot_audits = []
    phase_offsets = {
        "wrist_l": 0.0,
        "ankle_r": 0.0,
        "wrist_r": 0.5,
        "ankle_l": 0.5,
    }
    for motion in motions:
        nominal = nominal_feet(motion, config)
        clock, activity = motion_clock(motion, config)
        feet = {}
        lift = {}
        limb_audit = {}
        for name, values in nominal.items():
            feet[name], lift[name], limb_audit[name] = contact_aware_foot(
                values, clock, activity, phase_offsets[name], config
            )
        regularization = enforce_foot_side_and_reach(feet, motion)
        motion["feet_xy"] = feet
        motion["foot_lift_mm"] = lift
        motion["motion_clock_mm"] = clock
        motion["activity_mm"] = activity
        foot_audits.append({"limbs": limb_audit, "regularization": regularization})
    return {
        "frames": frames.astype(int),
        "motions": motions,
        "spacing_audit": spacing,
        "foot_audits": foot_audits,
        "config": config,
    }

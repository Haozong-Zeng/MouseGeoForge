#!/usr/bin/env python3
"""Fuse two VGGT monocular depth passes into conservative mouse body heights.

CRIM13 contains a metal tube above the mice.  A point sample can therefore be
the tube, the other mouse, or bedding instead of the intended body surface.
This script works on local depth patches, rejects pixels outside anatomical
height limits, and only retains a modest, temporally smoothed depth residual.
The result is suitable for body/spine controls; paws remain floor constrained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset/Homecage/depth_height/vggt_crim13_dual_mouse"
IDENTITIES = ("black_mouse", "white_mouse")
CONTROL_NAMES = ("tail_base", "rear_mid", "body_mid", "neck", "nose")
NEUTRAL_MM = np.asarray((14.0, 20.0, 24.0, 21.0, 16.0), dtype=np.float64)
# Broad surface limits used for pixel rejection, not final rig limits.
SURFACE_LIMITS_MM = np.asarray(
    ((7.0, 22.0), (11.0, 30.0), (14.0, 35.0), (11.0, 38.0), (8.0, 38.0)),
    dtype=np.float64,
)
# Conservative rig limits.  They prevent sparse monocular evidence from
# creating a rearing pose that is not supported by the top-view labels.
RIG_LIMITS_MM = np.asarray(
    ((11.0, 17.5), (16.0, 25.0), (19.0, 30.0), (17.0, 31.0), (12.0, 31.0)),
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--primary", default="baseline")
    parser.add_argument("--secondary", default="real_palm_best")
    parser.add_argument("--patch-radius", type=int, default=8)
    parser.add_argument("--no-temporal-smoothing", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_candidate(input_dir: Path, name: str) -> tuple[dict, dict[int, tuple[np.ndarray, np.ndarray]]]:
    report = json.loads((input_dir / f"{name}_body_height.json").read_text(encoding="utf-8"))
    # Materialize each compressed member once.  Indexing an NpzFile member in
    # the comprehension would decompress the entire stack for every frame.
    with np.load(input_dir / f"{name}_depth_maps.npz") as arrays:
        frames = arrays["source_image_frames"].astype(int)
        depth = arrays["depth"]
        confidence = arrays["depth_conf"]
    mapped = {
        int(frame): (depth[index], confidence[index])
        for index, frame in enumerate(frames)
    }
    return report, mapped


def patch(array: np.ndarray, xy: np.ndarray, radius: int) -> np.ndarray:
    x, y = np.round(xy).astype(int)
    y0, y1 = max(0, y - radius), min(array.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(array.shape[1], x + radius + 1)
    return np.asarray(array[y0:y1, x0:x1], dtype=np.float64).reshape(-1)


def local_surface_height(
    depth: np.ndarray,
    confidence: np.ndarray,
    point: np.ndarray,
    floor_depth: float,
    metric_scale: float,
    limits: np.ndarray,
    radius: int,
) -> tuple[float | None, float, int]:
    depths = patch(depth, point, radius)
    conf = patch(confidence, point, radius)
    valid = np.isfinite(depths) & (depths > 0) & np.isfinite(conf)
    if np.count_nonzero(valid) < 8:
        return None, 0.0, 0
    heights = (floor_depth - depths[valid]) * metric_scale
    confidence_values = conf[valid]
    threshold = float(np.quantile(confidence_values, 0.30))
    anatomical = (
        (heights >= limits[0])
        & (heights <= limits[1])
        & (confidence_values >= threshold)
    )
    accepted = heights[anatomical]
    if len(accepted) < 6:
        return None, float(len(accepted) / max(len(heights), 1)), int(len(accepted))
    # A trimmed median is robust to both the nearer overhead tube and farther
    # bedding pixels at the edge of the approximate body mask.
    low, high = np.quantile(accepted, (0.15, 0.85))
    accepted = accepted[(accepted >= low) & (accepted <= high)]
    return float(np.median(accepted)), float(len(accepted) / len(heights)), int(len(accepted))


def candidate_estimates(
    report: dict,
    arrays: dict[int, tuple[np.ndarray, np.ndarray]],
    radius: int,
) -> dict[str, np.ndarray]:
    result = {identity: [] for identity in IDENTITIES}
    for frame_record in report["keyframes"]:
        source_frame = int(frame_record["source_image_frame"])
        depth, confidence = arrays[source_frame]
        for identity in IDENTITIES:
            instance = frame_record["instances"][identity]
            points = np.asarray(instance["centerline_xy_model"], dtype=np.float64)
            values = []
            for index, point in enumerate(points):
                estimate, fraction, count = local_surface_height(
                    depth,
                    confidence,
                    point,
                    float(instance["floor_depth"]),
                    float(instance["depth_metric_scale_mm_per_model_unit"]),
                    SURFACE_LIMITS_MM[index],
                    radius,
                )
                values.append({"height_mm": estimate, "accepted_fraction": fraction, "accepted_pixels": count})
            result[identity].append(values)
    return result


def smooth_time(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    padded = np.pad(values, ((1, 1), (0, 0)), mode="edge")
    return 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]


def fuse_identity(
    primary: list,
    secondary: list,
    *,
    temporal_smoothing: bool = True,
) -> tuple[np.ndarray, list]:
    fused = np.tile(NEUTRAL_MM, (len(primary), 1))
    evidence_rows = []
    for frame_index, (first_row, second_row) in enumerate(zip(primary, secondary, strict=True)):
        evidence = []
        for control_index, (first, second) in enumerate(zip(first_row, second_row, strict=True)):
            first_value = first["height_mm"]
            second_value = second["height_mm"]
            agreement = None
            if first_value is not None and second_value is not None:
                agreement = abs(first_value - second_value)
                if agreement <= 7.0:
                    observed = 0.72 * first_value + 0.28 * second_value
                    weight = min(0.48, 0.20 + 0.65 * min(first["accepted_fraction"], second["accepted_fraction"]))
                    status = "two_model_agreement"
                else:
                    observed = first_value
                    weight = min(0.25, 0.10 + 0.35 * first["accepted_fraction"])
                    status = "primary_only_models_disagree"
            elif first_value is not None:
                observed = first_value
                weight = min(0.22, 0.08 + 0.35 * first["accepted_fraction"])
                status = "primary_only"
            elif second_value is not None:
                observed = second_value
                weight = min(0.12, 0.05 + 0.18 * second["accepted_fraction"])
                status = "secondary_only"
            else:
                observed = float(NEUTRAL_MM[control_index])
                weight = 0.0
                status = "neutral_fallback"
            fused[frame_index, control_index] = (
                NEUTRAL_MM[control_index] * (1.0 - weight) + observed * weight
            )
            evidence.append({
                "control": CONTROL_NAMES[control_index],
                "primary": first,
                "secondary": second,
                "model_agreement_mm": agreement,
                "status": status,
                "depth_weight": weight,
                "pre_temporal_height_mm": float(fused[frame_index, control_index]),
            })
        evidence_rows.append(evidence)

    if temporal_smoothing:
        fused = smooth_time(fused)
    fused = np.clip(fused, RIG_LIMITS_MM[:, 0], RIG_LIMITS_MM[:, 1])
    # Avoid sharp longitudinal kinks while preserving head/neck elevation.
    for frame_index in range(len(fused)):
        for index in range(1, len(CONTROL_NAMES)):
            fused[frame_index, index] = np.clip(
                fused[frame_index, index],
                fused[frame_index, index - 1] - 8.0,
                fused[frame_index, index - 1] + 11.0,
            )
    return fused, evidence_rows


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    primary_report, primary_arrays = load_candidate(input_dir, args.primary)
    secondary_report, secondary_arrays = load_candidate(input_dir, args.secondary)
    primary_frames = [int(row["source_image_frame"]) for row in primary_report["keyframes"]]
    secondary_frames = [int(row["source_image_frame"]) for row in secondary_report["keyframes"]]
    if primary_frames != secondary_frames:
        raise RuntimeError("Candidate reports do not contain the same source frames")

    primary = candidate_estimates(primary_report, primary_arrays, args.patch_radius)
    secondary = candidate_estimates(secondary_report, secondary_arrays, args.patch_radius)
    fused_by_identity = {}
    evidence_by_identity = {}
    for identity in IDENTITIES:
        fused, evidence = fuse_identity(
            primary[identity],
            secondary[identity],
            temporal_smoothing=not args.no_temporal_smoothing,
        )
        fused_by_identity[identity] = fused
        evidence_by_identity[identity] = evidence

    keyframes = []
    for frame_index, primary_record in enumerate(primary_report["keyframes"]):
        instances = {}
        for identity in IDENTITIES:
            instances[identity] = {
                "height_mm": fused_by_identity[identity][frame_index].tolist(),
                "evidence": evidence_by_identity[identity][frame_index],
            }
        keyframes.append({
            "scene_frame": int(primary_record["scene_frame"]),
            "source_image_frame": int(primary_record["source_image_frame"]),
            "source_image": primary_record["source_image"],
            "instances": instances,
        })

    output = args.output or input_dir / "fused_body_height.json"
    report = {
        "schema": "mousegeoforge.crim13_vggt_monocular_body_height_fusion.v1",
        "primary_candidate": args.primary,
        "secondary_candidate": args.secondary,
        "spine_controls": list(CONTROL_NAMES),
        "neutral_height_mm": NEUTRAL_MM.tolist(),
        "surface_rejection_limits_mm": SURFACE_LIMITS_MM.tolist(),
        "rig_limits_mm": RIG_LIMITS_MM.tolist(),
        "policy": (
            "local anatomical pixel rejection; baseline-primary two-model agreement; "
            "strong neutral prior; temporal smoothing; body controls only"
        ),
        "ground_contact_policy": "paws and floor remain governed by procedural contact IK",
        "temporal_smoothing": not args.no_temporal_smoothing,
        "keyframes": keyframes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        identity: {
            "minimum_mm": np.min(values, axis=0).round(3).tolist(),
            "maximum_mm": np.max(values, axis=0).round(3).tolist(),
            "mean_mm": np.mean(values, axis=0).round(3).tolist(),
        }
        for identity, values in fused_by_identity.items()
    }
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

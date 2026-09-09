#!/usr/bin/env python3
"""Align Anipose 3D keypoints to the Blender rig and derive hand controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_vggt_anipose_world_alignment import (
    calibration_extrinsics,
    restore_calibration_world,
)
from prepare_vggt_blender_hand_alignment import (
    build_hand_descriptors,
    find_camera_mapping,
    json_ready,
    keypoint_names,
    orientation_guided_similarity,
    static_landmark_summary,
    transform_keypoints,
    unique_blender_cameras,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-inspection", type=Path, required=True)
    parser.add_argument("--anipose-pose", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-cameras", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rig = json.loads(args.rig_inspection.read_text(encoding="utf-8"))
    target_cameras = unique_blender_cameras(rig)
    source_extrinsics = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in calibration_extrinsics(args.calibration).items()
    }
    mapping, mapping_diagnostics = find_camera_mapping(
        source_extrinsics, target_cameras
    )
    scale, rotation, translation, alignment_diagnostics = (
        orientation_guided_similarity(
            mapping,
            source_extrinsics,
            target_cameras,
        )
    )

    anipose_adjusted = pd.read_csv(args.anipose_pose)
    calibration_world, anipose_coordinate_frame = restore_calibration_world(
        anipose_adjusted
    )
    names = keypoint_names(list(calibration_world.columns))
    aligned = transform_keypoints(
        calibration_world,
        names,
        scale,
        rotation,
        translation,
    )
    descriptors = build_hand_descriptors(
        aligned,
        score_threshold=args.score_threshold,
        min_cameras=args.min_cameras,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "aligned_keypoints_blender.csv"
    descriptor_path = args.output_dir / "hand_descriptors.csv"
    report_path = args.output_dir / "alignment.json"
    aligned.to_csv(aligned_path, index=False)
    descriptors.to_csv(descriptor_path, index=False)

    report = {
        "source_anipose_pose": str(args.anipose_pose.resolve()),
        "source_calibration": str(args.calibration.resolve()),
        "rig_inspection": str(args.rig_inspection.resolve()),
        "coordinate_formula": (
            "X_calibration = inv(M) @ (X_anipose_adjusted + center); "
            "X_blender = scale * rotation @ X_calibration + translation"
        ),
        "anipose_coordinate_frame": anipose_coordinate_frame,
        "camera_mapping": mapping,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "mapping_diagnostics": mapping_diagnostics,
        "alignment_diagnostics": alignment_diagnostics,
        "keypoint_count": len(names),
        "frame_count": len(aligned),
        "valid_descriptor_rows": int(
            descriptors.get("valid", pd.Series(dtype=bool)).sum()
        ),
        "static_landmarks_blender": static_landmark_summary(
            aligned,
            ["spout_tip_A", "point1", "point2", "point3"],
        ),
        "outputs": {
            "aligned_keypoints": str(aligned_path.resolve()),
            "hand_descriptors": str(descriptor_path.resolve()),
        },
    }
    report_path.write_text(
        json.dumps(json_ready(report), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "camera_mapping": mapping,
                "center_rmse": alignment_diagnostics["center_rmse"],
                "rotation_mean_deg": alignment_diagnostics["rotation_mean_deg"],
                "frames": len(aligned),
                "valid_descriptor_rows": report["valid_descriptor_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare old and normalized-finger Blender animation audit samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


FINGERS = ("Index", "Middle", "Ring", "Pinky")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quaternion_angle_deg(first: list[float], second: list[float]) -> float:
    dot = abs(float(np.dot(first, second)))
    dot = min(max(dot, -1.0), 1.0)
    return math.degrees(2.0 * math.acos(dot))


def main() -> None:
    args = parse_args()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    if old["frame_range"] != new["frame_range"]:
        raise ValueError("Animation frame ranges differ")
    old_frames = {row["frame"]: row for row in old["frames"]}
    new_frames = {row["frame"]: row for row in new["frames"]}
    if set(old_frames) != set(new_frames):
        raise ValueError("Animation sampled frames differ")

    report: dict[str, object] = {
        "frame_range": old["frame_range"],
        "controller_regression": {},
        "finger_motion": {},
    }
    for side in ("L", "R"):
        locations = []
        rotations = []
        wrists = []
        side_motion = {}
        for frame in sorted(old_frames):
            old_side = old_frames[frame]["sides"][side]
            new_side = new_frames[frame]["sides"][side]
            locations.append(
                np.linalg.norm(
                    np.asarray(old_side["controller_location"])
                    - np.asarray(new_side["controller_location"])
                )
            )
            rotations.append(
                quaternion_angle_deg(
                    old_side["controller_quaternion"],
                    new_side["controller_quaternion"],
                )
            )
            wrists.append(
                np.linalg.norm(
                    np.asarray(old_side["wrist_tail"])
                    - np.asarray(new_side["wrist_tail"])
                )
            )
        report["controller_regression"][side] = {
            "maximum_controller_location_difference": float(max(locations)),
            "maximum_controller_rotation_difference_deg": float(max(rotations)),
            "maximum_wrist_tail_difference": float(max(wrists)),
        }

        for finger in FINGERS:
            curls = np.asarray(
                [
                    new_frames[frame]["sides"][side]["curl_properties"][finger]
                    for frame in sorted(new_frames)
                ],
                dtype=np.float64,
            )
            old_bend = np.asarray(
                [
                    old_frames[frame]["sides"][side]["fingers"][finger][
                        "bend_1_2_deg"
                    ]
                    + old_frames[frame]["sides"][side]["fingers"][finger][
                        "bend_2_3_deg"
                    ]
                    for frame in sorted(old_frames)
                ],
                dtype=np.float64,
            )
            new_bend = np.asarray(
                [
                    new_frames[frame]["sides"][side]["fingers"][finger][
                        "bend_1_2_deg"
                    ]
                    + new_frames[frame]["sides"][side]["fingers"][finger][
                        "bend_2_3_deg"
                    ]
                    for frame in sorted(new_frames)
                ],
                dtype=np.float64,
            )
            tip_changes = np.asarray(
                [
                    np.linalg.norm(
                        np.asarray(
                            old_frames[frame]["sides"][side]["fingers"][finger][
                                "tip"
                            ]
                        )
                        - np.asarray(
                            new_frames[frame]["sides"][side]["fingers"][finger][
                                "tip"
                            ]
                        )
                    )
                    for frame in sorted(new_frames)
                ]
            )
            side_motion[finger] = {
                "curl_min_deg": float(curls.min()),
                "curl_max_deg": float(curls.max()),
                "curl_max_frame_step_deg": float(
                    np.max(np.abs(np.diff(curls)))
                ),
                "old_bend_range_deg": float(old_bend.max() - old_bend.min()),
                "new_bend_range_deg": float(new_bend.max() - new_bend.min()),
                "new_bend_max_frame_step_deg": float(
                    np.max(np.abs(np.diff(new_bend)))
                ),
                "tip_change_median": float(np.median(tip_changes)),
                "tip_change_p95": float(np.quantile(tip_changes, 0.95)),
            }
        report["finger_motion"][side] = side_motion

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

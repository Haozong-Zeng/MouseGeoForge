#!/usr/bin/env python3
"""Regression checks for the 200-frame discrete CRIM13 pose plan."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from procedural_dual_mouse_motion import MotionConfig, synthesize


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "dataset/Homecage/LightningPose-CRIM13-Diverse200.plan.json").read_text()
    )
    assert plan["selection_policy"] == "diverse_discrete_frames"
    assert len(plan["keyframes"]) == 200
    assert len(set(plan["source_label_frames"])) == 200
    config = MotionConfig(
        body_smoothing_passes=0,
        heading_smoothing_passes=0,
        curvature_smoothing_passes=0,
    )
    result = synthesize(plan, config)
    assert list(result["frames"][[0, -1]]) == [1, 200]
    assert result["spacing_audit"]["minimum_axis_clearance_after_mm"] >= 19.999
    summaries = []
    for motion in result["motions"]:
        heading = motion["heading_xy"]
        lateral = np.stack((-heading[:, 1], heading[:, 0]), axis=1)
        for name, side in (("wrist_l", 1), ("ankle_l", 1), ("wrist_r", -1), ("ankle_r", -1)):
            signed = np.sum((motion["feet_xy"][name] - motion["root_xy"]) * lateral, axis=1) * side
            assert float(np.min(signed)) >= 6.999, (name, float(np.min(signed)))
        angle_bins = len(np.unique(np.round(np.arctan2(heading[:, 1], heading[:, 0]), 1)))
        assert angle_bins >= 20, angle_bins
        summaries.append({
            "body_length_mm": motion["body_length_mm"],
            "heading_angle_bins_0p1rad": angle_bins,
            "curvature_range_mm": [
                float(np.min(motion["curvature_mm"])),
                float(np.max(motion["curvature_mm"])),
            ],
            "motion_clock_mm": float(motion["motion_clock_mm"][-1]),
        })
    print(json.dumps({
        "frames": 200,
        "unique_source_frames": 200,
        "spacing": result["spacing_audit"],
        "instances": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()

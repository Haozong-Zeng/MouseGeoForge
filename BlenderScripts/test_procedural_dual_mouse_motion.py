#!/usr/bin/env python3
"""Regression checks for donor-independent sparse-label motion synthesis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from procedural_dual_mouse_motion import synthesize


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "dataset/Homecage/LightningPose-TwoColor-DualMouse-Pilot.plan.json").read_text()
    )
    result = synthesize(plan)
    assert list(result["frames"][[0, -1]]) == [1, 57]
    assert result["spacing_audit"]["minimum_axis_clearance_after_mm"] >= 19.999
    for motion in result["motions"]:
        heading = motion["heading_xy"]
        lateral = np.stack((-heading[:, 1], heading[:, 0]), axis=1)
        root_xy = motion["root_xy"]
        for name, side in (("wrist_l", 1), ("ankle_l", 1), ("wrist_r", -1), ("ankle_r", -1)):
            signed = np.sum((motion["feet_xy"][name] - root_xy) * lateral, axis=1) * side
            assert float(np.min(signed)) >= 6.999, (name, float(np.min(signed)))
            assert float(np.min(motion["foot_lift_mm"][name])) >= 0.0
        assert 54.0 <= motion["body_length_mm"] <= 64.0
    print(json.dumps({
        "frames": [1, 57],
        "spacing": result["spacing_audit"],
        "body_lengths_mm": [motion["body_length_mm"] for motion in result["motions"]],
        "foot_audits": result["foot_audits"],
    }, indent=2))


if __name__ == "__main__":
    main()

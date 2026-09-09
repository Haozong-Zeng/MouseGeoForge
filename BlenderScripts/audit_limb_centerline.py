#!/usr/bin/env python3
"""Audit left/right limb placement in each mouse's top-view body frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


SYSTEMS = {
    "Black": "LP_Black_Armature",
    "Brown": "LP_Brown_Armature",
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def point(armature: bpy.types.Object, bone: str) -> np.ndarray:
    return np.asarray(armature.matrix_world @ armature.pose.bones[bone].head, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(args_after_dash())
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    report = {"blend": bpy.data.filepath, "frame_range": [frames[0], frames[-1]], "systems": {}}
    for label, armature_name in SYSTEMS.items():
        armature = bpy.data.objects[armature_name]
        rows = []
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            root = point(armature, "Root_M")
            head = point(armature, "HeadEnd_M")
            hip_l = point(armature, "Hip_L")
            hip_r = point(armature, "Hip_R")
            lateral = hip_l - hip_r
            lateral[2] = 0.0
            lateral /= max(np.linalg.norm(lateral), 1.0e-8)
            forward = head - root
            forward[2] = 0.0
            forward /= max(np.linalg.norm(forward), 1.0e-8)
            # Remove any longitudinal component from the label-derived lateral axis.
            lateral -= forward * float(np.dot(lateral, forward))
            lateral /= max(np.linalg.norm(lateral), 1.0e-8)

            values = {}
            for group, left_bone, right_bone in (
                ("fore_wrist", "Wrist_L", "Wrist_R"),
                ("fore_finger", "Fingers1_L", "Fingers1_R"),
                ("hind_ankle", "Ankle_L", "Ankle_R"),
                ("hind_toe", "Toes1_L", "Toes1_R"),
            ):
                left = float(np.dot(point(armature, left_bone) - root, lateral))
                right = float(np.dot(point(armature, right_bone) - root, lateral))
                values[group] = {
                    "left_lateral_mm": left,
                    "right_lateral_mm": right,
                    "pair_midpoint_lateral_mm": 0.5 * (left + right),
                    "separation_mm": left - right,
                    "crossed_or_same_side": left <= 0.0 or right >= 0.0,
                }
            rows.append({"frame": frame, **values})
        summary = {}
        for group in ("fore_wrist", "fore_finger", "hind_ankle", "hind_toe"):
            group_rows = [row[group] for row in rows]
            worst_midpoint = max(group_rows, key=lambda row: abs(row["pair_midpoint_lateral_mm"]))
            summary[group] = {
                "crossed_or_same_side_frames": sum(row["crossed_or_same_side"] for row in group_rows),
                "maximum_pair_midpoint_offset_mm": max(abs(row["pair_midpoint_lateral_mm"]) for row in group_rows),
                "minimum_left_lateral_mm": min(row["left_lateral_mm"] for row in group_rows),
                "maximum_right_lateral_mm": max(row["right_lateral_mm"] for row in group_rows),
                "minimum_separation_mm": min(row["separation_mm"] for row in group_rows),
                "worst_midpoint": worst_midpoint,
            }
        report["systems"][label] = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({label: data["summary"] for label, data in report["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()

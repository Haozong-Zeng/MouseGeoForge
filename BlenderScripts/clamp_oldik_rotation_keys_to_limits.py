"""Clamp raw pose rotation channels to active LIMIT_ROTATION windows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", required=True)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=240)
    parser.add_argument("--bone-filter", action="append", default=[])
    return parser.parse_args(args_after_dash())


def match_bone(name: str, filters: list[str]) -> bool:
    return not filters or any(token.lower() in name.lower() for token in filters)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def limit_constraints(pb):
    return [c for c in pb.constraints if c.type == "LIMIT_ROTATION" and not c.mute and c.influence > 0.0]


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None:
        raise RuntimeError(f"Armature {args.armature!r} not found.")
    action = bpy.data.actions.get(args.action)
    if action is None:
        raise RuntimeError(f"Action {args.action!r} not found.")
    armature.animation_data_create()
    armature.animation_data.action = action

    rows = []
    for frame in range(args.frame_start, args.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for pb in armature.pose.bones:
            if not match_bone(pb.name, args.bone_filter):
                continue
            constraints = limit_constraints(pb)
            if not constraints:
                continue
            pb.rotation_mode = "XYZ"
            before = [float(v) for v in pb.rotation_euler]
            after = before[:]
            changed_axes = {}
            for constraint in constraints:
                for idx, axis in enumerate(("x", "y", "z")):
                    if not getattr(constraint, f"use_limit_{axis}", True):
                        continue
                    low = float(getattr(constraint, f"min_{axis}"))
                    high = float(getattr(constraint, f"max_{axis}"))
                    clamped = clamp(after[idx], low, high)
                    if abs(clamped - after[idx]) > 1e-8:
                        changed_axes[axis] = {
                            "before_deg": round(math.degrees(after[idx]), 4),
                            "after_deg": round(math.degrees(clamped), 4),
                            "min_deg": round(math.degrees(low), 4),
                            "max_deg": round(math.degrees(high), 4),
                        }
                    after[idx] = clamped
            if changed_axes:
                pb.rotation_euler = after
                pb.keyframe_insert(data_path="rotation_euler", frame=frame)
                rows.append({"frame": frame, "bone": pb.name, "axes": changed_axes})

    bpy.context.scene.frame_set(args.frame_start)
    bpy.context.view_layer.update()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report_path = args.report or args.output.with_suffix(".rotation-clamp-report.json")
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "action": action.name,
        "changed_count": len(rows),
        "changes": rows[:2000],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(report_path.resolve()), "changed_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

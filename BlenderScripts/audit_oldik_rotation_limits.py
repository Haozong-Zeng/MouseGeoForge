"""Audit pose-bone Euler channels against LIMIT_ROTATION constraints."""

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
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default=None)
    parser.add_argument("--frame", type=int, action="append", default=[])
    parser.add_argument("--bone-filter", action="append", default=[])
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    return parser.parse_args(args_after_dash())


def match_bone(name: str, filters: list[str]) -> bool:
    return not filters or any(token.lower() in name.lower() for token in filters)


def euler_deg(pb) -> tuple[float, float, float]:
    if pb.rotation_mode == "QUATERNION":
        e = pb.rotation_quaternion.to_euler("XYZ")
    elif pb.rotation_mode == "AXIS_ANGLE":
        e = pb.rotation_axis_angle.to_quaternion().to_euler("XYZ")
    else:
        e = pb.rotation_euler
    return tuple(math.degrees(float(v)) for v in e)


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None:
        raise RuntimeError(f"Armature {args.armature!r} not found.")
    if args.action:
        action = bpy.data.actions.get(args.action)
        if action is None:
            raise RuntimeError(f"Action {args.action!r} not found.")
        armature.animation_data_create()
        armature.animation_data.action = action

    frames = args.frame or [int(bpy.context.scene.frame_current)]
    tol = args.tolerance_deg
    report = {
        "input": str(args.input.resolve()),
        "action": args.action,
        "frames": {},
        "violation_count": 0,
    }
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        rows = []
        for pb in armature.pose.bones:
            if not match_bone(pb.name, args.bone_filter):
                continue
            limits = [c for c in pb.constraints if c.type == "LIMIT_ROTATION" and not c.mute and c.influence > 0]
            if not limits:
                continue
            rot = euler_deg(pb)
            for c in limits:
                axes = {}
                violated = False
                for idx, axis in enumerate(("x", "y", "z")):
                    low = math.degrees(float(getattr(c, f"min_{axis}")))
                    high = math.degrees(float(getattr(c, f"max_{axis}")))
                    value = rot[idx]
                    outside = value < low - tol or value > high + tol
                    violated = violated or outside
                    axes[axis] = {
                        "value_deg": round(value, 4),
                        "min_deg": round(low, 4),
                        "max_deg": round(high, 4),
                        "outside": outside,
                    }
                if violated:
                    report["violation_count"] += 1
                rows.append(
                    {
                        "bone": pb.name,
                        "constraint": c.name,
                        "rotation_mode": pb.rotation_mode,
                        "violated": violated,
                        "axes": axes,
                    }
                )
        report["frames"][str(frame)] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "violation_count": report["violation_count"]}, indent=2))


if __name__ == "__main__":
    main()

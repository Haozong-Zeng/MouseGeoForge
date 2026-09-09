#!/usr/bin/env python3
"""Export or apply an exact integer-frame prefix of dual-mouse control animation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


PREFIXES = ("LP_Black_", "LP_Brown_")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "apply"), required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-frame", type=int)
    return parser.parse_args(args_after_dash())


def animated_objects() -> list[bpy.types.Object]:
    return sorted(
        (
            obj for obj in bpy.data.objects
            if obj.name.startswith(PREFIXES)
            and obj.animation_data is not None
            and obj.animation_data.action is not None
        ),
        key=lambda obj: obj.name,
    )


def animated_curves() -> list[bpy.types.Object]:
    return sorted(
        (
            obj for obj in bpy.data.objects
            if obj.name.startswith(PREFIXES)
            and obj.type == "CURVE"
            and obj.data.animation_data is not None
            and obj.data.animation_data.action is not None
        ),
        key=lambda obj: obj.name,
    )


def object_state(obj: bpy.types.Object) -> dict:
    return {
        "location": list(obj.location),
        "rotation_mode": obj.rotation_mode,
        "rotation_euler": list(obj.rotation_euler),
        "rotation_quaternion": list(obj.rotation_quaternion),
        "rotation_axis_angle": list(obj.rotation_axis_angle),
        "scale": list(obj.scale),
    }


def curve_state(obj: bpy.types.Object) -> list[list[list[float]]]:
    return [
        [list(point.co) for point in spline.points]
        for spline in obj.data.splines
    ]


def export_state(args: argparse.Namespace) -> None:
    objects = animated_objects()
    curves = animated_curves()
    frames = []
    for frame in range(args.frame_start, args.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frames.append({
            "frame": frame,
            "objects": {obj.name: object_state(obj) for obj in objects},
            "curves": {obj.name: curve_state(obj) for obj in curves},
        })
    payload = {
        "schema": "mousegeoforge.dual_mouse_animation_prefix.v1",
        "source_blend": bpy.data.filepath,
        "frame_range": [args.frame_start, args.frame_end],
        "object_names": [obj.name for obj in objects],
        "curve_names": [obj.name for obj in curves],
        "frames": frames,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({
        "state": str(args.state),
        "frame_range": payload["frame_range"],
        "objects": len(objects),
        "curves": len(curves),
    }, indent=2))


def key_object(obj: bpy.types.Object, state: dict, frame: int) -> None:
    obj.location = state["location"]
    obj.rotation_mode = state["rotation_mode"]
    obj.rotation_euler = state["rotation_euler"]
    obj.rotation_quaternion = state["rotation_quaternion"]
    obj.rotation_axis_angle = state["rotation_axis_angle"]
    obj.scale = state["scale"]
    obj.keyframe_insert(data_path="location", frame=frame)
    if obj.rotation_mode == "QUATERNION":
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    elif obj.rotation_mode == "AXIS_ANGLE":
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


def key_curve(obj: bpy.types.Object, state: list, frame: int) -> None:
    if len(obj.data.splines) != len(state):
        raise RuntimeError(f"Spline count mismatch for {obj.name}")
    for spline, source_points in zip(obj.data.splines, state, strict=True):
        if len(spline.points) != len(source_points):
            raise RuntimeError(f"Point count mismatch for {obj.name}")
        for point, co in zip(spline.points, source_points, strict=True):
            point.co = co
            point.keyframe_insert(data_path="co", frame=frame)


def apply_state(args: argparse.Namespace) -> None:
    payload = json.loads(args.state.read_text(encoding="utf-8"))
    expected = [args.frame_start, args.frame_end]
    if payload["frame_range"] != expected:
        raise ValueError(f"State frame range {payload['frame_range']} != {expected}")
    missing = [
        name for name in payload["object_names"] + payload["curve_names"]
        if bpy.data.objects.get(name) is None
    ]
    if missing:
        raise KeyError(f"Target blend is missing animation controls: {missing}")
    for row in payload["frames"]:
        frame = int(row["frame"])
        for name, state in row["objects"].items():
            key_object(bpy.data.objects[name], state, frame)
        for name, state in row["curves"].items():
            key_curve(bpy.data.objects[name], state, frame)
    if args.review_frame is not None:
        bpy.context.scene.frame_set(args.review_frame)
    output = args.output or Path(bpy.data.filepath)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(json.dumps({
        "output": str(output),
        "applied_frame_range": expected,
        "objects": len(payload["object_names"]),
        "curves": len(payload["curve_names"]),
        "review_frame": bpy.context.scene.frame_current,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "export":
        export_state(args)
    else:
        apply_state(args)


if __name__ == "__main__":
    main()

"""Smooth baked helper bone rotations by limiting per-frame quaternion steps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


DEFAULT_BONES = [
    "thigh_stretch_dupli_001.r",
    "thigh_twist_dupli_001.r",
]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--new-action", default="V3_mesh_refined_v38_helper_detwist_smoothed")
    parser.add_argument("--armature", default="\u8001\u9f20_rig")
    parser.add_argument("--bone", action="append", default=[])
    parser.add_argument("--max-step-deg", type=float, default=45.0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    return parser.parse_args(args_after_dash())


def frame_range(action, scene, start: int | None, end: int | None, step: int) -> list[int]:
    action_start, action_end = action.frame_range
    first = int(math.floor(action_start)) if start is None else int(start)
    last = int(math.ceil(action_end)) if end is None else int(end)
    if last < first:
        first, last = int(scene.frame_start), int(scene.frame_end)
    return list(range(first, last + 1, max(int(step), 1)))


def set_frame(frame: int) -> None:
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()


def signed_continuous(quats):
    out = []
    prev = None
    for quat in quats:
        q = quat.copy().normalized()
        if prev is not None and prev.dot(q) < 0.0:
            q.negate()
        out.append(q)
        prev = q
    return out


def limit_forward(quats, max_step_rad: float):
    if not quats:
        return []
    out = [quats[0].copy().normalized()]
    for quat in quats[1:]:
        q = quat.copy().normalized()
        prev = out[-1]
        if prev.dot(q) < 0.0:
            q.negate()
        angle = prev.rotation_difference(q).angle
        if angle > max_step_rad and angle > 1.0e-8:
            q = prev.slerp(q, max_step_rad / angle).normalized()
        out.append(q)
    return out


def limit_bidirectional(quats, max_step_rad: float, passes: int):
    smoothed = signed_continuous(quats)
    for _ in range(max(1, passes)):
        smoothed = limit_forward(smoothed, max_step_rad)
        smoothed = list(reversed(limit_forward(list(reversed(smoothed)), max_step_rad)))
        smoothed = signed_continuous(smoothed)
    return smoothed


def max_delta_deg(quats) -> float:
    if len(quats) < 2:
        return 0.0
    continuous = signed_continuous(quats)
    return max(
        math.degrees(continuous[index - 1].rotation_difference(continuous[index]).angle)
        for index in range(1, len(continuous))
    )


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    rig = bpy.data.objects.get(args.armature)
    action = bpy.data.actions.get(args.action)
    if rig is None or action is None:
        raise RuntimeError("Missing armature or action.")

    rig.animation_data_create()
    source_action = action
    smoothed_action = source_action.copy()
    smoothed_action.name = args.new_action
    rig.animation_data.action = source_action

    frames = frame_range(source_action, bpy.context.scene, args.frame_start, args.frame_end, args.frame_step)
    bone_names = args.bone or DEFAULT_BONES
    max_step_rad = math.radians(args.max_step_deg)
    sampled = {}
    stats = {}
    for name in bone_names:
        pb = rig.pose.bones.get(name)
        if pb is None:
            continue
        quats = []
        for frame in frames:
            set_frame(frame)
            quats.append(pb.matrix_basis.to_quaternion().normalized())
        smoothed = limit_bidirectional(quats, max_step_rad, args.passes)
        sampled[name] = smoothed
        stats[name] = {
            "raw_max_step_deg": max_delta_deg(quats),
            "smoothed_max_step_deg": max_delta_deg(smoothed),
        }

    rig.animation_data.action = smoothed_action
    for index, frame in enumerate(frames, start=1):
        set_frame(frame)
        for name, quats in sampled.items():
            pb = rig.pose.bones[name]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = quats[index - 1]
            pb.keyframe_insert("rotation_quaternion", frame=frame)
        if index == 1 or index == len(frames) or index % 50 == 0:
            print(f"[smooth] {index}/{len(frames)} frame={frame}")

    rig.animation_data.action = smoothed_action
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "source_action": source_action.name,
        "new_action": smoothed_action.name,
        "bones": sorted(sampled),
        "max_step_deg": args.max_step_deg,
        "passes": args.passes,
        "stats": stats,
    }
    report_path = args.output.with_suffix(".smooth-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(report_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()

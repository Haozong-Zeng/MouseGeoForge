"""Apply separate rest-length Y scales to old-IK forelimb segments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--scapula-scale", type=float, default=1.0)
    parser.add_argument("--wrist-scale", type=float, default=1.0)
    return parser.parse_args(args_after_dash())


def descendants_by_parent(rig) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {bone.name: [] for bone in rig.data.bones}
    for bone in rig.data.bones:
        if bone.parent:
            children.setdefault(bone.parent.name, []).append(bone.name)
    return children


def collect_descendants(root: str, children: dict[str, list[str]]) -> list[str]:
    out = []
    stack = list(children.get(root, []))
    while stack:
        name = stack.pop(0)
        out.append(name)
        stack.extend(children.get(name, []))
    return out


def constraint_summary(rig) -> dict:
    types: dict[str, int] = {}
    total = 0
    muted = 0
    for pose_bone in rig.pose.bones:
        for constraint in pose_bone.constraints:
            total += 1
            types[constraint.type] = types.get(constraint.type, 0) + 1
            muted += int(bool(getattr(constraint, "mute", False)))
    return {"total": total, "muted": muted, "types": dict(sorted(types.items()))}


def apply_segment_rest_y_scales(rig, segment_scales: dict[str, float]) -> dict:
    children = descendants_by_parent(rig)
    original = {}
    connect = {}
    parent = {}
    for bone in rig.data.bones:
        original[bone.name] = {
            "head": bone.head_local.copy(),
            "tail": bone.tail_local.copy(),
            "length": float(bone.length),
        }
        connect[bone.name] = bool(bone.use_connect)
        parent[bone.name] = bone.parent.name if bone.parent else None

    new_positions = {
        name: {"head": row["head"].copy(), "tail": row["tail"].copy()}
        for name, row in original.items()
    }
    changed = {}

    for side in ("L", "R"):
        accumulated = Vector((0.0, 0.0, 0.0))
        chain = [f"Scapula_{side}", f"Wrist_{side}"]
        for index, bone_name in enumerate(chain):
            if bone_name not in original:
                continue
            scale_y = float(segment_scales.get(bone_name, 1.0))
            if math.isclose(scale_y, 1.0, abs_tol=1.0e-8):
                continue

            old_head = original[bone_name]["head"]
            old_tail = original[bone_name]["tail"]
            old_vec = old_tail - old_head
            new_head = old_head + accumulated
            new_tail = new_head + old_vec * scale_y
            extension = new_tail - (old_tail + accumulated)
            new_positions[bone_name] = {"head": new_head, "tail": new_tail}
            changed[bone_name] = {
                "scale_y": scale_y,
                "old_length": original[bone_name]["length"],
                "new_length": float((new_tail - new_head).length),
                "extension": [float(value) for value in extension],
            }

            later_scaled = set(chain[index + 1 :]).intersection(segment_scales)
            for descendant in collect_descendants(bone_name, children):
                if descendant in later_scaled:
                    continue
                new_positions[descendant]["head"] += extension
                new_positions[descendant]["tail"] += extension
            accumulated += extension

    impacted = set()
    for root in changed:
        impacted.add(root)
        impacted.update(collect_descendants(root, children))

    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for name in impacted:
            if name in rig.data.edit_bones:
                rig.data.edit_bones[name].use_connect = False
        for name in impacted:
            eb = rig.data.edit_bones.get(name)
            if eb is None:
                continue
            eb.head = new_positions[name]["head"]
            eb.tail = new_positions[name]["tail"]
        for name in impacted:
            eb = rig.data.edit_bones.get(name)
            if eb is None:
                continue
            parent_name = parent.get(name)
            if connect.get(name) and parent_name in new_positions:
                parent_tail = new_positions[parent_name]["tail"]
                if (new_positions[name]["head"] - parent_tail).length < 1.0e-5:
                    eb.use_connect = True
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return changed


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    rig = bpy.data.objects.get(args.armature)
    if rig is None:
        raise RuntimeError(f"Missing armature {args.armature!r}")

    before_constraints = constraint_summary(rig)
    segment_scales = {}
    for side in ("L", "R"):
        segment_scales[f"Scapula_{side}"] = float(args.scapula_scale)
        segment_scales[f"Wrist_{side}"] = float(args.wrist_scale)
    changed = apply_segment_rest_y_scales(rig, segment_scales)
    after_constraints = constraint_summary(rig)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    report = {
        "input": str(args.blend.resolve()),
        "output": str(args.output.resolve()),
        "armature": rig.name,
        "requested_scales": {
            "scapula": float(args.scapula_scale),
            "wrist": float(args.wrist_scale),
        },
        "changed_bones": changed,
        "constraints_before": before_constraints,
        "constraints_after": after_constraints,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OLDIK_SEGMENT_Y_SCALE_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OLDIK_SEGMENT_Y_SCALE_END")


if __name__ == "__main__":
    main()

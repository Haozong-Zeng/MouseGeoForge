#!/usr/bin/env python3
"""Print a compact rig/action/control summary for the currently opened blend."""

from __future__ import annotations

import json

import bpy


def action_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                curves.extend(channelbag.fcurves)
    return curves


def main() -> None:
    scene = bpy.context.scene
    candidates = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    rows = []
    for armature in candidates:
        action = armature.animation_data.action if armature.animation_data else None
        fcurves = action_fcurves(action) if action else []
        frames = sorted({
            int(round(point.co[0]))
            for curve in fcurves
            for point in curve.keyframe_points
        })
        scene.frame_set(scene.frame_start)
        bpy.context.view_layer.update()
        root = armature.pose.bones.get("Root_M")
        head = armature.pose.bones.get("HeadEnd_M")
        pose = {}
        if root and head:
            root_world = armature.matrix_world @ root.head
            for name in (
                "Root_M", "Spine2_M", "Neck3_M", "HeadEnd_M",
                "Wrist_L", "Wrist_R", "Ankle_L", "Ankle_R",
                "Knee_L", "Knee_R",
            ):
                bone = armature.pose.bones.get(name)
                if bone:
                    point = armature.matrix_world @ bone.head
                    pose[name] = [round(float(value), 4) for value in point - root_world]
        rest = {}
        rest_root = armature.data.bones.get("Root_M")
        if rest_root:
            for name in (
                "Root_M", "Spine2_M", "Neck3_M", "HeadEnd_M",
                "Wrist_L", "Wrist_R", "Ankle_L", "Ankle_R",
                "Knee_L", "Knee_R",
            ):
                bone = armature.data.bones.get(name)
                if bone:
                    rest[name] = [round(float(value), 4) for value in bone.head_local - rest_root.head_local]
        rows.append({
            "name": armature.name,
            "bones": len(armature.pose.bones),
            "action": action.name if action else None,
            "fcurves": len(fcurves),
            "keyframes": frames[:20],
            "keyframe_count": len(frames),
            "pose_at_scene_start_relative_root": pose,
            "rest_relative_root": rest,
        })
    controls = []
    tokens = ("Root_M", "Spine2_M", "Wrist_", "Ankle_", "KneePole", "Spine_NURBSCurve")
    for obj in bpy.data.objects:
        if any(token in obj.name for token in tokens):
            action = obj.animation_data.action if obj.animation_data else None
            fcurves = action_fcurves(action) if action else []
            controls.append({
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "action": action.name if action else None,
                "fcurves": len(fcurves),
                "location": [round(float(value), 4) for value in obj.location],
                "matrix_translation": [round(float(value), 4) for value in obj.matrix_world.translation],
            })
    print("MGF_RIG_INSPECT=" + json.dumps({
        "file": bpy.data.filepath,
        "frame_range": [scene.frame_start, scene.frame_end],
        "current_frame": scene.frame_current,
        "armatures": rows,
        "controls": controls,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

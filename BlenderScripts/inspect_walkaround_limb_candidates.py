#!/usr/bin/env python3
"""Inspect armatures and limb-joint candidates in a WalkAround Blender scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def blender_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def vector(value) -> list[float]:
    return [float(component) for component in value]


def matrix(value) -> list[list[float]]:
    return [[float(component) for component in row] for row in value]


def action_summary(obj: bpy.types.Object) -> dict | None:
    animation_data = obj.animation_data
    if animation_data is None or animation_data.action is None:
        return None
    action = animation_data.action
    frame_range = getattr(action, "frame_range", None)
    return {
        "name": action.name,
        "frame_range": vector(frame_range) if frame_range is not None else None,
        "slot_identifier": getattr(animation_data, "action_slot", None).identifier
        if getattr(animation_data, "action_slot", None) is not None
        else None,
    }


def modifier_armatures(obj: bpy.types.Object) -> list[str]:
    names = []
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            names.append(modifier.object.name)
    return names


def inspect_armature(obj: bpy.types.Object, sample_frames: list[int]) -> dict:
    bones = []
    for pose_bone in obj.pose.bones:
        data_bone = pose_bone.bone
        bones.append(
            {
                "name": pose_bone.name,
                "parent": pose_bone.parent.name if pose_bone.parent else None,
                "children": [child.name for child in pose_bone.children],
                "connected": bool(data_bone.use_connect),
                "head_local": vector(data_bone.head_local),
                "tail_local": vector(data_bone.tail_local),
                "length": float(data_bone.length),
            }
        )

    samples = {}
    scene = bpy.context.scene
    for frame in sample_frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples[str(frame)] = {
            pose_bone.name: {
                "head_world": vector(obj.matrix_world @ pose_bone.head),
                "tail_world": vector(obj.matrix_world @ pose_bone.tail),
            }
            for pose_bone in obj.pose.bones
        }

    meshes = []
    for candidate in bpy.data.objects:
        if candidate.type != "MESH":
            continue
        via_parent = candidate.parent == obj
        via_modifier = obj.name in modifier_armatures(candidate)
        if via_parent or via_modifier:
            meshes.append(
                {
                    "name": candidate.name,
                    "parent": candidate.parent.name if candidate.parent else None,
                    "armature_modifiers": modifier_armatures(candidate),
                    "hide_render": bool(candidate.hide_render),
                    "hide_viewport": bool(candidate.hide_viewport),
                    "dimensions": vector(candidate.dimensions),
                    "collections": [collection.name for collection in candidate.users_collection],
                }
            )

    return {
        "name": obj.name,
        "parent": obj.parent.name if obj.parent else None,
        "world_matrix": matrix(obj.matrix_world),
        "world_location": vector(obj.matrix_world.translation),
        "hide_render": bool(obj.hide_render),
        "hide_viewport": bool(obj.hide_viewport),
        "collections": [collection.name for collection in obj.users_collection],
        "action": action_summary(obj),
        "bones": bones,
        "bound_meshes": meshes,
        "samples": samples,
    }


def main() -> None:
    args = blender_args()
    output = Path(args[0]) if args else Path(bpy.path.abspath("//walkaround_limb_candidates.json"))
    scene = bpy.context.scene
    sample_frames = sorted(
        {
            int(scene.frame_start),
            int(round(scene.frame_start + (scene.frame_end - scene.frame_start) * 0.25)),
            int(round(scene.frame_start + (scene.frame_end - scene.frame_start) * 0.5)),
            int(round(scene.frame_start + (scene.frame_end - scene.frame_start) * 0.75)),
            int(scene.frame_end),
        }
    )
    armatures = [
        inspect_armature(obj, sample_frames)
        for obj in bpy.data.objects
        if obj.type == "ARMATURE"
    ]
    cameras = [
        {
            "name": obj.name,
            "world_matrix": matrix(obj.matrix_world),
            "world_location": vector(obj.matrix_world.translation),
            "hide_render": bool(obj.hide_render),
            "collections": [collection.name for collection in obj.users_collection],
            "lens": float(obj.data.lens),
            "sensor_width": float(obj.data.sensor_width),
        }
        for obj in bpy.data.objects
        if obj.type == "CAMERA"
    ]
    payload = {
        "blend_file": bpy.data.filepath,
        "scene": scene.name,
        "active_camera": scene.camera.name if scene.camera else None,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "sample_frames": sample_frames,
        "armatures": armatures,
        "cameras": cameras,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"inspection_written={output}")


if __name__ == "__main__":
    main()

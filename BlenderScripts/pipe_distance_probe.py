import argparse
import json
import math
import sys

import bpy
from mathutils import Vector


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armature", default=None)
    parser.add_argument("--action", default=None)
    parser.add_argument("--pipe", default="Part_WaterPipe")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args(args_after_dash())


def action_fcurves(action):
    fcurves = []
    if getattr(action, "fcurves", None) is not None:
        fcurves.extend(action.fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def action_frames(action):
    frames = set()
    for fcurve in action_fcurves(action):
        for key in fcurve.keyframe_points:
            frames.add(round(float(key.co.x), 6))
    if frames:
        return sorted(frames)

    start, end = action.frame_range
    return [float(frame) for frame in range(math.ceil(start), math.floor(end) + 1)]


def find_armature(name):
    if name:
        return bpy.data.objects[name]
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    with_action = [
        obj
        for obj in armatures
        if obj.animation_data and obj.animation_data.action
    ]
    return with_action[0] if with_action else armatures[0]


def set_action(armature, action_name):
    if action_name:
        action = bpy.data.actions[action_name]
        armature.animation_data_create()
        armature.animation_data.action = action
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    actions = [action for action in bpy.data.actions if action.name.startswith("MIG_")]
    if not actions:
        raise RuntimeError("No action found.")
    armature.animation_data_create()
    armature.animation_data.action = actions[0]
    return actions[0]


def evaluated_vertices_world(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        return [eval_obj.matrix_world @ vertex.co for vertex in eval_mesh.vertices]
    finally:
        eval_obj.to_mesh_clear()


def closest_distance(point, vertices):
    return min((point - vertex).length for vertex in vertices)


def main():
    args = parse_args()
    armature = find_armature(args.armature)
    action = set_action(armature, args.action)
    pipe = bpy.data.objects[args.pipe]
    pipe_vertices = evaluated_vertices_world(pipe)

    probe_bones = [
        "Fingers1_L",
        "IndexFinger3_L",
        "MiddleFinger3_L",
        "RingFinger3_L",
        "PinkyFinger3_L",
        "Fingers1_R",
        "IndexFinger3_R",
        "MiddleFinger3_R",
        "RingFinger3_R",
        "PinkyFinger3_R",
    ]

    rows = []
    for frame in action_frames(action):
        whole = math.floor(frame)
        bpy.context.scene.frame_set(whole, subframe=frame - whole)
        bpy.context.view_layer.update()

        best = None
        for bone_name in probe_bones:
            if bone_name not in armature.pose.bones:
                continue
            pose_bone = armature.pose.bones[bone_name]
            points = [armature.matrix_world @ pose_bone.matrix.translation]
            if bone_name.startswith(("Index", "Middle", "Ring", "Pinky")):
                points.append(
                    armature.matrix_world
                    @ (pose_bone.matrix @ Vector((0.0, pose_bone.bone.length, 0.0)))
                )
            for point_index, point in enumerate(points):
                distance = closest_distance(point, pipe_vertices)
                if best is None or distance < best["distance"]:
                    best = {
                        "frame": float(frame),
                        "bone": bone_name,
                        "point": "tail" if point_index else "head",
                        "distance": float(distance),
                        "world": [float(value) for value in point],
                    }
        if best:
            rows.append(best)

    rows.sort(key=lambda row: row["distance"])
    print("MOUSE_GEO_FORGE_PIPE_DISTANCE_BEGIN")
    print(
        json.dumps(
            {
                "file": bpy.data.filepath,
                "armature": armature.name,
                "action": action.name,
                "pipe": pipe.name,
                "closest": rows[: args.top],
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_PIPE_DISTANCE_END")


if __name__ == "__main__":
    main()

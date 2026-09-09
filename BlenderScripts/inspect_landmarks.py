import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--camera", default="Camera.001")
    parser.add_argument("--action", default=None)
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--object", action="append", default=[])
    return parser.parse_args(args_after_dash())


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def set_action(armature_name, action_name):
    if not action_name:
        return
    armature = bpy.data.objects[armature_name]
    action = bpy.data.actions[action_name]
    armature.animation_data_create()
    armature.animation_data.action = action


def evaluated_vertices_world(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    if obj.type == "MESH":
        eval_mesh = eval_obj.to_mesh()
        try:
            return [eval_obj.matrix_world @ vertex.co for vertex in eval_mesh.vertices]
        finally:
            eval_obj.to_mesh_clear()
    return [eval_obj.matrix_world.translation.copy()]


def bbox_payload(obj, camera):
    vertices = evaluated_vertices_world(obj)
    min_v = Vector((min(v.x for v in vertices), min(v.y for v in vertices), min(v.z for v in vertices)))
    max_v = Vector((max(v.x for v in vertices), max(v.y for v in vertices), max(v.z for v in vertices)))
    center = (min_v + max_v) * 0.5
    scene = bpy.context.scene
    ndc = world_to_camera_view(scene, camera, center)
    cam = camera.matrix_world.inverted() @ center
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "min": [min_v.x, min_v.y, min_v.z],
        "max": [max_v.x, max_v.y, max_v.z],
        "center": [center.x, center.y, center.z],
        "size": [(max_v - min_v).x, (max_v - min_v).y, (max_v - min_v).z],
        "camera": [cam.x, cam.y, cam.z],
        "ndc": [ndc.x, ndc.y, ndc.z],
    }


def point_payload(name, world, camera):
    scene = bpy.context.scene
    ndc = world_to_camera_view(scene, camera, world)
    cam = camera.matrix_world.inverted() @ world
    return {
        "name": name,
        "world": [world.x, world.y, world.z],
        "camera": [cam.x, cam.y, cam.z],
        "ndc": [ndc.x, ndc.y, ndc.z],
    }


def bone_head_world(armature, bone_name):
    return armature.matrix_world @ armature.pose.bones[bone_name].matrix.translation


def main():
    args = parse_args()
    set_action(args.armature, args.action)
    armature = bpy.data.objects[args.armature]
    mesh = bpy.data.objects[args.mesh]
    camera = bpy.data.objects[args.camera]
    frames = args.frame or [float(bpy.context.scene.frame_start)]

    default_objects = [
        "CranialWindow",
        "Brain_Lens.001",
        "Part_PrintedMouseCup",
        "Part_MouseCupSide.001",
        "Part_MouseCupSide.002",
        "LightGuide",
        "Part_HeadBar",
        "Part_HeadClamp001",
        "Part_HeadClamp002",
        "Part_PrintedHeadClampHolder",
        "MS_Mouse",
    ]
    object_names = args.object or default_objects
    object_names = [name for name in object_names if name in bpy.data.objects]

    landmark_bones = [
        "Root_M",
        "Chest_M",
        "Neck3_M",
        "Neck1_M",
        "Neck2_M",
        "Head1_M",
        "HeadEnd_M",
        "Tail0_M",
        "Tail1_M",
        "Tail2_M",
        "Tail5_M",
        "Tail8_M",
        "Tail11_M",
        "Scapula_L",
        "Wrist_L",
        "Fingers1_L",
        "Scapula_R",
        "Wrist_R",
        "Fingers1_R",
    ]

    payload = {
        "file": bpy.data.filepath,
        "camera": camera.name,
        "armature": armature.name,
        "action": armature.animation_data.action.name
        if armature.animation_data and armature.animation_data.action
        else None,
        "frames": [],
    }

    for frame in frames:
        set_frame(frame)
        frame_payload = {
            "frame": frame,
            "objects": [bbox_payload(bpy.data.objects[name], camera) for name in object_names],
            "bones": [
                point_payload(name, bone_head_world(armature, name), camera)
                for name in landmark_bones
                if name in armature.pose.bones
            ],
        }
        payload["frames"].append(frame_payload)

    print("MOUSE_GEO_FORGE_LANDMARKS_BEGIN")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("MOUSE_GEO_FORGE_LANDMARKS_END")


if __name__ == "__main__":
    main()

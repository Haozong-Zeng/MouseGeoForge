import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=float, default=None)
    parser.add_argument("--armature", default="\u8001\u9f20_rig")
    parser.add_argument("--action", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(args_after_dash())


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def evaluated_bbox(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    points = []
    if obj.type == "MESH":
        try:
            mesh = eval_obj.to_mesh()
            try:
                for vertex in mesh.vertices:
                    points.append(eval_obj.matrix_world @ vertex.co)
            finally:
                eval_obj.to_mesh_clear()
        except RuntimeError:
            pass
    if not points:
        points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    if not points:
        return None
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v


def vec_json(vector):
    return [round(float(value), 5) for value in vector]


def object_payload(obj):
    bbox = evaluated_bbox(obj)
    if not bbox:
        return None
    min_v, max_v = bbox
    return {
        "name": obj.name,
        "type": obj.type,
        "hidden_viewport": bool(obj.hide_viewport),
        "hidden_render": bool(obj.hide_render),
        "location": vec_json(obj.matrix_world.translation),
        "bbox_min": vec_json(min_v),
        "bbox_max": vec_json(max_v),
        "center": vec_json((min_v + max_v) * 0.5),
        "size": vec_json(max_v - min_v),
    }


def main():
    args = parse_args()
    if args.action:
        rig = bpy.data.objects[args.armature]
        action = bpy.data.actions[args.action]
        rig.animation_data_create()
        rig.animation_data.action = action
    if args.frame is not None:
        set_frame(args.frame)

    object_names = [
        name
        for name in bpy.data.objects.keys()
        if name.startswith("MS_Mouse")
        or name.startswith("Part_")
        or name.startswith("Cranial")
        or name in {"DeformationSystem.001", args.armature}
    ]
    objects = [object_payload(bpy.data.objects[name]) for name in sorted(object_names)]
    objects = [payload for payload in objects if payload]

    controls = {}
    rig = bpy.data.objects.get(args.armature)
    if rig and rig.type == "ARMATURE":
        for name in [
            "c_pos",
            "c_traj",
            "c_root_master.x",
            "c_root.x",
            "root.x",
            "c_spine_01.x",
            "c_spine_02.x",
            "c_spine_03.x",
            "c_neck.x",
            "c_head.x",
        ]:
            pb = rig.pose.bones.get(name)
            if pb:
                controls[name] = {
                    "location": vec_json(pb.location),
                    "world": vec_json(rig.matrix_world @ pb.matrix.translation),
                }

    payload = {
        "file": bpy.data.filepath,
        "frame": bpy.context.scene.frame_current + bpy.context.scene.frame_subframe,
        "scene_camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
        "objects": objects,
        "controls": controls,
    }
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import sys

import bpy


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default=None)
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--bone", action="append", default=[])
    return parser.parse_args(args_after_dash())


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def main():
    args = parse_args()
    armature = bpy.data.objects[args.armature]
    if args.action:
        action = bpy.data.actions[args.action]
        armature.animation_data_create()
        armature.animation_data.action = action
    frames = args.frame or [float(bpy.context.scene.frame_start)]
    bones = args.bone or [bone.name for bone in armature.data.bones]
    rows = []
    for frame in frames:
        set_frame(frame)
        for name in bones:
            pb = armature.pose.bones[name]
            rows.append(
                {
                    "frame": frame,
                    "bone": name,
                    "location": [float(v) for v in pb.location],
                    "rotation_quaternion": [float(v) for v in pb.rotation_quaternion],
                    "scale": [float(v) for v in pb.scale],
                    "matrix_translation": [float(v) for v in pb.matrix.translation],
                }
            )
    print("MOUSE_GEO_FORGE_POSE_SCALE_BEGIN")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("MOUSE_GEO_FORGE_POSE_SCALE_END")


if __name__ == "__main__":
    main()

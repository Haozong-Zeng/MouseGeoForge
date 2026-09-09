import argparse
import json
import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_forelimb_interpolation as interp


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-blend", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--base-action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--candidate-action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--side", action="append", choices=("L", "R"), default=[])
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--include-fingers", action="store_true")
    parser.add_argument("--frame", type=float, action="append", default=[])
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


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def set_action(armature, action):
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.view_layer.update()


def load_candidate_action(path, action_name):
    with bpy.data.libraries.load(str(Path(path).resolve()), link=False) as (data_from, data_to):
        if action_name not in data_from.actions:
            raise RuntimeError(f"Action {action_name!r} not found in {path}")
        data_to.actions = [action_name]
    action = data_to.actions[0]
    action.name = f"{action_name}_candidate_tail_shape"
    return action


def keyframe_bones(armature, bone_names, frame):
    for name in bone_names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        pose_bone.keyframe_insert(data_path="location", frame=frame)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pose_bone.keyframe_insert(data_path="scale", frame=frame)


def main():
    args = parse_args()
    armature = bpy.data.objects[args.armature]
    base_action = bpy.data.actions[args.base_action]
    candidate_action = load_candidate_action(args.candidate_blend, args.candidate_action)
    sides = args.side or ["L", "R"]
    alpha = max(0.0, min(float(args.alpha), 1.0))
    frames = sorted(args.frame) if args.frame else action_frames(base_action)

    records = []
    for frame in frames:
        set_action(armature, base_action)
        set_frame(frame)
        for side in sides:
            bone_names = interp.side_bone_names(side, args.include_fingers)
            base_bases = interp.decompose_pose_bases(armature, bone_names)

            set_action(armature, candidate_action)
            set_frame(frame)
            candidate_bases = interp.decompose_pose_bases(armature, bone_names)

            set_action(armature, base_action)
            set_frame(frame)
            interp.apply_interpolated_bases(armature, base_bases, candidate_bases, alpha)
            keyframe_bones(armature, bone_names, frame)
            records.append(
                {
                    "frame": float(frame),
                    "side": side,
                    "bone_count": len(bone_names),
                    "alpha": alpha,
                    "include_fingers": bool(args.include_fingers),
                }
            )

    set_action(armature, base_action)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.resolve()))
    report = {
        "base_file": bpy.data.filepath,
        "candidate_blend": str(Path(args.candidate_blend).resolve()),
        "output": str(output_path.resolve()),
        "base_action": base_action.name,
        "candidate_action": candidate_action.name,
        "sides": sides,
        "alpha": alpha,
        "include_fingers": bool(args.include_fingers),
        "frame_count": len(frames),
        "records": records[:20],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_BLEND_FORELIMB_BEGIN")
    print(json.dumps({k: report[k] for k in ("output", "sides", "alpha", "include_fingers", "frame_count")}, indent=2))
    print("MOUSE_GEO_FORGE_BLEND_FORELIMB_END")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontlimb_quality_check as qc
import migrate_animations_to_ik as mig


HEAD_CANDIDATES = (
    "HeadEnd_M",
    "Head1_M",
    "Neck3_M",
    "Neck2_M",
    "Neck1_M",
    "Jaw_M",
    "Jaw1_M",
    "Nose_M",
)


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--side", action="append", choices=("L", "R"), default=[])
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--report", required=True)
    parser.add_argument("--print-bones", action="store_true")
    return parser.parse_args(args_after_dash())


def point_or_none(value):
    if value is None:
        return None
    return [float(component) for component in value]


def distance(a, b):
    if a is None or b is None:
        return None
    return float((a - b).length)


def midpoint(points):
    points = [point for point in points if point is not None]
    if not points:
        return None
    return sum(points, Vector()) / len(points)


def pose_bone_point(armature, bone_name, endpoint):
    if endpoint == "head":
        return qc.pose_bone_head_world(armature, bone_name)
    if endpoint == "tail":
        return qc.pose_bone_tail_world(armature, bone_name)
    raise ValueError(endpoint)


def available_head_points(armature):
    rows = {}
    for name in HEAD_CANDIDATES:
        if name not in armature.pose.bones:
            continue
        rows[name] = {
            "head": qc.pose_bone_head_world(armature, name),
            "tail": qc.pose_bone_tail_world(armature, name),
        }
    return rows


def mouth_proxy(armature):
    head_end = qc.pose_bone_tail_world(armature, "HeadEnd_M")
    if head_end is not None:
        return head_end, "HeadEnd_M.tail"
    head_end = qc.pose_bone_head_world(armature, "HeadEnd_M")
    if head_end is not None:
        return head_end, "HeadEnd_M.head"
    head1_tail = qc.pose_bone_tail_world(armature, "Head1_M")
    if head1_tail is not None:
        return head1_tail, "Head1_M.tail"
    return None, None


def head_center_proxy(armature):
    head = qc.pose_bone_head_world(armature, "Head1_M")
    end = qc.pose_bone_tail_world(armature, "HeadEnd_M")
    center = midpoint([head, end])
    if center is not None:
        return center, "mid(Head1_M.head,HeadEnd_M.tail)"
    neck = qc.pose_bone_tail_world(armature, "Neck3_M")
    center = midpoint([head, neck])
    if center is not None:
        return center, "mid(Head1_M.head,Neck3_M.tail)"
    return head, "Head1_M.head" if head is not None else None


def side_points(armature, side):
    scapula = f"Scapula_{side}"
    wrist = f"Wrist_{side}"
    fingers = f"Fingers1_{side}"
    fingertips = []
    for prefix in ("IndexFinger3", "MiddleFinger3", "RingFinger3", "PinkyFinger3"):
        tip = qc.pose_bone_tail_world(armature, f"{prefix}_{side}")
        if tip is not None:
            fingertips.append(tip)
    fingertip_center = midpoint(fingertips)
    return {
        "shoulder": qc.pose_bone_head_world(armature, scapula),
        "elbow": qc.pose_bone_tail_world(armature, scapula),
        "wrist_head": qc.pose_bone_head_world(armature, wrist),
        "palm": qc.pose_bone_tail_world(armature, wrist),
        "fingers_base": qc.pose_bone_head_world(armature, fingers),
        "fingers_tip": qc.pose_bone_tail_world(armature, fingers),
        "fingertip_center": fingertip_center,
    }


def semantic_metrics(armature, side):
    points = side_points(armature, side)
    mouth, mouth_source = mouth_proxy(armature)
    head_center, head_center_source = head_center_proxy(armature)
    shoulder = points["shoulder"]
    elbow = points["elbow"]
    palm = points["palm"]
    fingers_tip = points["fingers_tip"] or points["fingertip_center"]

    upper = distance(shoulder, elbow)
    lower = distance(elbow, palm)
    arm_length = None
    if upper is not None and lower is not None:
        arm_length = upper + lower

    elbow_mouth = distance(elbow, mouth)
    palm_mouth = distance(palm, mouth)
    tip_mouth = distance(fingers_tip, mouth)
    elbow_head = distance(elbow, head_center)
    palm_head = distance(palm, head_center)

    chain_span = distance(shoulder, palm)
    elbow_projection_t = None
    elbow_line_distance = None
    if shoulder is not None and elbow is not None and palm is not None:
        axis = palm - shoulder
        if axis.length > 1.0e-8:
            rel = elbow - shoulder
            elbow_projection_t = float(rel.dot(axis) / axis.length_squared)
            elbow_line_distance = float((rel - axis * elbow_projection_t).length)

    elbow_nearer_mouth_than_palm = None
    if elbow_mouth is not None and palm_mouth is not None:
        elbow_nearer_mouth_than_palm = elbow_mouth < palm_mouth

    elbow_mouth_ratio = None
    if elbow_mouth is not None and arm_length:
        elbow_mouth_ratio = elbow_mouth / arm_length

    return {
        "mouth_proxy": mouth_source,
        "head_center_proxy": head_center_source,
        "points": {key: point_or_none(value) for key, value in points.items()},
        "upper_length": upper,
        "lower_length": lower,
        "arm_length": arm_length,
        "chain_span": chain_span,
        "elbow_projection_t": elbow_projection_t,
        "elbow_line_distance": elbow_line_distance,
        "dist_elbow_to_mouth": elbow_mouth,
        "dist_palm_to_mouth": palm_mouth,
        "dist_tip_to_mouth": tip_mouth,
        "dist_elbow_to_head_center": elbow_head,
        "dist_palm_to_head_center": palm_head,
        "elbow_nearer_mouth_than_palm": elbow_nearer_mouth_than_palm,
        "elbow_mouth_ratio": elbow_mouth_ratio,
        "flag_elbow_taking_palm_role": bool(
            elbow_nearer_mouth_than_palm
            and elbow_mouth_ratio is not None
            and elbow_mouth_ratio < 1.0
        ),
    }


def main():
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    armature = mig.find_armature(args.armature, require_action=False)
    action = bpy.data.actions.get(args.action) if args.action else None
    if action is None and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
    if action is None and bpy.data.actions:
        action = sorted(bpy.data.actions, key=lambda item: item.name)[0]
    if action is None:
        raise RuntimeError("No action found.")
    qc.set_action(armature, action)

    frames = sorted(args.frame) if args.frame else [float(bpy.context.scene.frame_current)]
    sides = args.side or ["R"]
    report = {
        "blend": str(Path(args.blend).resolve()),
        "action": action.name,
        "sides": sides,
        "frames": [],
    }
    if args.print_bones:
        report["matching_bones"] = [
            bone.name
            for bone in armature.pose.bones
            if any(token.lower() in bone.name.lower() for token in ("head", "jaw", "nose", "neck"))
        ]

    for frame in frames:
        qc.set_frame(frame)
        frame_row = {"frame": float(frame), "sides": {}, "head_points": {}}
        for name, endpoints in available_head_points(armature).items():
            frame_row["head_points"][name] = {
                key: point_or_none(value)
                for key, value in endpoints.items()
            }
        for side in sides:
            frame_row["sides"][side] = semantic_metrics(armature, side)
        report["frames"].append(frame_row)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_SEMANTIC_SPACE_AUDIT_BEGIN")
    print(json.dumps({"report": str(path.resolve()), "frame_count": len(frames)}, indent=2))
    print("MOUSE_GEO_FORGE_SEMANTIC_SPACE_AUDIT_END")


if __name__ == "__main__":
    main()

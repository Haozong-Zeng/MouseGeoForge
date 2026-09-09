"""Audit hand-to-mouth distance and finger bone/mesh alignment for one frame."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import frontlimb_quality_check as qc  # noqa: E402
import migrate_animations_to_ik as mig  # noqa: E402


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--target-blend", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--target-armature", default="DeformationSystem")
    parser.add_argument("--source-mesh", default="MS_Mouse")
    parser.add_argument("--target-mesh", default="MS_Mouse")
    parser.add_argument("--source-action", default=None)
    parser.add_argument("--target-action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--frame", type=float, required=True)
    parser.add_argument("--side", choices=("L", "R"), default="R")
    parser.add_argument("--min-weight", type=float, default=0.001)
    return parser.parse_args(args_after_dash())


def set_action(armature, action_name: str | None):
    armature.animation_data_create()
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Action {action_name!r} not found")
        armature.animation_data.action = action
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    actions = list(bpy.data.actions)
    if not actions:
        raise RuntimeError("No actions found")
    action = sorted(actions, key=lambda item: item.name)[0]
    armature.animation_data.action = action
    return action


def vec_payload(value: Vector | None) -> list[float] | None:
    if value is None:
        return None
    return [float(value.x), float(value.y), float(value.z)]


def distance(a: Vector | None, b: Vector | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a - b).length)


def pose_head(armature, bone_name: str) -> Vector | None:
    return qc.pose_bone_head_world(armature, bone_name)


def pose_tail(armature, bone_name: str) -> Vector | None:
    return qc.pose_bone_tail_world(armature, bone_name)


def mouth_proxy(armature):
    for bone_name, endpoint in (
        ("HeadEnd_M", "tail"),
        ("JawEnd_M", "tail"),
        ("HeadEnd_M", "head"),
        ("Jaw_M", "tail"),
        ("Head1_M", "tail"),
    ):
        point = pose_head(armature, bone_name) if endpoint == "head" else pose_tail(armature, bone_name)
        if point is not None:
            return point, f"{bone_name}.{endpoint}"
    return None, None


def head_proxy(armature):
    head = pose_head(armature, "Head1_M")
    end = pose_tail(armature, "HeadEnd_M")
    if head is not None and end is not None:
        return (head + end) * 0.5, "mid(Head1_M.head,HeadEnd_M.tail)"
    neck = pose_tail(armature, "Neck3_M")
    if head is not None and neck is not None:
        return (head + neck) * 0.5, "mid(Head1_M.head,Neck3_M.tail)"
    return head, "Head1_M.head" if head is not None else None


def hand_points(armature, side: str) -> dict[str, Vector | None]:
    tips = []
    for prefix in ("IndexFinger3", "MiddleFinger3", "RingFinger3", "PinkyFinger3"):
        point = pose_tail(armature, f"{prefix}_{side}")
        if point is not None:
            tips.append(point)
    return {
        "shoulder": pose_head(armature, f"Scapula_{side}"),
        "elbow": pose_tail(armature, f"Scapula_{side}"),
        "wrist_head": pose_head(armature, f"Wrist_{side}"),
        "palm": pose_tail(armature, f"Wrist_{side}"),
        "fingers_base": pose_head(armature, f"Fingers1_{side}"),
        "fingers_tip": pose_tail(armature, f"Fingers1_{side}"),
        "fingertip_center": (sum(tips, Vector()) / len(tips)) if tips else None,
    }


def finger_bone_names(side: str) -> list[str]:
    names = [f"Fingers1_{side}"]
    for prefix in ("IndexFinger", "MiddleFinger", "RingFinger", "PinkyFinger"):
        for index in (1, 2, 3):
            names.append(f"{prefix}{index}_{side}")
    return names


def point_segment_distance(point: Vector, start: Vector, end: Vector) -> float:
    segment = end - start
    denom = segment.length_squared
    if denom <= 1.0e-12:
        return float((point - start).length)
    t = max(0.0, min(1.0, (point - start).dot(segment) / denom))
    closest = start + segment * t
    return float((point - closest).length)


def point_segment_parameter(point: Vector, start: Vector, end: Vector) -> float | None:
    segment = end - start
    denom = segment.length_squared
    if denom <= 1.0e-12:
        return None
    return float((point - start).dot(segment) / denom)


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(int(round((len(ordered) - 1) * 0.95)), len(ordered) - 1)
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def weighted_group_indices(mesh, group_name: str, min_weight: float) -> list[int]:
    group = mesh.vertex_groups.get(group_name)
    if group is None:
        return []
    out = []
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            if membership.group == group.index and membership.weight >= min_weight:
                out.append(vertex.index)
                break
    return out


def constraint_payload(constraint) -> dict:
    row = {
        "name": constraint.name,
        "type": constraint.type,
        "target": getattr(getattr(constraint, "target", None), "name", None),
        "subtarget": getattr(constraint, "subtarget", ""),
        "mute": bool(getattr(constraint, "mute", False)),
        "influence": float(getattr(constraint, "influence", 0.0)),
    }
    if constraint.type == "LIMIT_SCALE":
        for attr in ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "owner_space"):
            if hasattr(constraint, attr):
                value = getattr(constraint, attr)
                row[attr] = float(value) if isinstance(value, (int, float)) else value
    if constraint.type == "IK":
        for attr in ("chain_count", "use_stretch"):
            if hasattr(constraint, attr):
                row[attr] = getattr(constraint, attr)
    return row


def finger_alignment(armature, mesh, side: str, min_weight: float) -> dict:
    vertices, _polygons = qc.evaluated_mesh_world(mesh)
    out = {}
    for bone_name in finger_bone_names(side):
        bone = armature.data.bones.get(bone_name)
        pose_bone = armature.pose.bones.get(bone_name)
        if bone is None or pose_bone is None:
            continue
        head = pose_head(armature, bone_name)
        tail = pose_tail(armature, bone_name)
        indices = weighted_group_indices(mesh, bone_name, min_weight)
        distances = []
        params = []
        if head is not None and tail is not None:
            for index in indices:
                if index < len(vertices):
                    point = vertices[index]
                    distances.append(point_segment_distance(point, head, tail))
                    param = point_segment_parameter(point, head, tail)
                    if param is not None:
                        params.append(param)
        param_ordered = sorted(params)
        outside_before = sum(1 for value in params if value < 0.0)
        outside_after = sum(1 for value in params if value > 1.0)
        out[bone_name] = {
            "rest_length": float(bone.length),
            "pose_length": distance(head, tail),
            "pose_scale": [float(value) for value in pose_bone.scale],
            "constraints": [constraint_payload(constraint) for constraint in pose_bone.constraints],
            "weighted_vertex_count": len(indices),
            "mesh_to_pose_segment": stats(distances),
            "mesh_projection_t": {
                "count": len(params),
                "min": None if not params else float(param_ordered[0]),
                "p05": None if not params else float(param_ordered[min(int(round((len(param_ordered) - 1) * 0.05)), len(param_ordered) - 1)]),
                "median": None if not params else float(param_ordered[len(param_ordered) // 2]),
                "p95": None if not params else float(param_ordered[min(int(round((len(param_ordered) - 1) * 0.95)), len(param_ordered) - 1)]),
                "max": None if not params else float(param_ordered[-1]),
                "outside_before_count": int(outside_before),
                "outside_after_count": int(outside_after),
                "outside_fraction": 0.0 if not params else float((outside_before + outside_after) / len(params)),
            },
        }
    return out


def arm_chain_metrics(points: dict[str, Vector | None]) -> dict:
    shoulder = points.get("shoulder")
    elbow = points.get("elbow")
    palm = points.get("palm")
    if shoulder is None or elbow is None or palm is None:
        return {"available": False}
    upper = (elbow - shoulder).length
    lower = (palm - elbow).length
    reach = (palm - shoulder).length
    angle = qc.angle_degrees(shoulder - elbow, palm - elbow)
    return {
        "available": True,
        "upper_length": float(upper),
        "lower_length": float(lower),
        "reach": float(reach),
        "reach_ratio": float(reach / max(upper + lower, 1.0e-8)),
        "elbow_angle_deg": None if angle is None else float(angle),
    }


def audit_blend(path: Path, armature_name: str, mesh_name: str, action_name: str | None, frame: float, side: str, min_weight: float) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    armature = mig.find_armature(armature_name, require_action=False)
    mesh = bpy.data.objects.get(mesh_name)
    if mesh is None:
        raise RuntimeError(f"Mesh {mesh_name!r} not found in {path}")
    action = set_action(armature, action_name)
    qc.set_frame(frame)
    mouth, mouth_source = mouth_proxy(armature)
    head, head_source = head_proxy(armature)
    points = hand_points(armature, side)
    distances_to_mouth = {name: distance(point, mouth) for name, point in points.items()}
    distances_to_head = {name: distance(point, head) for name, point in points.items()}
    return {
        "blend": str(path.resolve()),
        "armature": armature.name,
        "mesh": mesh.name,
        "action": action.name,
        "frame": float(frame),
        "side": side,
        "mouth_proxy": {"source": mouth_source, "point": vec_payload(mouth)},
        "head_proxy": {"source": head_source, "point": vec_payload(head)},
        "hand_points": {name: vec_payload(point) for name, point in points.items()},
        "distance_to_mouth": distances_to_mouth,
        "distance_to_head": distances_to_head,
        "arm_chain": arm_chain_metrics(points),
        "finger_alignment": finger_alignment(armature, mesh, side, min_weight),
    }


def main() -> None:
    args = parse_args()
    source = audit_blend(
        args.source_blend,
        args.source_armature,
        args.source_mesh,
        args.source_action,
        args.frame,
        args.side,
        args.min_weight,
    )
    targets = [
        audit_blend(
            path,
            args.target_armature,
            args.target_mesh,
            args.target_action,
            args.frame,
            args.side,
            args.min_weight,
        )
        for path in args.target_blend
    ]
    report = {"source": source, "targets": targets}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HAND_MOUTH_FINGER_AUDIT_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("HAND_MOUTH_FINGER_AUDIT_END")


if __name__ == "__main__":
    main()

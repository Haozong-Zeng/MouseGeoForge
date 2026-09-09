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


FINGER_TIP_PREFIXES = ("IndexFinger3", "MiddleFinger3", "RingFinger3", "PinkyFinger3")


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default=None)
    parser.add_argument("--side", action="append", choices=("L", "R"), default=[])
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--report", required=True)
    return parser.parse_args(args_after_dash())


def point_payload(point):
    if point is None:
        return None
    return [float(value) for value in point]


def distance(a, b):
    if a is None or b is None:
        return None
    return float((a - b).length)


def midpoint(points):
    points = [point for point in points if point is not None]
    if not points:
        return None
    return sum(points, Vector()) / len(points)


def choose_action(armature, action_name):
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Action {action_name!r} not found.")
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    if bpy.data.actions:
        return sorted(bpy.data.actions, key=lambda item: item.name)[0]
    raise RuntimeError("No action found.")


def collect_vertex_indices_by_group_substring(mesh, substring, side):
    group_indices = {
        group.index
        for group in mesh.vertex_groups
        if substring.lower() in group.name.lower() and group.name.endswith(f"_{side}")
    }
    indices = set()
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            if membership.group in group_indices and membership.weight > 0.001:
                indices.add(vertex.index)
                break
    return sorted(indices)


def nearest_vertex_distance(point, vertices, indices):
    if point is None or not indices:
        return None
    best = None
    for index in indices:
        if index >= len(vertices):
            continue
        dist = float((point - vertices[index]).length)
        if best is None or dist < best["distance"]:
            best = {
                "distance": dist,
                "vertex_index": int(index),
                "vertex": point_payload(vertices[index]),
            }
    return best


def forelimb_points(armature, side):
    tips = [
        qc.pose_bone_tail_world(armature, f"{prefix}_{side}")
        for prefix in FINGER_TIP_PREFIXES
    ]
    return {
        "shoulder": qc.pose_bone_head_world(armature, f"Scapula_{side}"),
        "elbow": qc.pose_bone_tail_world(armature, f"Scapula_{side}"),
        "wrist_head": qc.pose_bone_head_world(armature, f"Wrist_{side}"),
        "palm": qc.pose_bone_tail_world(armature, f"Wrist_{side}"),
        "fingers_base": qc.pose_bone_head_world(armature, f"Fingers1_{side}"),
        "fingers_tip": qc.pose_bone_tail_world(armature, f"Fingers1_{side}"),
        "fingertip_center": midpoint(tips),
    }


def ear_bone_points(armature, side):
    rows = {}
    for prefix in ("Ear01", "Ear02", "Ear03"):
        bone_name = f"{prefix}_{side}"
        if bone_name not in armature.pose.bones:
            continue
        rows[bone_name] = {
            "head": qc.pose_bone_head_world(armature, bone_name),
            "tail": qc.pose_bone_tail_world(armature, bone_name),
        }
    return rows


def nearest_ear_bone_distance(point, ear_points):
    best = None
    for bone_name, endpoints in ear_points.items():
        for endpoint, value in endpoints.items():
            dist = distance(point, value)
            if dist is None:
                continue
            row = {
                "distance": dist,
                "bone": bone_name,
                "endpoint": endpoint,
                "point": point_payload(value),
            }
            if best is None or dist < best["distance"]:
                best = row
    return best


def frame_side_metrics(armature, mesh, side):
    vertices, _polygons = qc.evaluated_mesh_world(mesh)
    points = forelimb_points(armature, side)
    ear_indices_same = collect_vertex_indices_by_group_substring(mesh, "Ear", side)
    other_side = "L" if side == "R" else "R"
    ear_indices_other = collect_vertex_indices_by_group_substring(mesh, "Ear", other_side)
    ear_bones_same = ear_bone_points(armature, side)
    ear_bones_other = ear_bone_points(armature, other_side)
    probes = ("wrist_head", "palm", "fingers_base", "fingers_tip", "fingertip_center")
    rows = {}
    for probe in probes:
        point = points.get(probe)
        rows[probe] = {
            "point": point_payload(point),
            "same_ear_mesh": nearest_vertex_distance(point, vertices, ear_indices_same),
            "other_ear_mesh": nearest_vertex_distance(point, vertices, ear_indices_other),
            "same_ear_bone": nearest_ear_bone_distance(point, ear_bones_same),
            "other_ear_bone": nearest_ear_bone_distance(point, ear_bones_other),
        }
    return {
        "points": {key: point_payload(value) for key, value in points.items()},
        "same_ear_vertex_count": len(ear_indices_same),
        "other_ear_vertex_count": len(ear_indices_other),
        "probes": rows,
    }


def main():
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    armature = bpy.data.objects.get(args.armature) or mig.find_armature(args.armature, require_action=False)
    mesh = bpy.data.objects.get(args.mesh)
    if mesh is None:
        raise RuntimeError(f"Mesh {args.mesh!r} not found.")
    action = choose_action(armature, args.action)
    qc.set_action(armature, action)

    frames = sorted(args.frame) if args.frame else [float(bpy.context.scene.frame_current)]
    sides = args.side or ["R"]
    report = {
        "blend": str(Path(args.blend).resolve()),
        "action": action.name,
        "sides": sides,
        "frames": [],
    }
    for frame in frames:
        qc.set_frame(frame)
        row = {"frame": float(frame), "sides": {}}
        for side in sides:
            row["sides"][side] = frame_side_metrics(armature, mesh, side)
        report["frames"].append(row)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_WRIST_EAR_AUDIT_BEGIN")
    print(json.dumps({"report": str(path.resolve()), "frame_count": len(frames)}, indent=2))
    print("MOUSE_GEO_FORGE_WRIST_EAR_AUDIT_END")


if __name__ == "__main__":
    main()

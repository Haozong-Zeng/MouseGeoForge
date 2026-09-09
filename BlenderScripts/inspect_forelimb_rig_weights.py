import argparse
import json
import sys
from pathlib import Path

import bpy


FORELIMB_PREFIXES = [
    "Scapula",
    "Wrist",
    "Fingers1",
    "IndexFinger1",
    "IndexFinger2",
    "IndexFinger3",
    "MiddleFinger1",
    "MiddleFinger2",
    "MiddleFinger3",
    "RingFinger1",
    "RingFinger2",
    "RingFinger3",
    "PinkyFinger1",
    "PinkyFinger2",
    "PinkyFinger3",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--output", required=True)
    return parser.parse_args(args_after_dash())


def vector_json(vector):
    return [float(vector.x), float(vector.y), float(vector.z)]


def object_records():
    records = []
    for obj in bpy.data.objects:
        if obj.type not in {"MESH", "ARMATURE"}:
            continue
        armature_mods = []
        for modifier in getattr(obj, "modifiers", []):
            if modifier.type == "ARMATURE":
                armature_mods.append(
                    {
                        "name": modifier.name,
                        "object": modifier.object.name if modifier.object else None,
                    }
                )
        records.append(
            {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "parent_type": obj.parent_type,
                "hide_render": bool(obj.hide_render),
                "hide_viewport": bool(obj.hide_viewport),
                "armature_modifiers": armature_mods,
                "vertex_group_count": len(getattr(obj, "vertex_groups", [])),
                "action": obj.animation_data.action.name
                if obj.animation_data and obj.animation_data.action
                else None,
            }
        )
    return records


def pose_bone_record(armature, name):
    pb = armature.pose.bones.get(name)
    bone = armature.data.bones.get(name)
    if pb is None or bone is None:
        return {"name": name, "available": False}
    constraints = []
    for constraint in pb.constraints:
        constraints.append(
            {
                "name": constraint.name,
                "type": constraint.type,
                "target": constraint.target.name if getattr(constraint, "target", None) else None,
                "subtarget": getattr(constraint, "subtarget", None),
                "influence": float(getattr(constraint, "influence", 0.0)),
            }
        )
    return {
        "name": name,
        "available": True,
        "parent": bone.parent.name if bone.parent else None,
        "use_deform": bool(bone.use_deform),
        "use_connect": bool(bone.use_connect),
        "head_local": vector_json(bone.head_local),
        "tail_local": vector_json(bone.tail_local),
        "length": float(bone.length),
        "roll": None if not hasattr(bone, "roll") else float(bone.roll),
        "pose_location_length": float(pb.location.length),
        "pose_scale": [float(value) for value in pb.scale],
        "constraints": constraints,
    }


def vertex_group_index(mesh, name):
    group = mesh.vertex_groups.get(name)
    return None if group is None else group.index


def vertex_weight(vertex, group_index):
    if group_index is None:
        return 0.0
    for item in vertex.groups:
        if item.group == group_index:
            return float(item.weight)
    return 0.0


def side_weight_summary(mesh, side):
    names = [f"{prefix}_{side}" for prefix in FORELIMB_PREFIXES]
    group_indices = {name: vertex_group_index(mesh, name) for name in names}
    group_counts = {name: 0 for name in names}
    max_group_counts = {name: 0 for name in names}
    forelimb_vertices = 0
    multi_influence_vertices = 0
    scapula_wrist_blend_vertices = 0
    abrupt_joint_vertices = 0
    total_weights = {name: 0.0 for name in names}
    max_weight_values = {name: 0.0 for name in names}
    examples = []

    scapula = group_indices.get(f"Scapula_{side}")
    wrist = group_indices.get(f"Wrist_{side}")
    finger_indices = {
        index
        for name, index in group_indices.items()
        if index is not None and not name.startswith(("Scapula_", "Wrist_"))
    }

    for vertex in mesh.data.vertices:
        weights = {}
        for name, index in group_indices.items():
            weight = vertex_weight(vertex, index)
            if weight > 1.0e-4:
                weights[name] = weight
                group_counts[name] += 1
                total_weights[name] += weight
                max_weight_values[name] = max(max_weight_values[name], weight)
        if not weights:
            continue
        forelimb_vertices += 1
        max_name, max_weight = max(weights.items(), key=lambda item: item[1])
        max_group_counts[max_name] += 1
        if len(weights) >= 2:
            multi_influence_vertices += 1
        scapula_weight = vertex_weight(vertex, scapula)
        wrist_weight = vertex_weight(vertex, wrist)
        finger_weight = sum(vertex_weight(vertex, index) for index in finger_indices)
        if scapula_weight > 0.05 and wrist_weight > 0.05:
            scapula_wrist_blend_vertices += 1
        if (scapula_weight > 0.85 and wrist_weight < 0.02) or (wrist_weight > 0.85 and scapula_weight < 0.02):
            if finger_weight < 0.05:
                abrupt_joint_vertices += 1
        if len(examples) < 12 and max_weight > 0.98 and len(weights) == 1:
            examples.append(
                {
                    "vertex_index": int(vertex.index),
                    "co": vector_json(vertex.co),
                    "sole_group": max_name,
                    "weight": float(max_weight),
                }
            )

    return {
        "side": side,
        "group_indices": group_indices,
        "forelimb_vertex_count": int(forelimb_vertices),
        "multi_influence_vertex_count": int(multi_influence_vertices),
        "multi_influence_fraction": float(multi_influence_vertices / max(forelimb_vertices, 1)),
        "scapula_wrist_blend_vertex_count": int(scapula_wrist_blend_vertices),
        "scapula_wrist_blend_fraction": float(scapula_wrist_blend_vertices / max(forelimb_vertices, 1)),
        "abrupt_single_bone_vertex_count": int(abrupt_joint_vertices),
        "abrupt_single_bone_fraction": float(abrupt_joint_vertices / max(forelimb_vertices, 1)),
        "group_vertex_counts": group_counts,
        "max_group_counts": max_group_counts,
        "mean_weight_by_group": {
            name: float(total_weights[name] / max(group_counts[name], 1))
            for name in names
        },
        "max_weight_by_group": max_weight_values,
        "single_group_examples": examples,
    }


def main():
    args = parse_args()
    armature = bpy.data.objects.get(args.armature)
    mesh = bpy.data.objects.get(args.mesh)
    if armature is None or mesh is None:
        raise RuntimeError("Armature or mesh not found.")

    forelimb_bones = {
        side: [pose_bone_record(armature, f"{prefix}_{side}") for prefix in FORELIMB_PREFIXES]
        for side in ("L", "R")
    }
    weights = {side: side_weight_summary(mesh, side) for side in ("L", "R")}
    report = {
        "file": bpy.data.filepath,
        "armature": armature.name,
        "mesh": mesh.name,
        "objects": object_records(),
        "forelimb_bones": forelimb_bones,
        "weights": weights,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_FORELIMB_RIG_WEIGHTS_BEGIN")
    print(
        json.dumps(
            {
                "output": str(output),
                "objects": len(report["objects"]),
                "left_vertices": weights["L"]["forelimb_vertex_count"],
                "right_vertices": weights["R"]["forelimb_vertex_count"],
                "left_scapula_wrist_blend_fraction": weights["L"]["scapula_wrist_blend_fraction"],
                "right_scapula_wrist_blend_fraction": weights["R"]["scapula_wrist_blend_fraction"],
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_FORELIMB_RIG_WEIGHTS_END")


if __name__ == "__main__":
    main()

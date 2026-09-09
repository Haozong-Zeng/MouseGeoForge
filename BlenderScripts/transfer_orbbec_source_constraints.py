"""Transfer source-rig pose constraints from the orbbec controller file.

This is intentionally conservative: it first compares the donor and target
rest-bone endpoints/rolls, then writes a new blend instead of modifying the
input target in place.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Quaternion


SKIP_CONSTRAINT_PROPS = {
    "rna_type",
    "type",
    "name",
    "is_valid",
    "error_location",
    "error_rotation",
}

POSE_BONE_PROPS = (
    "rotation_mode",
    "lock_location",
    "lock_rotation",
    "lock_rotation_w",
    "lock_scale",
    "ik_stretch",
    "ik_min_x",
    "ik_max_x",
    "ik_min_y",
    "ik_max_y",
    "ik_min_z",
    "ik_max_z",
    "ik_stiffness_x",
    "ik_stiffness_y",
    "ik_stiffness_z",
    "use_ik_limit_x",
    "use_ik_limit_y",
    "use_ik_limit_z",
    "use_ik_rotation_control",
    "ik_rotation_weight",
    "use_ik_linear_control",
    "ik_linear_weight",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-armature", default=None)
    parser.add_argument("--target-armature", default=None)
    parser.add_argument("--max-endpoint-delta", type=float, default=1.0e-4)
    parser.add_argument("--max-roll-delta", type=float, default=1.0e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args(args_after_dash())


def flat_matrix(matrix: Matrix, digits: int | None = None) -> list[float]:
    values = [float(matrix[row][col]) for row in range(4) for col in range(4)]
    if digits is not None:
        values = [round(value, digits) for value in values]
    return values


def vec3(value, digits: int | None = None) -> list[float]:
    values = [float(value.x), float(value.y), float(value.z)]
    if digits is not None:
        values = [round(item, digits) for item in values]
    return values


def quat4(value: Quaternion, digits: int | None = None) -> list[float]:
    value = value.normalized()
    values = [float(value.w), float(value.x), float(value.y), float(value.z)]
    if digits is not None:
        values = [round(item, digits) for item in values]
    return values


def quaternion_angle_delta(a: list[float], b: list[float]) -> float:
    qa = Quaternion((a[0], a[1], a[2], a[3])).normalized()
    qb = Quaternion((b[0], b[1], b[2], b[3])).normalized()
    return float(qa.rotation_difference(qb).angle)


def constraint_count(obj) -> int:
    return sum(len(pb.constraints) for pb in obj.pose.bones) if obj.type == "ARMATURE" else 0


def armatures_with_bones() -> list:
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "ARMATURE" and len(getattr(obj.data, "bones", [])) > 0
    ]


def choose_source_armature(name: str | None):
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "ARMATURE":
            raise RuntimeError(f"Missing source armature {name!r}")
        return obj
    candidates = armatures_with_bones()
    if not candidates:
        raise RuntimeError("No source armature with bones found")
    return max(candidates, key=lambda obj: (constraint_count(obj), len(obj.data.bones)))


def choose_target_armature(name: str | None, source_bone_names: set[str]):
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "ARMATURE":
            raise RuntimeError(f"Missing target armature {name!r}")
        return obj
    candidates = armatures_with_bones()
    if not candidates:
        raise RuntimeError("No target armature with bones found")

    def score(obj):
        target_names = {bone.name for bone in obj.data.bones}
        common = len(source_bone_names & target_names)
        missing = len(source_bone_names - target_names)
        extra = len(target_names - source_bone_names)
        return (common, -missing, -extra, -len(obj.data.bones))

    return max(candidates, key=score)


def serializable_sequence(value) -> list[Any] | None:
    try:
        items = list(value)
    except TypeError:
        return None
    out = []
    for item in items:
        if isinstance(item, (bool, int, float, str)):
            out.append(item)
        else:
            return None
    return out


def serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Matrix):
        return {"__matrix4__": flat_matrix(value)}
    if isinstance(value, bpy.types.Object):
        return {"__object__": value.name}
    if isinstance(value, bpy.types.ID):
        return {"__id__": value.name, "__id_type__": value.bl_rna.identifier}
    sequence = serializable_sequence(value)
    if sequence is not None:
        return sequence
    return {"__skipped_repr__": repr(value)}


def collect_constraint(constraint) -> dict:
    props = {}
    skipped = []
    for prop in constraint.bl_rna.properties:
        identifier = prop.identifier
        if identifier in SKIP_CONSTRAINT_PROPS or prop.is_readonly:
            continue
        if prop.type == "COLLECTION":
            skipped.append(identifier)
            continue
        try:
            props[identifier] = serialize_value(getattr(constraint, identifier))
        except Exception as exc:  # Blender RNA can reject some transient props.
            skipped.append(f"{identifier}: {exc}")
    return {
        "name": constraint.name,
        "type": constraint.type,
        "props": props,
        "skipped_props": skipped,
    }


def collect_bone_data(armature) -> dict[str, dict]:
    out = {}
    for bone in armature.data.bones:
        out[bone.name] = {
            "name": bone.name,
            "parent": bone.parent.name if bone.parent else None,
            "head_local": vec3(bone.head_local),
            "tail_local": vec3(bone.tail_local),
            "length": float(bone.length),
            "matrix_local": flat_matrix(bone.matrix_local),
            "basis_quaternion": quat4(bone.matrix_local.to_quaternion()),
            "use_connect": bool(bone.use_connect),
            "use_deform": bool(bone.use_deform),
        }
    return out


def collect_pose_settings(pose_bone) -> dict:
    settings = {}
    for name in POSE_BONE_PROPS:
        if not hasattr(pose_bone, name):
            continue
        try:
            settings[name] = serialize_value(getattr(pose_bone, name))
        except Exception:
            pass
    return settings


def collect_source_payload(armature) -> dict:
    bones = collect_bone_data(armature)
    constraints = {}
    pose_settings = {}
    target_objects: set[str] = set()
    constraint_types: dict[str, int] = {}

    for pose_bone in armature.pose.bones:
        if pose_bone.constraints:
            rows = [collect_constraint(constraint) for constraint in pose_bone.constraints]
            constraints[pose_bone.name] = rows
            for row in rows:
                constraint_types[row["type"]] = constraint_types.get(row["type"], 0) + 1
                for value in row["props"].values():
                    if isinstance(value, dict) and "__object__" in value:
                        target_objects.add(value["__object__"])
        pose_settings[pose_bone.name] = collect_pose_settings(pose_bone)

    target_objects.discard(armature.name)
    return {
        "armature": armature.name,
        "bone_count": len(bones),
        "constraint_count": sum(len(rows) for rows in constraints.values()),
        "constraint_bone_count": len(constraints),
        "constraint_types": dict(sorted(constraint_types.items())),
        "bones": bones,
        "constraints": constraints,
        "pose_settings": pose_settings,
        "target_objects": sorted(target_objects),
    }


def compare_rest_pose(source_bones: dict[str, dict], target_armature) -> dict:
    target_bones = collect_bone_data(target_armature)
    source_names = set(source_bones)
    target_names = set(target_bones)
    common = sorted(source_names & target_names)
    rows = []
    parent_mismatches = []
    use_connect_mismatches = []
    for name in common:
        src = source_bones[name]
        dst = target_bones[name]
        head_delta = math.dist(src["head_local"], dst["head_local"])
        tail_delta = math.dist(src["tail_local"], dst["tail_local"])
        endpoint_delta = max(head_delta, tail_delta)
        roll_delta = quaternion_angle_delta(src["basis_quaternion"], dst["basis_quaternion"])
        matrix_delta = max(
            abs(float(a) - float(b)) for a, b in zip(src["matrix_local"], dst["matrix_local"])
        )
        length_delta = abs(src["length"] - dst["length"])
        row = {
            "bone": name,
            "head_delta": head_delta,
            "tail_delta": tail_delta,
            "endpoint_delta": endpoint_delta,
            "roll_delta": roll_delta,
            "length_delta": length_delta,
            "matrix_delta": matrix_delta,
        }
        rows.append(row)
        if src["parent"] != dst["parent"]:
            parent_mismatches.append({"bone": name, "source": src["parent"], "target": dst["parent"]})
        if src["use_connect"] != dst["use_connect"]:
            use_connect_mismatches.append(
                {"bone": name, "source": src["use_connect"], "target": dst["use_connect"]}
            )

    max_endpoint = max((row["endpoint_delta"] for row in rows), default=None)
    max_roll = max((row["roll_delta"] for row in rows), default=None)
    max_matrix = max((row["matrix_delta"] for row in rows), default=None)
    max_length = max((row["length_delta"] for row in rows), default=None)
    return {
        "source_bone_count": len(source_bones),
        "target_bone_count": len(target_bones),
        "common_bone_count": len(common),
        "missing_in_target": sorted(source_names - target_names),
        "extra_in_target": sorted(target_names - source_names),
        "max_endpoint_delta": max_endpoint,
        "max_roll_delta": max_roll,
        "max_matrix_delta": max_matrix,
        "max_length_delta": max_length,
        "top_endpoint_deltas": sorted(rows, key=lambda item: item["endpoint_delta"], reverse=True)[:20],
        "top_roll_deltas": sorted(rows, key=lambda item: item["roll_delta"], reverse=True)[:20],
        "parent_mismatches": parent_mismatches,
        "use_connect_mismatches": use_connect_mismatches,
    }


def matrix_from_flat(values: list[float]) -> Matrix:
    return Matrix([values[0:4], values[4:8], values[8:12], values[12:16]])


def json_default(value):
    if isinstance(value, bpy.types.ID):
        return value.name
    if isinstance(value, Matrix):
        return flat_matrix(value)
    return repr(value)


def resolve_value(value: Any, target_armature, source_armature_name: str) -> Any:
    if isinstance(value, dict):
        if "__matrix4__" in value:
            return matrix_from_flat(value["__matrix4__"])
        if "__object__" in value:
            object_name = value["__object__"]
            if object_name == source_armature_name:
                return target_armature
            return bpy.data.objects.get(object_name)
        if "__id__" in value:
            return None
    return value


def append_missing_target_objects(source_path: Path, object_names: list[str]) -> dict:
    missing = [name for name in object_names if name not in bpy.data.objects]
    found = []
    unavailable = []
    if not missing:
        return {"requested": object_names, "appended": [], "unavailable": []}

    with bpy.data.libraries.load(str(source_path.resolve()), link=False) as (data_from, data_to):
        available = set(data_from.objects)
        found = [name for name in missing if name in available]
        unavailable = [name for name in missing if name not in available]
        data_to.objects = found

    for obj in data_to.objects:
        if obj is not None and not obj.users_collection:
            bpy.context.scene.collection.objects.link(obj)

    return {"requested": object_names, "appended": found, "unavailable": unavailable}


def set_pose_settings(target_armature, source_payload: dict) -> dict:
    copied = 0
    failed = []
    for bone_name, settings in source_payload["pose_settings"].items():
        pose_bone = target_armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        for attr, value in settings.items():
            try:
                setattr(pose_bone, attr, resolve_value(value, target_armature, source_payload["armature"]))
                copied += 1
            except Exception as exc:
                failed.append({"bone": bone_name, "property": attr, "error": str(exc)})
    return {"copied": copied, "failed": failed[:100], "failed_count": len(failed)}


def transfer_constraints(target_armature, source_payload: dict, keep_existing: bool) -> dict:
    copied = []
    missing_bones = []
    failed = []
    replaced_constraint_count = 0

    for bone_name, rows in source_payload["constraints"].items():
        pose_bone = target_armature.pose.bones.get(bone_name)
        if pose_bone is None:
            missing_bones.append(bone_name)
            continue
        if not keep_existing:
            replaced_constraint_count += len(pose_bone.constraints)
            while pose_bone.constraints:
                pose_bone.constraints.remove(pose_bone.constraints[0])
        for row in rows:
            try:
                new_constraint = pose_bone.constraints.new(type=row["type"])
                new_constraint.name = row["name"]
            except Exception as exc:
                failed.append(
                    {
                        "bone": bone_name,
                        "constraint": row["name"],
                        "property": "<create>",
                        "error": str(exc),
                    }
                )
                continue

            prop_items = list(row["props"].items())
            prop_items.sort(
                key=lambda item: (
                    0
                    if item[0] in {"target", "pole_target", "space_object", "subtarget", "pole_subtarget"}
                    else 1,
                    item[0],
                )
            )
            failed_props = []
            for prop_name, encoded_value in prop_items:
                if prop_name == "name":
                    continue
                value = resolve_value(encoded_value, target_armature, source_payload["armature"])
                if isinstance(encoded_value, dict) and "__object__" in encoded_value and value is None:
                    failed_props.append({"property": prop_name, "error": "object target unavailable"})
                    continue
                try:
                    setattr(new_constraint, prop_name, value)
                except Exception as exc:
                    failed_props.append({"property": prop_name, "error": str(exc)})
            if failed_props:
                failed.append(
                    {
                        "bone": bone_name,
                        "constraint": row["name"],
                        "type": row["type"],
                        "failed_props": failed_props,
                    }
                )
            copied.append({"bone": bone_name, "name": row["name"], "type": row["type"]})

    return {
        "copied_count": len(copied),
        "copied_bone_count": len({row["bone"] for row in copied}),
        "missing_bones": missing_bones,
        "failed": failed[:200],
        "failed_count": len(failed),
        "replaced_constraint_count": replaced_constraint_count,
        "constraint_type_counts": {
            key: sum(1 for row in copied if row["type"] == key)
            for key in sorted({row["type"] for row in copied})
        },
    }


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    target_path = args.target.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    bpy.ops.wm.open_mainfile(filepath=str(source_path))
    source_armature = choose_source_armature(args.source_armature)
    source_payload = collect_source_payload(source_armature)

    bpy.ops.wm.open_mainfile(filepath=str(target_path))
    target_armature = choose_target_armature(args.target_armature, set(source_payload["bones"]))
    comparison = compare_rest_pose(source_payload["bones"], target_armature)
    consistent = (
        not comparison["missing_in_target"]
        and not comparison["parent_mismatches"]
        and (comparison["max_endpoint_delta"] is not None)
        and comparison["max_endpoint_delta"] <= args.max_endpoint_delta
        and (comparison["max_roll_delta"] is not None)
        and comparison["max_roll_delta"] <= args.max_roll_delta
    )

    appended = {"requested": [], "appended": [], "unavailable": []}
    pose_settings_report = {"copied": 0, "failed": [], "failed_count": 0}
    transfer_report = {
        "copied_count": 0,
        "copied_bone_count": 0,
        "missing_bones": [],
        "failed": [],
        "failed_count": 0,
        "replaced_constraint_count": 0,
        "constraint_type_counts": {},
    }
    saved = False
    if consistent or args.force:
        appended = append_missing_target_objects(source_path, source_payload["target_objects"])
        pose_settings_report = set_pose_settings(target_armature, source_payload)
        transfer_report = transfer_constraints(
            target_armature,
            source_payload,
            keep_existing=bool(args.keep_existing),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        saved = True

    report = {
        "source": str(source_path),
        "target": str(target_path),
        "output": str(output_path),
        "saved": saved,
        "source_armature": source_payload["armature"],
        "target_armature": target_armature.name,
        "source_bone_count": source_payload["bone_count"],
        "source_constraint_count": source_payload["constraint_count"],
        "source_constraint_bone_count": source_payload["constraint_bone_count"],
        "source_constraint_types": source_payload["constraint_types"],
        "source_target_objects": source_payload["target_objects"],
        "rest_pose_comparison": comparison,
        "thresholds": {
            "max_endpoint_delta": args.max_endpoint_delta,
            "max_roll_delta": args.max_roll_delta,
            "force": bool(args.force),
        },
        "rest_pose_consistent": consistent,
        "appended_objects": appended,
        "pose_settings": pose_settings_report,
        "transfer": transfer_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print("ORBBEC_CONSTRAINT_TRANSFER_REPORT_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))
    print("ORBBEC_CONSTRAINT_TRANSFER_REPORT_END")


if __name__ == "__main__":
    main()

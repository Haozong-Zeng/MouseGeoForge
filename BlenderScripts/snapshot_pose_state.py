"""Write deterministic pose and evaluated-mesh fingerprints for a Blender file."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import bpy


def blender_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 31, 81, 131, 181, 200])
    return parser.parse_args(argv)


def digest_values(values: list[float]) -> str:
    payload = struct.pack(f"<{len(values)}d", *values)
    return hashlib.sha256(payload).hexdigest()


def quantized_digest(values: list[float], digits: int) -> str:
    return digest_values([round(value, digits) for value in values])


def matrix_values(matrix: object) -> list[float]:
    return [float(value) for row in matrix for value in row]


def action_signature(obj: bpy.types.Object) -> dict[str, object] | None:
    if obj.animation_data is None or obj.animation_data.action is None:
        return None
    action = obj.animation_data.action
    values: list[float] = []
    paths: list[str] = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                for curve in bag.fcurves:
                    paths.append(f"{curve.data_path}[{curve.array_index}]")
                    for point in curve.keyframe_points:
                        values.extend((float(point.co[0]), float(point.co[1])))
    return {
        "name": action.name,
        "paths": paths,
        "keyframe_digest": digest_values(values),
        "keyframe_value_count": len(values),
    }


def main() -> None:
    args = blender_args()
    scene = bpy.context.scene
    armatures = sorted((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), key=lambda obj: obj.name)
    meshes = sorted((obj for obj in bpy.data.objects if obj.type == "MESH"), key=lambda obj: obj.name)
    report: dict[str, object] = {
        "file": bpy.data.filepath,
        "frames": args.frames,
        "actions": {obj.name: action_signature(obj) for obj in armatures},
        "samples": {},
    }
    for frame in args.frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        armature_rows: dict[str, object] = {}
        for obj in armatures:
            evaluated = obj.evaluated_get(depsgraph)
            values = matrix_values(evaluated.matrix_world)
            bone_names = sorted(bone.name for bone in evaluated.pose.bones)
            for name in bone_names:
                values.extend(matrix_values(evaluated.matrix_world @ evaluated.pose.bones[name].matrix))
            armature_rows[obj.name] = {
                "bone_count": len(bone_names),
                "bone_names": bone_names,
                "digest": digest_values(values),
                "digest_1e9": quantized_digest(values, 9),
                "digest_1e6": quantized_digest(values, 6),
            }
        mesh_rows: dict[str, object] = {}
        for obj in meshes:
            if obj.name != "MS_Mouse.001":
                continue
            evaluated = obj.evaluated_get(depsgraph)
            evaluated_mesh = evaluated.to_mesh()
            try:
                values = matrix_values(evaluated.matrix_world)
                for vertex in evaluated_mesh.vertices:
                    world = evaluated.matrix_world @ vertex.co
                    values.extend((float(world.x), float(world.y), float(world.z)))
                mesh_rows[obj.name] = {
                    "vertex_count": len(evaluated_mesh.vertices),
                    "polygon_count": len(evaluated_mesh.polygons),
                    "digest": digest_values(values),
                    "digest_1e9": quantized_digest(values, 9),
                    "digest_1e6": quantized_digest(values, 6),
                }
            finally:
                evaluated.to_mesh_clear()
        report["samples"][str(frame)] = {"armatures": armature_rows, "meshes": mesh_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"POSE_SNAPSHOT_SAVED {args.output}")


if __name__ == "__main__":
    main()

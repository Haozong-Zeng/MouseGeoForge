#!/usr/bin/env python3
"""Export evaluated pose-bone matrices after constraints and controller animation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armature", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args_after_dash())


def evaluated_bounds(obj: bpy.types.Object, depsgraph) -> np.ndarray:
    evaluated = obj.evaluated_get(depsgraph)
    corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    xyz = np.asarray(corners, dtype=np.float64)
    return np.stack((xyz.min(axis=0), xyz.max(axis=0)))


def main() -> None:
    args = parse_args()
    armature = bpy.data.objects[args.armature]
    mesh = bpy.data.objects[args.mesh]
    bone_names = [bone.name for bone in armature.data.bones]
    frames = np.arange(args.start, args.end + 1, dtype=np.int32)
    matrices = np.empty((len(frames), len(bone_names), 4, 4), dtype=np.float64)
    visual_basis = np.empty_like(matrices)
    bounds = np.empty((len(frames), 2, 3), dtype=np.float64)
    scene = bpy.context.scene
    for frame_index, frame in enumerate(frames):
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_armature = armature.evaluated_get(depsgraph)
        for bone_index, bone_name in enumerate(bone_names):
            pose_bone = evaluated_armature.pose.bones[bone_name]
            matrices[frame_index, bone_index] = np.asarray(pose_bone.matrix, dtype=np.float64)
            conversion_args = {
                "matrix": pose_bone.matrix,
                "matrix_local": pose_bone.bone.matrix_local,
                "invert": True,
            }
            if pose_bone.parent:
                conversion_args["parent_matrix"] = pose_bone.parent.matrix
                conversion_args["parent_matrix_local"] = pose_bone.parent.bone.matrix_local
            visual_basis[frame_index, bone_index] = np.asarray(
                pose_bone.bone.convert_local_to_pose(**conversion_args),
                dtype=np.float64,
            )
        bounds[frame_index] = evaluated_bounds(mesh, depsgraph)
        if frame == args.start or frame == args.end or frame % 25 == 0:
            print(f"exported pose frame {frame}/{args.end}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format=np.asarray("mgf_evaluated_pose_matrices_v1"),
        source_blend=np.asarray(bpy.data.filepath),
        armature=np.asarray(armature.name),
        mesh=np.asarray(mesh.name),
        frame_numbers=frames,
        bone_names=np.asarray(bone_names),
        pose_matrices=matrices,
        visual_matrix_basis=visual_basis,
        mesh_world_bounds=bounds,
    )
    audit = {
        "format": "mgf_evaluated_pose_matrices_v1",
        "source_blend": bpy.data.filepath,
        "armature": armature.name,
        "mesh": mesh.name,
        "frames": [int(frames[0]), int(frames[-1])],
        "frame_count": len(frames),
        "bone_count": len(bone_names),
        "mesh_union_min": bounds[:, 0].min(axis=0).tolist(),
        "mesh_union_max": bounds[:, 1].max(axis=0).tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

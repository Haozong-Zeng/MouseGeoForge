#!/usr/bin/env python3
"""Export evaluated pose-bone matrices from a Blender animation to NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--armature", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--visual-bake",
        action="store_true",
        help="Bake evaluated constraints to local pose channels before export.",
    )
    args = parser.parse_args(args_after_dash())

    scene = bpy.context.scene
    armature = bpy.data.objects[args.armature]
    if args.visual_bake:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        armature.hide_set(False)
        armature.hide_viewport = False
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
        bpy.ops.nla.bake(
            frame_start=args.start,
            frame_end=args.start + args.count - 1,
            step=1,
            only_selected=False,
            visual_keying=True,
            clear_constraints=True,
            clear_parents=False,
            use_current_action=False,
            clean_curves=False,
            bake_types={"POSE"},
        )
        bpy.ops.object.mode_set(mode="OBJECT")
    bone_names = [bone.name for bone in armature.pose.bones]
    rest_matrices = np.stack([
        np.asarray(armature.data.bones[name].matrix_local, dtype=np.float64)
        for name in bone_names
    ])
    frames = np.arange(args.start, args.start + args.count, dtype=np.int32)
    matrices = np.empty((len(frames), len(bone_names), 4, 4), dtype=np.float64)
    heads = np.empty((len(frames), len(bone_names), 3), dtype=np.float64)
    basis_matrices = np.empty_like(matrices)
    for frame_index, frame in enumerate(frames):
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        evaluated = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
        for bone_index, name in enumerate(bone_names):
            pose_bone = evaluated.pose.bones[name]
            matrices[frame_index, bone_index] = np.asarray(pose_bone.matrix, dtype=np.float64)
            heads[frame_index, bone_index] = np.asarray(pose_bone.head, dtype=np.float64)
            basis_matrices[frame_index, bone_index] = np.asarray(
                armature.pose.bones[name].matrix_basis, dtype=np.float64
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("mousegeoforge.evaluated_pose_sequence.v1"),
        source_blend=np.asarray(bpy.data.filepath),
        armature=np.asarray(args.armature),
        frames=frames,
        bone_names=np.asarray(bone_names),
        rest_matrices=rest_matrices,
        pose_matrices=matrices,
        basis_matrices=basis_matrices,
        pose_heads=heads,
    )
    print(f"exported {len(frames)} frames x {len(bone_names)} bones to {args.output}")


if __name__ == "__main__":
    main()

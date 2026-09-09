#!/usr/bin/env python3
"""Measure internal pose dynamics independently of whole-body placement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np


BONES = (
    "Root_M", "Spine1_M", "Spine2_M", "Spine3_M", "Spine4_M", "Neck3_M", "HeadEnd_M",
    "Hip_L", "Knee_L", "Ankle_L", "Wrist_L", "Fingers1_L",
    "Hip_R", "Knee_R", "Ankle_R", "Wrist_R", "Fingers1_R",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--armatures", required=True, help="Comma-separated object names")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args_after_dash())


def body_coordinates(armature: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    points = {
        name: np.asarray(armature.matrix_world @ armature.pose.bones[name].head, dtype=np.float64)
        for name in BONES if name in armature.pose.bones
    }
    root = points["Root_M"]
    heading = points["Neck3_M"] - root
    heading[2] = 0.0
    if np.linalg.norm(heading) < 1.0e-8:
        heading = np.asarray((1.0, 0.0, 0.0))
    else:
        heading /= np.linalg.norm(heading)
    lateral = np.asarray((-heading[1], heading[0], 0.0))
    vertical = np.asarray((0.0, 0.0, 1.0))
    rotation = np.stack((heading, lateral, vertical), axis=1)
    local = np.stack([(points[name] - root) @ rotation for name in points])
    quaternions = np.stack([
        np.asarray((armature.matrix_world @ armature.pose.bones[name].matrix).to_quaternion(), dtype=np.float64)
        for name in points
    ])
    return local, quaternions


def quaternion_angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(first * second, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def audit_armature(name: str, frames: list[int]) -> dict:
    scene = bpy.context.scene
    armature = bpy.data.objects[name]
    local_rows = []
    quaternion_rows = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        local, quaternions = body_coordinates(armature)
        local_rows.append(local)
        quaternion_rows.append(quaternions)
    local_rows = np.asarray(local_rows)
    quaternion_rows = np.asarray(quaternion_rows)
    position_steps = np.linalg.norm(np.diff(local_rows, axis=0), axis=2)
    angle_steps = np.stack([
        quaternion_angle_degrees(quaternion_rows[index], quaternion_rows[index + 1])
        for index in range(len(frames) - 1)
    ])
    centered = local_rows - np.mean(local_rows, axis=0, keepdims=True)
    measured_bones = [bone for bone in BONES if bone in armature.pose.bones]
    maximum_step_index = np.unravel_index(np.argmax(position_steps), position_steps.shape)
    signatures = {
        tuple(np.round(row.reshape(-1), 2))
        for row in local_rows
    }
    return {
        "armature": name,
        "frames": [frames[0], frames[-1]],
        "sample_count": len(frames),
        "unique_internal_pose_signatures_0p01mm": len(signatures),
        "mean_internal_joint_step_mm": float(np.mean(position_steps)),
        "p95_internal_joint_step_mm": float(np.percentile(position_steps, 95)),
        "maximum_internal_joint_step_mm": float(np.max(position_steps)),
        "maximum_internal_joint_step": {
            "frame_from": frames[maximum_step_index[0]],
            "frame_to": frames[maximum_step_index[0] + 1],
            "bone": measured_bones[maximum_step_index[1]],
            "millimeters": float(position_steps[maximum_step_index]),
        },
        "mean_world_bone_rotation_step_degrees": float(np.mean(angle_steps)),
        "p95_world_bone_rotation_step_degrees": float(np.percentile(angle_steps, 95)),
        "internal_pose_rms_variation_mm": float(np.sqrt(np.mean(centered * centered))),
    }


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    report = {
        "blend": bpy.data.filepath,
        "scene_frames": [frames[0], frames[-1]],
        "armatures": [audit_armature(name, frames) for name in args.armatures.split(",")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

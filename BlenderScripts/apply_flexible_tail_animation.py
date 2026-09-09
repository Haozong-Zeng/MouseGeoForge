#!/usr/bin/env python3
"""Replace only the distal-tail animation with collision-safe flexible motion.

Run with Blender, for example::

    blender -b input.blend --python apply_flexible_tail_animation.py -- \
      --output output.blend --report output.tail-audit.json

The existing body and limb animation is read but never keyed or rewritten.  The
tail base and body-to-tail tangent are cached first, then a geometry-only solver
drives the existing Spline IK target curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flexible_tail_solver import (  # noqa: E402
    ArenaBounds,
    EllipsoidProxy,
    TailSolverConfig,
    solve_tail_sequence,
)


TAIL_BONES = ["TailStart_M"] + [f"Tail{index}_M" for index in range(12)]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_vector(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--armature", default="DeformationSystem.001")
    parser.add_argument("--curve", default="Tail_NURBSCurve.001")
    parser.add_argument("--tail-base-bone", default="TailStart_M")
    parser.add_argument("--body-reference-bone", default="Spine2_M")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--seed", type=int, default=5706)
    parser.add_argument("--floor-z", type=float)
    parser.add_argument("--arena-min", type=parse_vector, default=(-132.0, -98.0, -240.0))
    parser.add_argument("--arena-max", type=parse_vector, default=(132.0, 98.0, -90.0))
    parser.add_argument("--body-radii", type=parse_vector, default=(20.0, 12.0, 10.0))
    parser.add_argument("--other-armature", action="append", default=[])
    parser.add_argument("--spline-type", choices=("POLY", "NURBS"), default="POLY")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(args_after_dash())


def world_bone_head(armature: bpy.types.Object, bone_name: str) -> np.ndarray:
    return np.asarray(armature.matrix_world @ armature.pose.bones[bone_name].head, dtype=np.float64)


def body_frame(base: np.ndarray, body_reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = body_reference - base
    forward[2] *= 0.35
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    lateral = np.cross(up, forward)
    if np.linalg.norm(lateral) < 1.0e-6:
        lateral = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    vertical /= max(float(np.linalg.norm(vertical)), 1.0e-9)
    rotation = np.column_stack((forward, lateral, vertical))
    return forward, rotation


def tail_lengths_world(armature: bpy.types.Object) -> np.ndarray:
    lengths = []
    for name in TAIL_BONES:
        bone = armature.pose.bones[name]
        head = armature.matrix_world @ bone.head
        tail = armature.matrix_world @ bone.tail
        lengths.append((tail - head).length)
    return np.asarray(lengths, dtype=np.float64)


def cache_conditioning(
    armature: bpy.types.Object,
    frames: list[int],
    tail_base_bone: str,
    body_reference_bone: str,
    body_radii: tuple[float, float, float],
    other_armatures: list[bpy.types.Object],
) -> tuple[np.ndarray, np.ndarray, list[tuple[EllipsoidProxy, ...]]]:
    bases = []
    tangents = []
    proxies_by_frame = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        base = world_bone_head(armature, tail_base_bone)
        reference = world_bone_head(armature, body_reference_bone)
        forward, rotation = body_frame(base, reference)
        bases.append(base)
        tangents.append(-forward)
        proxies = [
            EllipsoidProxy(
                center=tuple(base + forward * (body_radii[0] * 0.72) + rotation[:, 2] * 1.0),
                radii=body_radii,
                name=f"{armature.name}:posterior_body",
                margin=0.4,
                rotation=tuple(rotation.reshape(-1)),
            )
        ]
        for other in other_armatures:
            other_base = world_bone_head(other, tail_base_bone)
            other_reference = world_bone_head(other, body_reference_bone)
            other_forward, other_rotation = body_frame(other_base, other_reference)
            proxies.append(
                EllipsoidProxy(
                    center=tuple(other_base + other_forward * (body_radii[0] * 0.72)),
                    radii=body_radii,
                    name=f"{other.name}:body",
                    margin=1.2,
                    rotation=tuple(other_rotation.reshape(-1)),
                )
            )
        proxies_by_frame.append(tuple(proxies))
    return np.asarray(bases), np.asarray(tangents), proxies_by_frame


def non_tail_pose_digest(armature: bpy.types.Object, frames: list[int]) -> str:
    digest = hashlib.sha256()
    names = sorted(name for name in armature.pose.bones.keys() if name not in TAIL_BONES)
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        digest.update(np.asarray([frame], dtype=np.int32).tobytes())
        for name in names:
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(armature.pose.bones[name].matrix_basis, dtype=np.float64).tobytes())
    return digest.hexdigest()


def rebuild_tail_spline(curve: bpy.types.Object, point_count: int, spline_type: str) -> None:
    curve.data.animation_data_clear()
    curve.data.splines.clear()
    spline = curve.data.splines.new(spline_type)
    spline.points.add(point_count - 1)
    if spline_type == "NURBS":
        spline.order_u = min(4, point_count)
        spline.use_endpoint_u = True
    spline.use_cyclic_u = False


def action_fcurves(action) -> list:
    """Return F-curves for both legacy and Blender 4.4+ layered Actions."""
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    fcurves = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                fcurves.extend(channelbag.fcurves)
    return fcurves


def key_tail_curve(curve: bpy.types.Object, frames: list[int], chains: np.ndarray) -> None:
    scene = bpy.context.scene
    original_frame = scene.frame_current
    spline = curve.data.splines[0]
    for frame, chain in zip(frames, chains, strict=True):
        # The curve is parented to the animated mouse placement.  Its world
        # matrix therefore changes with the root trajectory and must be
        # inverted at the destination frame, not once at frame_start.
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        inverse = curve.matrix_world.inverted()
        for point, world in zip(spline.points, chain, strict=True):
            local = inverse @ Vector(world)
            point.co = (*local, 1.0)
            point.keyframe_insert(data_path="co", frame=frame)
    scene.frame_set(original_frame)
    bpy.context.view_layer.update()
    action = curve.data.animation_data.action if curve.data.animation_data else None
    if action is not None:
        for fcurve in action_fcurves(action):
            for key in fcurve.keyframe_points:
                key.interpolation = "LINEAR"


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    armature = bpy.data.objects[args.armature]
    curve = bpy.data.objects[args.curve]
    other_armatures = [bpy.data.objects[name] for name in args.other_armature]
    frame_start = args.frame_start if args.frame_start is not None else scene.frame_start
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end
    frames = list(range(frame_start, frame_end + 1))
    floor_z = args.floor_z
    if floor_z is None:
        floor_z = float(scene.get("MGF_ground_contact_floor_z_mm", args.arena_min[2]))

    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    lengths = tail_lengths_world(armature)
    bases, tangents, proxies = cache_conditioning(
        armature,
        frames,
        args.tail_base_bone,
        args.body_reference_bone,
        args.body_radii,
        other_armatures,
    )
    verification_frames = sorted({frame_start, (frame_start + frame_end) // 2, frame_end})
    non_tail_before = non_tail_pose_digest(armature, verification_frames)
    config = TailSolverConfig(body_collision_start_point=3)
    arena_min = list(args.arena_min)
    arena_max = list(args.arena_max)
    arena_min[2] = floor_z
    arena = ArenaBounds(tuple(arena_min), tuple(arena_max))
    chains, audits = solve_tail_sequence(
        bases,
        tangents,
        lengths,
        seed=args.seed,
        floor_z=floor_z,
        arena=arena,
        collision_proxies_by_frame=proxies,
        config=config,
    )

    if not args.dry_run:
        rebuild_tail_spline(curve, chains.shape[1], args.spline_type)
        key_tail_curve(curve, frames, chains)
        bpy.context.view_layer.update()
    non_tail_after = non_tail_pose_digest(armature, verification_frames)
    posture_unchanged = non_tail_before == non_tail_after

    report = {
        "schema": "mousegeoforge.flexible_tail.audit.v1",
        "source_blend": bpy.data.filepath,
        "output_blend": str(args.output),
        "armature": armature.name,
        "curve": curve.name,
        "frames": [frame_start, frame_end],
        "seed": args.seed,
        "source_pose_dependency": {
            "hard": ["tail_base_world"],
            "weak": ["body_to_tail_root_tangent"],
            "ignored": ["distal_tail_source_pose"],
        },
        "tail_segment_lengths_mm": lengths.tolist(),
        "tail_total_length_mm": float(lengths.sum()),
        "floor_z_mm": floor_z,
        "arena": {"minimum": arena.minimum, "maximum": arena.maximum},
        "spline_type": args.spline_type,
        "accepted_frames": sum(audit.accepted for audit in audits),
        "rejected_frames": sum(not audit.accepted for audit in audits),
        "maximum_residual_penetration_mm": max(audit.maximum_residual_penetration for audit in audits),
        "maximum_segment_length_error_mm": max(audit.length_error_max for audit in audits),
        "correction_counts": {
            "floor": sum(audit.floor_pushes for audit in audits),
            "wall": sum(audit.wall_pushes for audit in audits),
            "body": sum(audit.body_pushes for audit in audits),
        },
        "non_tail_pose_digest_before": non_tail_before,
        "non_tail_pose_digest_after": non_tail_after,
        "non_tail_posture_unchanged": posture_unchanged,
        "dry_run": args.dry_run,
        "frame_audits": [audit.to_dict() for audit in audits],
    }
    report_path = args.report or args.output.with_suffix(".tail-audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not all(audit.accepted for audit in audits):
        raise RuntimeError(f"unresolved tail penetration; see {report_path}")
    if not posture_unchanged:
        raise RuntimeError("non-tail pose digest changed")
    if not args.dry_run:
        scene["MGF_flexible_tail_profile"] = "tail-base-conditioned-collision-corrected-v1"
        scene["MGF_flexible_tail_seed"] = args.seed
        scene["MGF_flexible_tail_audit"] = str(report_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(json.dumps({key: report[key] for key in (
        "accepted_frames",
        "rejected_frames",
        "maximum_residual_penetration_mm",
        "maximum_segment_length_error_mm",
        "non_tail_posture_unchanged",
    )}, indent=2))


if __name__ == "__main__":
    main()

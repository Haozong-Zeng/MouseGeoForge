#!/usr/bin/env python3
"""Build a donor-independent black/brown LightningPose dual-mouse scene.

Only the rig's static matrix basis, bone lengths, constraints, and left/right
semantics are reused.  No animated spine, wrist, ankle, or body-height channel
is copied from WhiteFur Controller-MoSeq or any other video.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_lightningpose_two_color_dual_mouse_pilot import (  # noqa: E402
    SOURCE_CONTROL_COLLECTION,
    SOURCE_MOUSE_COLLECTION,
    SOURCE_ROOT,
    SOURCE_TARGET_COLLECTION,
    append_brown_material,
    assign_brown_fur,
    clone_mouse_system,
    set_all_interpolation,
)
from apply_flexible_tail_animation import (  # noqa: E402
    body_frame,
    key_tail_curve,
    rebuild_tail_spline,
)
from flexible_tail_solver import (  # noqa: E402
    ArenaBounds,
    EllipsoidProxy,
    TailSolverConfig,
    solve_tail_sequence,
)
from procedural_dual_mouse_motion import MotionConfig, synthesize  # noqa: E402


CONTROL_SOURCES = {
    "root": "Root_M.001",
    "spine2": "Spine2_M.001",
    "wrist_l": "Wrist_L.001",
    "wrist_r": "Wrist_R.001",
    "ankle_l": "Ankle_L.001",
    "ankle_r": "Ankle_R.001",
    "knee_pole_l": "MGF_FT_KneePole_L",
    "knee_pole_r": "MGF_FT_KneePole_R",
}
SPINE_HEIGHTS_ABOVE_FLOOR_MM = (18.0, 23.0, 24.0, 21.0, 18.0)
FOOT_CONTROL_HEIGHTS_ABOVE_FLOOR_MM = {
    "wrist_l": 4.5,
    "wrist_r": 4.5,
    "ankle_l": 3.0,
    "ankle_r": 3.0,
}
TAIL_BONES = ["TailStart_M"] + [f"Tail{index}_M" for index in range(12)]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--brown-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--body-height-report", type=Path)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument(
        "--discrete-pose-frames",
        action="store_true",
        help="Use each plan frame as a real labeled pose and disable body smoothing.",
    )
    parser.add_argument(
        "--tail-qa-only-frames",
        help="Comma-separated scene frames to solve for candidate QA; do not save a blend.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-start", type=int)
    parser.add_argument("--preflight-end", type=int)
    parser.add_argument("--brown-source-material", default="MGF_BrownFur_KeypointMoseq")
    parser.add_argument("--seed", type=int, default=5706)
    parser.add_argument(
        "--minimum-body-axis-clearance-mm",
        type=float,
        default=36.0,
        help="Pre-mesh-audit centerline clearance for independent social poses.",
    )
    return parser.parse_args(args_after_dash())


def retime_plan(plan: dict, frame_count: int | None) -> tuple[dict, dict]:
    """Linearly retime sparse conditioning frames while preserving their order."""
    original_start, original_end = (int(value) for value in plan["scene_frame_range"])
    original_count = original_end - original_start + 1
    if frame_count is None or frame_count == original_count:
        return plan, {
            "original_frame_range": [original_start, original_end],
            "output_frame_range": [original_start, original_end],
            "scale": 1.0,
            "keyframes": [int(row["scene_frame"]) for row in plan["keyframes"]],
        }
    if frame_count < 2:
        raise ValueError("--frame-count must be at least 2")
    output = copy.deepcopy(plan)
    output_start = 1
    output_end = frame_count
    scale = (output_end - output_start) / max(original_end - original_start, 1)
    previous = output_start - 1
    remapped = []
    for keyframe in output["keyframes"]:
        source = int(keyframe["scene_frame"])
        target = int(round(output_start + (source - original_start) * scale))
        target = max(previous + 1, min(target, output_end))
        keyframe["scene_frame"] = target
        previous = target
        remapped.append(target)
    if remapped[-1] != output_end:
        output["keyframes"][-1]["scene_frame"] = output_end
        remapped[-1] = output_end
    output["scene_frame_range"] = [output_start, output_end]
    return output, {
        "original_frame_range": [original_start, original_end],
        "output_frame_range": [output_start, output_end],
        "scale": scale,
        "keyframes": remapped,
        "policy": "linear sparse-keyframe retime; trajectory geometry preserved",
    }


def attach_depth_body_heights(
    synthesized: dict,
    plan: dict,
    report_path: Path,
    *,
    smooth_dense: bool = True,
) -> dict:
    """Map conservative depth residuals onto the validated neutral rig pose.

    The VGGT report estimates anatomical surface height.  Rig controls have a
    different zero/offset, so absolute values must not be copied directly.
    Instead, report-neutral residuals are added to the existing neutral control
    heights.  This preserves rig calibration and floor contact.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    controls = tuple(report["spine_controls"])
    if controls != ("tail_base", "rear_mid", "body_mid", "neck", "nose"):
        raise RuntimeError(f"Unexpected depth-height controls: {controls}")
    report_source_frames = np.asarray(
        [row["source_image_frame"] for row in report["keyframes"]], dtype=int
    )
    plan_source_frames = np.asarray(
        [row["source_image_frame"] for row in plan["keyframes"]], dtype=int
    )
    report_frames = np.asarray([row["scene_frame"] for row in plan["keyframes"]], dtype=np.float64)
    if not np.array_equal(report_source_frames, plan_source_frames):
        raise RuntimeError("Depth-height report source frames do not match the animation plan")
    neutral_surface = np.asarray(report["neutral_height_mm"], dtype=np.float64)
    neutral_rig = np.asarray(SPINE_HEIGHTS_ABOVE_FLOOR_MM, dtype=np.float64)
    dense_frames = np.asarray(synthesized["frames"], dtype=np.float64)
    summaries = []
    for instance_index, motion in enumerate(synthesized["motions"]):
        identity = plan["keyframes"][0]["instances"][instance_index]["source_identity"]
        sparse_surface = np.asarray([
            row["instances"][identity]["height_mm"] for row in report["keyframes"]
        ], dtype=np.float64)
        sparse_rig = neutral_rig + np.clip(sparse_surface - neutral_surface, -4.0, 4.0)
        dense = np.stack([
            np.interp(dense_frames, report_frames, sparse_rig[:, column])
            for column in range(sparse_rig.shape[1])
        ], axis=1)
        if smooth_dense:
            # One small zero-phase binomial pass avoids vertical kinks for a
            # sparse trajectory.  Discrete dataset frames skip this entirely.
            padded = np.pad(dense, ((2, 2), (0, 0)), mode="edge")
            dense = (
                padded[:-4] + 4.0 * padded[1:-3] + 6.0 * padded[2:-2]
                + 4.0 * padded[3:-1] + padded[4:]
            ) / 16.0
        motion["spine_height_mm"] = dense
        summaries.append({
            "identity": identity,
            "sparse_surface_height_mm": sparse_surface.tolist(),
            "sparse_rig_control_height_mm": sparse_rig.tolist(),
            "dense_minimum_mm": np.min(dense, axis=0).tolist(),
            "dense_maximum_mm": np.max(dense, axis=0).tolist(),
        })
    return {
        "report": str(report_path),
        "schema": report.get("schema"),
        "mapping": "validated neutral rig height + clipped VGGT anatomical residual",
        "residual_limit_mm": 4.0,
        "dense_temporal_smoothing": smooth_dense,
        "ground_contact_changed": False,
        "instances": summaries,
    }


def set_world_location(obj: bpy.types.Object, world: Vector, frame: int) -> None:
    if obj.parent is None:
        obj.location = world - obj.delta_location
    else:
        local = obj.parent.matrix_world.inverted() @ world
        obj.location = local - obj.delta_location
    obj.keyframe_insert(data_path="location", frame=frame)


def set_curve_point_world(
    curve: bpy.types.Object,
    point_index: int,
    world: Vector,
    frame: int,
) -> None:
    point = curve.data.splines[0].points[point_index]
    local = curve.matrix_world.inverted() @ world
    point.co = (*local, 1.0)
    point.keyframe_insert(data_path="co", frame=frame)


def system_controls(system: dict) -> dict[str, bpy.types.Object]:
    controls = {}
    for key, source_name in CONTROL_SOURCES.items():
        source = bpy.data.objects[source_name]
        controls[key] = system["mapping"][source]
        controls[key].animation_data_clear()
    system["spine_curve"].data.animation_data_clear()
    system["tail_curve"].data.animation_data_clear()
    return controls


def configure_ik(system: dict, controls: dict[str, bpy.types.Object]) -> dict:
    armature = system["armature"]
    report = {}
    pole_angles = {"L": 3.0 * math.pi / 4.0, "R": math.pi}
    for bone_name in ("Wrist_L", "Wrist_R", "Ankle_L", "Ankle_R"):
        constraints = [constraint for constraint in armature.pose.bones[bone_name].constraints if constraint.type == "IK"]
        if not constraints:
            raise RuntimeError(f"{armature.name}:{bone_name} has no IK constraint")
        constraint = constraints[0]
        constraint.mute = False
        constraint.influence = 1.0
        constraint.use_stretch = False
        if bone_name.startswith("Ankle_"):
            side = bone_name[-1]
            constraint.chain_count = 3
            constraint.pole_target = controls[f"knee_pole_{side.lower()}"]
            constraint.pole_subtarget = ""
            constraint.pole_angle = pole_angles[side]
        else:
            constraint.chain_count = 2
        report[bone_name] = {
            "chain_count": constraint.chain_count,
            "target": constraint.target.name if constraint.target else None,
            "pole_target": constraint.pole_target.name if constraint.pole_target else None,
            "stretch": constraint.use_stretch,
        }
    return report


def key_motion(
    system: dict,
    controls: dict[str, bpy.types.Object],
    motion: dict,
    frames: np.ndarray,
    floor_z: float,
) -> None:
    spine_curve = system["spine_curve"]
    if len(spine_curve.data.splines[0].points) != 5:
        raise RuntimeError(f"Expected five spine controls on {spine_curve.name}")
    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        spine_xy = motion["spine_xy"][index]
        heights = motion.get("spine_height_mm")
        frame_heights = heights[index] if heights is not None else SPINE_HEIGHTS_ABOVE_FLOOR_MM
        spine_world = [
            Vector((float(point[0]), float(point[1]), floor_z + float(frame_heights[point_index])))
            for point_index, point in enumerate(spine_xy)
        ]
        for point_index, world in enumerate(spine_world):
            set_curve_point_world(spine_curve, point_index, world, frame)
        set_world_location(controls["root"], spine_world[0], frame)
        set_world_location(controls["spine2"], spine_world[1], frame)

        heading = Vector((float(motion["heading_xy"][index, 0]), float(motion["heading_xy"][index, 1]), 0.0))
        lateral = Vector((-heading.y, heading.x, 0.0))
        root_world = spine_world[0]
        for side, side_sign in (("l", 1.0), ("r", -1.0)):
            pole = root_world + heading * 18.0 + lateral * side_sign * 15.0
            pole.z = floor_z + 17.0
            set_world_location(controls[f"knee_pole_{side}"], pole, frame)

        for limb in ("wrist_l", "wrist_r", "ankle_l", "ankle_r"):
            xy = motion["feet_xy"][limb][index]
            lift = float(motion["foot_lift_mm"][limb][index])
            control_height = FOOT_CONTROL_HEIGHTS_ABOVE_FLOOR_MM[limb]
            # Small compensation for the depth-driven trunk residual.  It
            # keeps the fore digits inside the established <=3 mm support
            # band without changing their XY gait or swing lift.
            if motion.get("spine_height_mm") is not None and limb.startswith("wrist"):
                control_height -= 1.0
            world = Vector((
                float(xy[0]),
                float(xy[1]),
                floor_z + control_height + lift,
            ))
            set_world_location(controls[limb], world, frame)


def hide_source_and_viewport_fur(systems: list[dict]) -> list[str]:
    for collection_name in (SOURCE_MOUSE_COLLECTION, SOURCE_TARGET_COLLECTION, SOURCE_CONTROL_COLLECTION):
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            collection.hide_render = True
            collection.hide_viewport = True
    source_root = bpy.data.objects.get(SOURCE_ROOT)
    if source_root is not None:
        source_root.hide_render = True
        source_root.hide_viewport = True

    hidden = []
    for system in systems:
        for obj in system["mapping"].values():
            if obj.type == "CURVES":
                obj.hide_viewport = True
                obj.hide_render = False
                hidden.append(obj.name)
            elif obj.name.endswith(("KneePole_L", "KneePole_R")):
                obj.hide_render = True
    return hidden


def world_head(armature: bpy.types.Object, bone_name: str) -> Vector:
    return armature.matrix_world @ armature.pose.bones[bone_name].head


def evaluated_audit(
    systems: list[dict],
    frames: np.ndarray,
    floor_z: float,
    *,
    stop_on_same_side: bool = False,
) -> dict:
    instance_reports = []
    minimum_axis_distance = float("inf")
    same_side = 0
    maximum_root_height_deviation = 0.0
    rows = []
    for frame_value in frames:
        frame = int(frame_value)
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        axes = []
        frame_row = {"frame": frame, "instances": []}
        for system in systems:
            armature = system["armature"]
            root = world_head(armature, "Root_M")
            head = world_head(armature, "HeadEnd_M")
            forward = head - root
            forward.z = 0.0
            if forward.length < 1.0e-6:
                forward = Vector((0.0, 1.0, 0.0))
            else:
                forward.normalize()
            lateral = Vector((-forward.y, forward.x, 0.0))
            axes.append((np.asarray(root.xy), np.asarray(head.xy)))
            limb_row = {}
            for stem, side_sign in (
                ("Wrist_L", 1.0), ("Ankle_L", 1.0),
                ("Wrist_R", -1.0), ("Ankle_R", -1.0),
            ):
                point = world_head(armature, stem)
                signed = (point - root).dot(lateral) * side_sign
                if signed <= 0.0:
                    same_side += 1
                limb_row[stem] = {
                    "signed_lateral_mm": float(signed),
                    "height_above_floor_mm": float(point.z - floor_z),
                }
            maximum_root_height_deviation = max(
                maximum_root_height_deviation,
                abs(float(root.z - floor_z) - SPINE_HEIGHTS_ABOVE_FLOOR_MM[0]),
            )
            frame_row["instances"].append({
                "label": system["label"],
                "root_xyz_mm": list(root),
                "head_xyz_mm": list(head),
                "limbs": limb_row,
            })
        # Dense motion synthesis already solves finite body-axis distance.  The
        # evaluated rig audit uses a conservative five-point sample per axis.
        samples = [
            np.stack([start * (1.0 - alpha) + end * alpha for alpha in np.linspace(0.0, 1.0, 21)])
            for start, end in axes
        ]
        distance = float(np.min(np.linalg.norm(samples[0][:, None, :] - samples[1][None, :, :], axis=2)))
        minimum_axis_distance = min(minimum_axis_distance, distance)
        rows.append(frame_row)
        if stop_on_same_side and same_side:
            break
    for system in systems:
        instance_reports.append({
            "label": system["label"],
            "armature": system["armature"].name,
            "mesh": system["mesh"].name,
        })
    return {
        "instances": instance_reports,
        "minimum_evaluated_body_axis_distance_mm": minimum_axis_distance,
        "same_side_limb_frames": same_side,
        "maximum_root_height_deviation_mm": maximum_root_height_deviation,
        "rows": rows,
    }


def body_collision_proxies(armature: bpy.types.Object, *, other: bool) -> tuple[EllipsoidProxy, ...]:
    base = np.asarray(world_head(armature, "Root_M"), dtype=np.float64)
    head = np.asarray(world_head(armature, "HeadEnd_M"), dtype=np.float64)
    forward, rotation = body_frame(base, head)
    up = rotation[:, 2]
    if other:
        definitions = (
            (10.0, (22.0, 15.5, 12.0), 1.2, "rear"),
            (38.0, (25.0, 15.0, 12.0), 1.2, "front"),
        )
    else:
        definitions = (
            (11.0, (20.0, 13.0, 10.5), 0.5, "rear"),
            (37.0, (23.0, 13.0, 10.5), 0.5, "front"),
        )
    return tuple(
        EllipsoidProxy(
            center=tuple(base + forward * offset + up),
            radii=radii,
            name=f"{armature.name}:{token}",
            margin=margin,
            rotation=tuple(rotation.reshape(-1)),
        )
        for offset, radii, margin, token in definitions
    )


def solve_system_tail_strict(
    system: dict,
    other: dict,
    frames: list[int],
    seed: int,
    floor_z: float,
    arena: ArenaBounds,
    temporal_ar: float = 0.97,
    independent_frames: bool = False,
    seed_indices: list[int] | None = None,
) -> dict:
    armature = system["armature"]
    other_armature = other["armature"]
    bases = []
    tangents = []
    proxies = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        base = np.asarray(world_head(armature, "Root_M"), dtype=np.float64)
        head = np.asarray(world_head(armature, "HeadEnd_M"), dtype=np.float64)
        forward, _rotation = body_frame(base, head)
        bases.append(base)
        tangents.append(-forward)
        proxies.append(
            body_collision_proxies(armature, other=False)
            + body_collision_proxies(other_armature, other=True)
        )
    bpy.context.scene.frame_set(frames[0])
    lengths = np.asarray([
        (
            (armature.matrix_world @ armature.data.bones[name].tail_local)
            - (armature.matrix_world @ armature.data.bones[name].head_local)
        ).length
        for name in TAIL_BONES
    ], dtype=np.float64)
    config = TailSolverConfig(
        body_collision_start_point=3,
        temporal_ar=0.0 if independent_frames else temporal_ar,
        correction_iterations=120,
        maximum_bend_degrees=38.0,
    )
    if independent_frames:
        solver_tiers = (
            ("strict", config, 12),
            (
                "flexible",
                TailSolverConfig(
                    body_collision_start_point=3,
                    temporal_ar=0.0,
                    correction_iterations=240,
                    maximum_bend_degrees=52.0,
                ),
                16,
            ),
            (
                "proximal_attachment",
                TailSolverConfig(
                    body_collision_start_point=4,
                    temporal_ar=0.0,
                    correction_iterations=320,
                    maximum_bend_degrees=60.0,
                ),
                36,
            ),
        )
        chain_rows = []
        audits = []
        accepted_seeds = []
        attempt_counts = []
        accepted_tiers = []
        for frame_index in range(len(frames)):
            frame_best = None
            attempt_serial = 0
            accepted = False
            for tier_name, tier_config, tier_attempts in solver_tiers:
                for _tier_attempt in range(tier_attempts):
                    seed_index = seed_indices[frame_index] if seed_indices is not None else frame_index
                    attempt_seed = seed + seed_index * 104729 + attempt_serial * 7919
                    attempt_serial += 1
                    frame_chains, frame_audits = solve_tail_sequence(
                        np.asarray(bases[frame_index:frame_index + 1]),
                        np.asarray(tangents[frame_index:frame_index + 1]),
                        lengths,
                        seed=attempt_seed,
                        floor_z=floor_z,
                        arena=arena,
                        collision_proxies_by_frame=proxies[frame_index:frame_index + 1],
                        config=tier_config,
                    )
                    audit = frame_audits[0]
                    score = (not audit.accepted, audit.maximum_residual_penetration)
                    if frame_best is None or score < frame_best[0]:
                        frame_best = (
                            score, attempt_seed, frame_chains[0], audit, tier_name
                        )
                    if audit.accepted:
                        accepted = True
                        break
                if accepted:
                    break
            _score, frame_seed, frame_chain, frame_audit, tier_name = frame_best
            chain_rows.append(frame_chain)
            audits.append(frame_audit)
            accepted_seeds.append(frame_seed)
            attempt_counts.append(attempt_serial)
            accepted_tiers.append(tier_name)
        chains = np.stack(chain_rows)
        accepted_seed = {
            "mode": "per_frame",
            "first": accepted_seeds[0],
            "last": accepted_seeds[-1],
        }
        attempts_used = {
            "total": int(sum(attempt_counts)),
            "maximum_per_frame": int(max(attempt_counts)),
            "frames_requiring_resample": int(sum(value > 1 for value in attempt_counts)),
            "accepted_tier_counts": {
                tier_name: int(accepted_tiers.count(tier_name))
                for tier_name, _config, _attempts in solver_tiers
            },
        }
    else:
        best = None
        for attempt in range(20):
            attempt_seed = seed + attempt * 7919
            chains, audits = solve_tail_sequence(
                np.asarray(bases),
                np.asarray(tangents),
                lengths,
                seed=attempt_seed,
                floor_z=floor_z,
                arena=arena,
                collision_proxies_by_frame=proxies,
                config=config,
            )
            score = (
                sum(not audit.accepted for audit in audits),
                max(audit.maximum_residual_penetration for audit in audits),
            )
            if best is None or score < best[0]:
                best = (score, attempt_seed, chains, audits, attempt + 1)
            if score[0] == 0:
                break
        _score, accepted_seed, chains, audits, attempts_used = best
    rebuild_tail_spline(system["tail_curve"], chains.shape[1], "POLY")
    key_tail_curve(system["tail_curve"], frames, chains)
    return {
        "accepted_frames": sum(audit.accepted for audit in audits),
        "rejected_frames": sum(not audit.accepted for audit in audits),
        "maximum_residual_penetration_mm": max(a.maximum_residual_penetration for a in audits),
        "maximum_length_error_mm": max(a.length_error_max for a in audits),
        "accepted_seed": accepted_seed,
        "resampling_attempts": attempts_used,
        "proxy_profile": "two ellipsoids per mouse: rear + front",
        "rejected_details": [
            {
                "frame": int(frames[index]),
                "base": np.asarray(bases[index]).tolist(),
                "residual_mm": audit.maximum_residual_penetration,
            }
            for index, audit in enumerate(audits)
            if not audit.accepted
        ],
        "corrections": {
            "floor": sum(a.floor_pushes for a in audits),
            "wall": sum(a.wall_pushes for a in audits),
            "body": sum(a.body_pushes for a in audits),
        },
    }


def json_motion_summary(result: dict) -> dict:
    summaries = []
    for motion, feet in zip(result["motions"], result["foot_audits"], strict=True):
        summaries.append({
            "body_length_mm": motion["body_length_mm"],
            "source_length_median_mm": float(np.median(motion["source_lengths_mm"])),
            "curvature_range_mm": [float(np.min(motion["curvature_mm"])), float(np.max(motion["curvature_mm"]))],
            "motion_clock_mm": float(motion["motion_clock_mm"][-1]),
            "gait": feet,
        })
    return summaries


def main() -> None:
    args = parse_args()
    source_plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan, retiming = retime_plan(source_plan, args.frame_count)
    scene = bpy.context.scene
    source_scene_path = bpy.data.filepath
    scene.frame_set(1)
    bpy.context.view_layer.update()

    black = clone_mouse_system("Black")
    brown = clone_mouse_system("Brown")
    systems = [black, brown]
    brown_material = append_brown_material(args.brown_source, args.brown_source_material)
    brown_fur = assign_brown_fur(brown, brown_material)
    viewport_hidden_fur = hide_source_and_viewport_fur(systems)

    original_count = retiming["original_frame_range"][1] - retiming["original_frame_range"][0] + 1
    output_count = retiming["output_frame_range"][1] - retiming["output_frame_range"][0] + 1
    discrete_pose_frames = bool(
        args.discrete_pose_frames or plan.get("selection_policy") == "diverse_discrete_frames"
    )
    config = MotionConfig(
        active_motion_threshold_mm=MotionConfig().active_motion_threshold_mm
        * original_count / output_count,
        # A 20 mm centerline gap is narrower than two evaluated mouse torsos.
        # Independent social-contact samples therefore use a conservative
        # centerline clearance; final delivery still gets an exact mesh audit.
        minimum_body_axis_clearance_mm=(
            args.minimum_body_axis_clearance_mm if discrete_pose_frames else 20.0
        ),
        body_smoothing_passes=0 if discrete_pose_frames else 1,
        heading_smoothing_passes=0 if discrete_pose_frames else 2,
        curvature_smoothing_passes=0 if discrete_pose_frames else 2,
    )
    synthesized = synthesize(plan, config)
    depth_height = None
    if args.body_height_report is not None:
        depth_height = attach_depth_body_heights(
            synthesized,
            plan,
            args.body_height_report.resolve(),
            smooth_dense=not discrete_pose_frames,
        )
    frames = synthesized["frames"]
    floor_z = float(scene.get("MGF_ground_contact_floor_z_mm", -237.19223022460938))
    controls_by_system = []
    ik_reports = []
    for system, motion in zip(systems, synthesized["motions"], strict=True):
        controls = system_controls(system)
        controls_by_system.append(controls)
        ik_reports.append(configure_ik(system, controls))
        key_motion(system, controls, motion, frames, floor_z)

    set_all_interpolation(systems, "LINEAR")
    scene.frame_start = int(frames[0])
    scene.frame_end = int(frames[-1])
    scene.render.fps = int(plan.get("scene_fps", 8))
    bpy.context.view_layer.update()

    # Reject bad evaluated IK before the expensive per-frame tail solver.  The
    # tail curves do not affect the armature or paw-side test, so this audit is
    # final for limb geometry and makes source-pose replacement inexpensive.
    audit_frames = frames
    if args.preflight_start is not None:
        audit_frames = audit_frames[audit_frames >= args.preflight_start]
    if args.preflight_end is not None:
        audit_frames = audit_frames[audit_frames <= args.preflight_end]
    if not len(audit_frames):
        raise ValueError("Preflight frame range does not intersect the animation")
    evaluated = evaluated_audit(
        systems,
        audit_frames,
        floor_z,
        stop_on_same_side=args.preflight_only,
    )
    if evaluated["same_side_limb_frames"]:
        violations = []
        for row in evaluated["rows"]:
            for instance in row["instances"]:
                for limb, values in instance["limbs"].items():
                    if values["signed_lateral_mm"] <= 0.0:
                        violations.append({
                            "frame": row["frame"],
                            "instance": instance["label"],
                            "limb": limb,
                            "signed_lateral_mm": values["signed_lateral_mm"],
                        })
        raise RuntimeError(
            "Evaluated IK crossed the body centerline: "
            + json.dumps(violations, ensure_ascii=False)
        )
    if args.preflight_only:
        print(json.dumps({
            "preflight_frames": [int(audit_frames[0]), int(audit_frames[-1])],
            "evaluated_frame_count": len(evaluated["rows"]),
            "same_side_limb_frames": evaluated["same_side_limb_frames"],
            "minimum_evaluated_body_axis_distance_mm": (
                evaluated["minimum_evaluated_body_axis_distance_mm"]
            ),
        }, indent=2))
        return

    arena = ArenaBounds((-132.0, -98.0, floor_z), (132.0, 98.0, floor_z + 150.0))
    tail_frames = list(frames)
    tail_seed_indices = None
    if args.tail_qa_only_frames:
        tail_frames = [
            int(value.strip())
            for value in args.tail_qa_only_frames.split(",")
            if value.strip()
        ]
        unknown = sorted(set(tail_frames) - set(int(value) for value in frames))
        if unknown:
            raise ValueError(f"Tail QA frames are outside the plan: {unknown}")
        frame_to_index = {int(value): index for index, value in enumerate(frames)}
        tail_seed_indices = [frame_to_index[value] for value in tail_frames]
    tail_audits = {
        black["label"]: solve_system_tail_strict(
            black, brown, tail_frames, args.seed, floor_z, arena,
            temporal_ar=0.45 if discrete_pose_frames else 0.97,
            independent_frames=discrete_pose_frames,
            seed_indices=tail_seed_indices,
        ),
        brown["label"]: solve_system_tail_strict(
            brown, black, tail_frames, args.seed + 1009, floor_z, arena,
            temporal_ar=0.45 if discrete_pose_frames else 0.97,
            independent_frames=discrete_pose_frames,
            seed_indices=tail_seed_indices,
        ),
    }
    if any(value["rejected_frames"] for value in tail_audits.values()):
        raise RuntimeError(f"Tail solver rejected frames: {tail_audits}")
    if args.tail_qa_only_frames:
        print(json.dumps({
            "tail_qa_only_frames": tail_frames,
            "tail": tail_audits,
        }, indent=2))
        return


    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene["MGF_project_kind"] = "LightningPose CRIM13 dual mouse procedural kinematics v1"
    scene["MGF_motion_source"] = "current LightningPose segment + deterministic procedural contact gait"
    scene["MGF_dynamic_donor_motion"] = "none"
    scene["MGF_static_rig_reference_only"] = True
    scene["MGF_body_height_policy"] = (
        "VGGT monocular relative height residual on validated neutral rig; paws remain contact constrained"
        if depth_height is not None
        else "fixed neutral height profile; no donor-video height channel"
    )
    scene["MGF_body_height_report"] = depth_height["report"] if depth_height is not None else "none"
    scene["MGF_limb_policy"] = "contact-aware procedural IK; no donor-video wrist/ankle channel"
    scene["MGF_spine_policy"] = "current-segment heading and neck curvature; no donor-video spine channel"
    scene["MGF_sparse_conditioning_warning"] = (
        "none: every Blender frame is backed by a selected real labeled source frame"
        if discrete_pose_frames
        else "sparse labels; intermediate frames are synthetic"
    )
    scene["MGF_discrete_pose_frames"] = discrete_pose_frames
    scene["MGF_pose_frame_policy"] = (
        "one real labeled source pose per Blender frame; no body interpolation smoothing"
        if discrete_pose_frames
        else "sparse source labels with synthetic intermediate frames"
    )
    scene["MGF_retiming"] = json.dumps(retiming)
    scene["MGF_dual_mouse_instances"] = json.dumps([system["armature"].name for system in systems])
    scene["MGF_fur_viewport_default"] = "hidden"
    scene["MGF_fur_render_enabled"] = True
    scene["MGF_fur_viewport_objects"] = json.dumps(viewport_hidden_fur)
    scene.frame_set(int(frames[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), check_existing=False)
    report = {
        "schema": "mousegeoforge.lightningpose_dual_mouse_procedural_kinematics.audit.v1",
        "output_blend": str(args.output),
        "plan": str(args.plan),
        "source_scene": source_scene_path,
        "frame_range": [int(frames[0]), int(frames[-1])],
        "motion_provenance": {
            "static_rig_basis": "source rig frame-1 single-key matrix basis and bone lengths",
            "dynamic_body": "current LightningPose segment only",
            "dynamic_feet": "deterministic gait clock from current-segment translation and heading change",
            "dynamic_tail": "geometry-first collision-corrected solver",
            "whitefur_controller_moseq_dynamic_channels": [],
            "other_video_dynamic_channels": [],
            "discrete_pose_frames": discrete_pose_frames,
        },
        "config": asdict(config),
        "retiming": retiming,
        "motion": json_motion_summary(synthesized),
        "depth_body_height": depth_height,
        "spacing": synthesized["spacing_audit"],
        "ik": ik_reports,
        "brown_material": brown_material.name,
        "brown_fur_objects": brown_fur,
        "tail": tail_audits,
        "evaluated": evaluated,
    }
    report_path = args.report or args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(report_path),
        "spacing": report["spacing"],
        "motion": report["motion"],
        "evaluated": {key: value for key, value in evaluated.items() if key != "rows"},
    }, indent=2))


if __name__ == "__main__":
    main()

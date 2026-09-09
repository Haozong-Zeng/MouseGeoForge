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
import mjx_forelimb_prior as mjxprior
import probe_forelimb_interpolation as interp
import refine_forelimb_collisions as refine


DEFAULT_OBSTACLES = [
    "Part_MouseCupSide.001",
    "Part_MouseCupSide.002",
    "Part_PrintedMouseCup",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refine collapsed forelimb frames by matching the target evaluated mesh "
            "to the source evaluated mesh in a torso-aligned frame."
        )
    )
    parser.add_argument("--source-blend", required=True)
    parser.add_argument("--target-blend", required=True)
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--compare-json", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--source-mesh", default="MS_Mouse")
    parser.add_argument("--source-action", default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--obstacle", action="append", default=[])
    parser.add_argument("--volume-threshold", type=float, default=0.72)
    parser.add_argument("--min-volume-delta", type=float, default=0.006)
    parser.add_argument("--max-source-nrmse-increase", type=float, default=0.025)
    parser.add_argument("--allowed-clearance-drop", type=float, default=0.05)
    parser.add_argument("--clearance-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-threshold", type=int, default=50)
    parser.add_argument("--top", type=int, default=999)
    parser.add_argument("--target-flag", action="append", default=[])
    parser.add_argument("--target-frame-side", action="append", default=[])
    parser.add_argument("--compare-top", type=int, default=0)
    parser.add_argument("--compare-min-arm-paw-nrmse", type=float, default=0.10)
    parser.add_argument("--compare-min-arm-nrmse", type=float, default=None)
    parser.add_argument("--compare-min-abs-reach-delta", type=float, default=None)
    parser.add_argument("--step", type=float, action="append", default=[])
    parser.add_argument(
        "--source-palm-mode",
        action="append",
        choices=("bone", "paw-centroid", "wrist-centroid", "paw-wrist-midpoint"),
        default=[],
    )
    parser.add_argument("--source-palm-blend", type=float, action="append", default=[])
    parser.add_argument("--scapula-roll-deg", type=float, action="append", default=[])
    parser.add_argument("--wrist-roll-deg", type=float, action="append", default=[])
    parser.add_argument("--mesh-weight", type=float, default=1.0)
    parser.add_argument("--volume-weight", type=float, default=5.0)
    parser.add_argument("--source-volume-weight", type=float, default=0.0)
    parser.add_argument("--source-volume-threshold", type=float, default=0.45)
    parser.add_argument("--allowed-source-volume-drop", type=float, default=0.03)
    parser.add_argument(
        "--source-volume-palm-mode",
        choices=("bone", "paw-centroid", "wrist-centroid", "paw-wrist-midpoint"),
        default="paw-wrist-midpoint",
    )
    parser.add_argument("--clearance-weight", type=float, default=0.75)
    parser.add_argument("--overlap-weight", type=float, default=1000.0)
    parser.add_argument("--reach-ratio-threshold", type=float, default=0.57)
    parser.add_argument("--reach-weight", type=float, default=0.0)
    parser.add_argument("--mjx-prior-data-dir", default=None)
    parser.add_argument("--mjx-prior-weight", type=float, default=0.0)
    parser.add_argument("--mjx-prior-nearest-top", type=int, default=5)
    parser.add_argument("--pose-change-weight", type=float, default=0.08)
    parser.add_argument("--arm-volume-bins", type=int, default=6)
    parser.add_argument("--arm-volume-trim", type=float, default=0.12)
    parser.add_argument("--include-paw-mesh", action="store_true")
    parser.add_argument("--paw-mesh-weight", type=float, default=0.35)
    parser.add_argument("--allow-alignment-scale", action="store_true")
    parser.add_argument("--keyframe-all-forelimb-bones", action="store_true")
    args = parser.parse_args(args_after_dash())
    args.target_flag = normalize_target_flags(args.target_flag)
    args.target_frame_side = normalize_frame_side_targets(args.target_frame_side)
    return args


def normalize_target_flags(flags):
    normalized = []
    seen = set()
    for value in flags or []:
        for part in str(value).split(","):
            flag = part.strip()
            if flag and flag not in seen:
                seen.add(flag)
                normalized.append(flag)
    return normalized


def normalize_frame_side_targets(values):
    targets = []
    seen = set()
    for value in values or []:
        for part in str(value).split(","):
            text = part.strip()
            if not text:
                continue
            if ":" in text:
                frame_text, side = text.split(":", 1)
            elif "=" in text:
                frame_text, side = text.split("=", 1)
            else:
                raise ValueError(f"Expected frame:side target, got {text!r}")
            frame = int(round(float(frame_text.strip())))
            side = side.strip().upper()
            if side not in {"L", "R"}:
                raise ValueError(f"Expected side L or R in {text!r}")
            key = (frame, side)
            if key not in seen:
                seen.add(key)
                targets.append(key)
    return targets


def side_from_flag(flag):
    if flag.startswith("L_"):
        return "L"
    if flag.startswith("R_"):
        return "R"
    return None


def flag_matches(flag, requested):
    return flag == requested or flag.endswith(f"_{requested}")


def target_severity(row, side, volume_threshold):
    volume_ratio = interp.ratio_for(row, side)
    pose = row.get("forelimb_pose", {}).get(side, {})
    reach_ratio = pose.get("reach_ratio")
    clearance = row.get("paw_clearance", {}).get(side, {})
    min_distance = clearance.get("min_distance")
    if volume_ratio is not None and volume_ratio < volume_threshold:
        return float(volume_ratio)
    if reach_ratio is not None:
        return float(reach_ratio)
    if min_distance is not None:
        return float(min_distance)
    return 1.0


def float_or_none(value):
    return None if value is None else float(value)


def compare_targets(args):
    if not args.compare_json or args.compare_top <= 0:
        return []
    compare = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    rows = []
    if compare.get("worst_by_arm_paw_nrmse"):
        rows = compare["worst_by_arm_paw_nrmse"]
    else:
        for frame_row in compare.get("rows", []):
            for side in ("L", "R"):
                side_row = frame_row.get("sides", {}).get(side, {})
                rows.append(
                    {
                        "frame": frame_row.get("frame"),
                        "side": side,
                        "arm_nrmse": side_row.get("arm_error", {}).get("nrmse"),
                        "arm_paw_nrmse": side_row.get("arm_paw_error", {}).get("nrmse"),
                        "reach_ratio_delta": side_row.get("reach_ratio_delta"),
                    }
                )
    selected = []
    seen = set()
    for row in rows:
        frame = row.get("frame")
        side = row.get("side")
        if frame is None or side not in {"L", "R"}:
            continue
        arm_paw = float_or_none(row.get("arm_paw_nrmse"))
        arm = float_or_none(row.get("arm_nrmse"))
        reach_delta = float_or_none(row.get("reach_ratio_delta"))
        if args.compare_min_arm_paw_nrmse is not None:
            if arm_paw is None or arm_paw < args.compare_min_arm_paw_nrmse:
                continue
        if args.compare_min_arm_nrmse is not None:
            if arm is None or arm < args.compare_min_arm_nrmse:
                continue
        if args.compare_min_abs_reach_delta is not None:
            if reach_delta is None or abs(reach_delta) < args.compare_min_abs_reach_delta:
                continue
        key = (int(round(float(frame))), side)
        if key in seen:
            continue
        seen.add(key)
        severity = -(arm_paw if arm_paw is not None else arm if arm is not None else 0.0)
        selected.append((severity, key[0], side))
        if len(selected) >= args.compare_top:
            break
    return selected


def quality_targets(quality_path, action_name, threshold, top, target_flags, args):
    quality = json.loads(Path(quality_path).read_text(encoding="utf-8"))
    quality_action = next(
        action for action in quality["actions"] if action["action"] == action_name
    )
    rows_by_frame = interp.frame_rows_by_number(quality_action)
    targets_by_key = {}
    for frame, row in rows_by_frame.items():
        row_flags = set(row.get("flags", []))
        if target_flags:
            matched = [
                flag
                for flag in row_flags
                if any(flag_matches(flag, requested) for requested in target_flags)
            ]
            for flag in matched:
                side = side_from_flag(flag)
                sides = (side,) if side is not None else ("L", "R")
                for target_side in sides:
                    key = (frame, target_side)
                    severity = target_severity(row, target_side, threshold)
                    previous = targets_by_key.get(key)
                    if previous is None or severity < previous[0]:
                        targets_by_key[key] = (severity, frame, target_side)
            continue
        for side in ("L", "R"):
            ratio = interp.ratio_for(row, side)
            if ratio is not None and ratio < threshold:
                targets_by_key[(frame, side)] = (ratio, frame, side)
    for severity, frame, side in compare_targets(args):
        key = (frame, side)
        previous = targets_by_key.get(key)
        if previous is None or severity < previous[0]:
            targets_by_key[key] = (severity, frame, side)
    for frame, side in args.target_frame_side:
        targets_by_key[(frame, side)] = (-999.0, frame, side)
    targets = list(targets_by_key.values())
    targets.sort(key=lambda item: (item[0], item[1], item[2]))
    return targets[:top], rows_by_frame, quality_action


def set_source_action(armature, action_name):
    armature.animation_data_create()
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Source action {action_name!r} not found.")
        armature.animation_data.action = action
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    actions = list(bpy.data.actions)
    if not actions:
        raise RuntimeError("No source action found.")
    action = sorted(actions, key=lambda item: item.name)[0]
    armature.animation_data.action = action
    return action


def source_bone_points_world(armature, bone_names):
    heads = {}
    tails = {}
    for name in bone_names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        heads[name] = (armature.matrix_world @ pose_bone.matrix.translation).copy()
        tails[name] = (
            armature.matrix_world
            @ (pose_bone.matrix @ Vector((0.0, pose_bone.bone.length, 0.0)))
        ).copy()
    return heads, tails


def collect_source_payload(args, frames):
    print(f"[source] opening {args.source_blend}")
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.source_blend).resolve()))
    armature = mig.find_armature(args.source_armature, require_action=False)
    mesh_obj = bpy.data.objects.get(args.source_mesh)
    if mesh_obj is None:
        raise RuntimeError(f"Source mesh {args.source_mesh!r} not found.")
    action = set_source_action(armature, args.source_action)

    arm_indices, arm_groups = qc.collect_arm_vertex_indices(mesh_obj, 0.001)
    paw_indices, paw_groups = qc.collect_paw_vertex_indices(mesh_obj, False, 0.001)
    bone_names = sorted(
        set(mig.BODY_TAIL_ALIGNMENT_BONES)
        | {f"Wrist_{side}" for side in ("L", "R")}
        | {f"Scapula_{side}" for side in ("L", "R")}
    )
    payload = {
        "blend": str(Path(args.source_blend).resolve()),
        "armature": armature.name,
        "mesh": mesh_obj.name,
        "action": action.name,
        "mesh_vertex_count": len(mesh_obj.data.vertices),
        "arm_vertex_counts": {
            side: {segment: len(values) for segment, values in segments.items()}
            for side, segments in arm_indices.items()
        },
        "paw_vertex_counts": {side: len(values) for side, values in paw_indices.items()},
        "arm_groups": arm_groups,
        "paw_groups": paw_groups,
        "frames": {},
    }
    for index, frame in enumerate(frames, start=1):
        qc.set_frame(frame)
        vertices, _polygons = qc.evaluated_mesh_world(mesh_obj)
        heads, tails = source_bone_points_world(armature, bone_names)
        payload["frames"][int(frame)] = {
            "vertex_count": len(vertices),
            "vertices": vertices,
            "heads": heads,
            "tails": tails,
        }
        if index == 1 or index == len(frames) or index % 10 == 0:
            print(f"  source mesh frame {index}/{len(frames)}")
    return payload


def target_body_heads(armature):
    return {
        name: qc.pose_bone_head_world(armature, name)
        for name in mig.BODY_TAIL_ALIGNMENT_BONES
        if qc.pose_bone_head_world(armature, name) is not None
    }


def alignment_from_source_to_target(source_frame, target_heads, allow_scale):
    source_points = []
    target_points = []
    for name in mig.BODY_TAIL_ALIGNMENT_BONES:
        source_head = source_frame["heads"].get(name)
        target_head = target_heads.get(name)
        if source_head is not None and target_head is not None:
            source_points.append(source_head)
            target_points.append(target_head)
    matrix, scale, count = mig.similarity_matrix_from_points(
        source_points,
        target_points,
        allow_scale=allow_scale,
    )
    return matrix, scale, count


def selected_source_targets(source_frame, alignment, indices):
    vertices = source_frame["vertices"]
    targets = {}
    for index in indices:
        if index < len(vertices):
            targets[index] = alignment @ vertices[index]
    return targets


def bbox_diagonal(points):
    if not points:
        return 1.0
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    min_z = min(point.z for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    max_z = max(point.z for point in points)
    diagonal = Vector((max_x - min_x, max_y - min_y, max_z - min_z)).length
    return max(diagonal, 1.0e-8)


def source_mesh_error(vertices, selected_targets):
    if not selected_targets:
        return {
            "count": 0,
            "rms": None,
            "nrmse": None,
            "bbox_diagonal": None,
        }
    squared = 0.0
    used_points = []
    for index, target in selected_targets.items():
        if index >= len(vertices):
            continue
        delta = vertices[index] - target
        squared += delta.length_squared
        used_points.append(target)
    if not used_points:
        return {
            "count": 0,
            "rms": None,
            "nrmse": None,
            "bbox_diagonal": None,
        }
    rms = math.sqrt(squared / len(used_points))
    diagonal = bbox_diagonal(used_points)
    return {
        "count": len(used_points),
        "rms": float(rms),
        "nrmse": float(rms / diagonal),
        "bbox_diagonal": float(diagonal),
    }


def volume_ratios(volume, reference):
    ratios = {}
    worst_ratio = None
    worst_segment = None
    for segment, value in (
        ("upper", volume.get("upper", {}).get("volume_proxy")),
        ("lower", volume.get("lower", {}).get("volume_proxy")),
        ("total", volume.get("total_volume_proxy")),
    ):
        ratio = None
        ref = reference.get(segment)
        if ref and value is not None:
            ratio = float(value / ref)
            if worst_ratio is None or ratio < worst_ratio:
                worst_ratio = ratio
                worst_segment = segment
        ratios[segment] = ratio
    return ratios, worst_ratio, worst_segment


def volume_from_chain(vertices, side_indices, shoulder, elbow, palm, bins, trim):
    if shoulder is None or elbow is None or palm is None:
        return {"available": False, "upper": {}, "lower": {}, "total_volume_proxy": None}
    upper = qc.volume_proxy_for_segment(
        vertices,
        side_indices["upper"],
        shoulder,
        elbow,
        bins,
        trim,
    )
    lower = qc.volume_proxy_for_segment(
        vertices,
        side_indices["lower"],
        elbow,
        palm,
        bins,
        trim,
    )
    total = 0.0
    available = False
    for segment in (upper, lower):
        if segment["available"] and segment["volume_proxy"] is not None:
            total += segment["volume_proxy"]
            available = True
    return {
        "available": available,
        "upper": upper,
        "lower": lower,
        "total_volume_proxy": float(total) if available else None,
    }


def volume_reference_from_volume(volume):
    return {
        "upper": volume.get("upper", {}).get("volume_proxy"),
        "lower": volume.get("lower", {}).get("volume_proxy"),
        "total": volume.get("total_volume_proxy"),
    }


def evaluate_state(
    armature,
    mesh_obj,
    arm_indices,
    paw_indices,
    obstacles,
    side,
    volume_reference,
    source_volume_reference,
    source_targets,
    source_paw_targets,
    args,
    current_palm,
    physical_prior,
):
    vertices, polygons = qc.evaluated_mesh_world(mesh_obj)
    arm_error = source_mesh_error(vertices, source_targets)
    paw_error = (
        source_mesh_error(vertices, source_paw_targets)
        if args.include_paw_mesh
        else {"count": 0, "rms": None, "nrmse": 0.0, "bbox_diagonal": None}
    )
    volume = qc.forelimb_volume_metrics(
        vertices,
        armature,
        side,
        arm_indices,
        args.arm_volume_bins,
        args.arm_volume_trim,
    )
    ratios, worst_ratio, worst_segment = volume_ratios(volume, volume_reference)
    source_volume_ratios, source_worst_ratio, source_worst_segment = (
        volume_ratios(volume, source_volume_reference)
        if source_volume_reference is not None
        else ({}, None, None)
    )
    clearance = qc.paw_clearance_for_side(
        vertices,
        polygons,
        paw_indices[side],
        side,
        obstacles,
        "any",
    )
    pose = qc.forelimb_pose_metrics(armature, side)
    palm = refine.bone_tail_world(armature, f"Wrist_{side}")
    palm_delta = (palm - current_palm).length if palm is not None and current_palm is not None else 0.0

    arm_nrmse = arm_error["nrmse"] if arm_error["nrmse"] is not None else float("inf")
    paw_nrmse = paw_error["nrmse"] if paw_error["nrmse"] is not None else 0.0
    volume_shortfall = 0.0
    if worst_ratio is None:
        volume_shortfall = 1.0
    elif worst_ratio < args.volume_threshold:
        volume_shortfall = (args.volume_threshold - worst_ratio) / max(args.volume_threshold, 1.0e-8)

    source_volume_shortfall = 0.0
    if args.source_volume_weight != 0.0:
        if source_worst_ratio is None:
            source_volume_shortfall = 1.0
        elif source_worst_ratio < args.source_volume_threshold:
            source_volume_shortfall = (
                args.source_volume_threshold - source_worst_ratio
            ) / max(args.source_volume_threshold, 1.0e-8)

    clearance_penalty = 0.0
    if clearance["min_distance"] is not None and clearance["min_distance"] < args.clearance_threshold:
        clearance_penalty = args.clearance_threshold - clearance["min_distance"]
    reach_ratio = pose.get("reach_ratio")
    reach_shortfall = 0.0
    if reach_ratio is None:
        reach_shortfall = 1.0
    elif reach_ratio < args.reach_ratio_threshold:
        reach_shortfall = (
            args.reach_ratio_threshold - reach_ratio
        ) / max(args.reach_ratio_threshold, 1.0e-8)

    mjx_score = None
    mjx_distance = 0.0
    if physical_prior is not None and args.mjx_prior_weight != 0.0:
        mjx_features = mjxprior.blender_forelimb_features(armature, side)
        mjx_score = physical_prior.score_features(
            mjx_features,
            args.mjx_prior_nearest_top,
        )
        if mjx_score.get("available"):
            mjx_distance = float(mjx_score.get("distance", 0.0))
        else:
            mjx_distance = 1.0
    score = (
        args.mesh_weight * arm_nrmse
        + args.paw_mesh_weight * paw_nrmse
        + args.volume_weight * volume_shortfall
        + args.source_volume_weight * source_volume_shortfall
        + args.clearance_weight * clearance_penalty
        + args.overlap_weight * max(0, clearance["overlap_count"] - args.overlap_threshold + 1)
        + args.reach_weight * reach_shortfall
        + args.mjx_prior_weight * mjx_distance
        + args.pose_change_weight * palm_delta
    )
    return {
        "score": float(score),
        "arm_source_error": arm_error,
        "paw_source_error": paw_error,
        "worst_volume_ratio": worst_ratio,
        "worst_volume_segment": worst_segment,
        "upper_volume_ratio": ratios.get("upper"),
        "lower_volume_ratio": ratios.get("lower"),
        "total_volume_ratio": ratios.get("total"),
        "volume_shortfall": float(volume_shortfall),
        "source_worst_volume_ratio": source_worst_ratio,
        "source_worst_volume_segment": source_worst_segment,
        "source_upper_volume_ratio": source_volume_ratios.get("upper"),
        "source_lower_volume_ratio": source_volume_ratios.get("lower"),
        "source_total_volume_ratio": source_volume_ratios.get("total"),
        "source_volume_shortfall": float(source_volume_shortfall),
        "overlap_count": int(clearance["overlap_count"]),
        "min_distance": clearance.get("min_distance"),
        "reach_ratio": reach_ratio,
        "reach_shortfall": float(reach_shortfall),
        "elbow_angle_deg": pose.get("elbow_angle_deg"),
        "mjx_prior": mjx_score,
        "mjx_prior_distance": float(mjx_distance),
        "palm_delta": float(palm_delta),
    }


def unique_vectors(vectors, precision=5):
    unique = []
    seen = set()
    for vector in vectors:
        key = tuple(round(float(value), precision) for value in vector)
        if key in seen:
            continue
        seen.add(key)
        unique.append(vector.copy())
    return unique


def centroid_of_targets(targets):
    values = list(targets.values())
    if not values:
        return None
    return sum(values, Vector()) / len(values)


def candidate_palms(current_palm, source_palms, args):
    steps = args.step or [0.35, 0.8]
    vectors = refine.candidate_vectors(steps)
    palms = [current_palm + vector for vector in vectors]
    blends = args.source_palm_blend or [0.25, 0.5, 0.75, 1.0]
    for source_palm in source_palms:
        if source_palm is None:
            continue
        for blend in blends:
            amount = max(0.0, min(float(blend), 1.0))
            palms.append(current_palm.lerp(source_palm, amount))
    return unique_vectors(palms)


def candidate_is_safe(baseline, candidate, args):
    if candidate["overlap_count"] > baseline["overlap_count"]:
        return False, "overlap_worse"
    if candidate["overlap_count"] >= args.overlap_threshold:
        return False, "severe_overlap"

    base_distance = baseline["min_distance"]
    candidate_distance = candidate["min_distance"]
    if base_distance is not None and candidate_distance is not None:
        if base_distance >= args.clearance_threshold and candidate_distance < args.clearance_threshold:
            return False, "clearance_became_warning"
        if (
            min(base_distance, candidate_distance) < args.clearance_threshold
            and candidate_distance + args.allowed_clearance_drop < base_distance
        ):
            return False, "clearance_drop"

    base_ratio = baseline["worst_volume_ratio"]
    candidate_ratio = candidate["worst_volume_ratio"]
    if base_ratio is None or candidate_ratio is None:
        return False, "missing_volume"
    if base_ratio < args.volume_threshold and candidate_ratio < args.volume_threshold:
        if candidate_ratio - base_ratio < args.min_volume_delta:
            return False, "small_volume_gain"

    if args.source_volume_weight != 0.0:
        base_source_ratio = baseline.get("source_worst_volume_ratio")
        candidate_source_ratio = candidate.get("source_worst_volume_ratio")
        if base_source_ratio is not None and candidate_source_ratio is not None:
            if candidate_source_ratio + args.allowed_source_volume_drop < base_source_ratio:
                return False, "source_volume_drop"

    base_nrmse = baseline["arm_source_error"]["nrmse"]
    candidate_nrmse = candidate["arm_source_error"]["nrmse"]
    if base_nrmse is None or candidate_nrmse is None:
        return False, "missing_source_error"
    if candidate_nrmse > base_nrmse + args.max_source_nrmse_increase:
        return False, "source_mesh_worse"

    if candidate["score"] >= baseline["score"]:
        return False, "score_not_improved"
    return True, "accepted"


def keyframe_bones(armature, side, frame, all_forelimb):
    refine.keyframe_refined_bones(armature, side, frame, all_forelimb)


def refine_frame_side(
    armature,
    mesh_obj,
    arm_indices,
    paw_indices,
    obstacles,
    side,
    frame,
    source_frame,
    volume_reference,
    args,
    physical_prior,
):
    target_heads = target_body_heads(armature)
    alignment, alignment_scale, alignment_count = alignment_from_source_to_target(
        source_frame,
        target_heads,
        args.allow_alignment_scale,
    )
    selected_arm = set(arm_indices[side]["upper"]) | set(arm_indices[side]["lower"])
    source_targets = selected_source_targets(source_frame, alignment, selected_arm)
    source_paw_targets = selected_source_targets(source_frame, alignment, paw_indices[side])
    source_wrist_targets = selected_source_targets(source_frame, alignment, arm_indices[side]["lower"])
    source_bone_palm = source_frame["tails"].get(f"Wrist_{side}")
    aligned_source_bone_palm = alignment @ source_bone_palm if source_bone_palm is not None else None
    source_paw_centroid = centroid_of_targets(source_paw_targets)
    source_wrist_centroid = centroid_of_targets(source_wrist_targets)
    source_paw_wrist_midpoint = (
        source_wrist_centroid.lerp(source_paw_centroid, 0.5)
        if source_wrist_centroid is not None and source_paw_centroid is not None
        else None
    )
    source_palm_modes = args.source_palm_mode or ["bone"]
    source_palm_by_mode = {
        "bone": aligned_source_bone_palm,
        "paw-centroid": source_paw_centroid,
        "wrist-centroid": source_wrist_centroid,
        "paw-wrist-midpoint": source_paw_wrist_midpoint,
    }
    source_palms = [
        source_palm_by_mode[mode]
        for mode in source_palm_modes
        if source_palm_by_mode.get(mode) is not None
    ]

    aligned_source_vertices = [alignment @ vertex for vertex in source_frame["vertices"]]
    source_shoulder = source_frame["heads"].get(f"Scapula_{side}")
    source_elbow = source_frame["tails"].get(f"Scapula_{side}")
    source_volume_palm = source_palm_by_mode.get(args.source_volume_palm_mode)
    source_volume = volume_from_chain(
        aligned_source_vertices,
        arm_indices[side],
        alignment @ source_shoulder if source_shoulder is not None else None,
        alignment @ source_elbow if source_elbow is not None else None,
        source_volume_palm,
        args.arm_volume_bins,
        args.arm_volume_trim,
    )
    source_volume_reference = volume_reference_from_volume(source_volume)

    current_palm = refine.bone_tail_world(armature, f"Wrist_{side}")
    pose_snapshot = refine.snapshot_pose_basis(armature, [f"Scapula_{side}", f"Wrist_{side}"])
    baseline = evaluate_state(
        armature,
        mesh_obj,
        arm_indices,
        paw_indices,
        obstacles,
        side,
        volume_reference,
        source_volume_reference,
        source_targets,
        source_paw_targets,
        args,
        current_palm,
        physical_prior,
    )
    best = baseline
    best_palm = current_palm.copy()
    best_scapula_roll = 0.0
    best_wrist_roll = 0.0
    best_safe = None
    best_safe_palm = None
    best_safe_scapula_roll = 0.0
    best_safe_wrist_roll = 0.0
    evaluated = 0

    scapula_rolls = sorted(set([0.0] + [float(value) for value in args.scapula_roll_deg]))
    wrist_rolls = sorted(set([0.0] + [float(value) for value in args.wrist_roll_deg]))
    if not args.scapula_roll_deg:
        scapula_rolls = [-24.0, -12.0, 0.0, 12.0, 24.0]
    if not args.wrist_roll_deg:
        wrist_rolls = [-36.0, -18.0, 0.0, 18.0, 36.0]

    for palm_target in candidate_palms(current_palm, source_palms, args):
        for scapula_roll in scapula_rolls:
            for wrist_roll in wrist_rolls:
                refine.restore_pose_basis(armature, pose_snapshot)
                refine.set_forelimb_palm_world(
                    armature,
                    side,
                    palm_target,
                    min_reach_ratio=args.reach_ratio_threshold,
                )
                refine.roll_pose_bone_around_local_y(armature, f"Scapula_{side}", scapula_roll)
                refine.roll_pose_bone_around_local_y(armature, f"Wrist_{side}", wrist_roll)
                candidate = evaluate_state(
                    armature,
                    mesh_obj,
                    arm_indices,
                    paw_indices,
                    obstacles,
                    side,
                    volume_reference,
                    source_volume_reference,
                    source_targets,
                    source_paw_targets,
                    args,
                    current_palm,
                    physical_prior,
                )
                evaluated += 1
                if candidate["score"] < best["score"]:
                    best = candidate
                    best_palm = palm_target.copy()
                    best_scapula_roll = float(scapula_roll)
                    best_wrist_roll = float(wrist_roll)
                candidate_safe, _candidate_reason = candidate_is_safe(baseline, candidate, args)
                if candidate_safe and (best_safe is None or candidate["score"] < best_safe["score"]):
                    best_safe = candidate
                    best_safe_palm = palm_target.copy()
                    best_safe_scapula_roll = float(scapula_roll)
                    best_safe_wrist_roll = float(wrist_roll)

    best_overall = best
    best_overall_palm = best_palm
    best_overall_scapula_roll = best_scapula_roll
    best_overall_wrist_roll = best_wrist_roll
    if best_safe is not None:
        best = best_safe
        best_palm = best_safe_palm
        best_scapula_roll = best_safe_scapula_roll
        best_wrist_roll = best_safe_wrist_roll
        safe = True
        reason = "accepted"
    else:
        safe, reason = candidate_is_safe(baseline, best, args)
    refine.restore_pose_basis(armature, pose_snapshot)
    if safe:
        refine.set_forelimb_palm_world(
            armature,
            side,
            best_palm,
            min_reach_ratio=args.reach_ratio_threshold,
        )
        refine.roll_pose_bone_around_local_y(armature, f"Scapula_{side}", best_scapula_roll)
        refine.roll_pose_bone_around_local_y(armature, f"Wrist_{side}", best_wrist_roll)
        keyframe_bones(armature, side, frame, args.keyframe_all_forelimb_bones)

    return {
        "frame": float(frame),
        "side": side,
        "decision": "accepted" if safe else "rejected",
        "reason": reason,
        "alignment_scale": float(alignment_scale),
        "alignment_point_count": int(alignment_count),
        "source_palm_candidates": {
            mode: None
            if source_palm_by_mode.get(mode) is None
            else [float(value) for value in source_palm_by_mode[mode]]
            for mode in ("bone", "paw-centroid", "wrist-centroid", "paw-wrist-midpoint")
        },
        "source_palm_distances": {
            mode: None
            if source_palm_by_mode.get(mode) is None
            else float((source_palm_by_mode[mode] - current_palm).length)
            for mode in ("bone", "paw-centroid", "wrist-centroid", "paw-wrist-midpoint")
        },
        "source_volume_reference": source_volume_reference,
        "candidate_count": int(evaluated),
        "best_scapula_roll_deg": best_scapula_roll,
        "best_wrist_roll_deg": best_wrist_roll,
        "best_palm": [float(value) for value in best_palm],
        "baseline": baseline,
        "best": best,
        "best_overall": best_overall,
        "best_overall_scapula_roll_deg": best_overall_scapula_roll,
        "best_overall_wrist_roll_deg": best_overall_wrist_roll,
        "best_overall_palm": [float(value) for value in best_overall_palm],
        "volume_delta": None
        if baseline["worst_volume_ratio"] is None or best["worst_volume_ratio"] is None
        else float(best["worst_volume_ratio"] - baseline["worst_volume_ratio"]),
        "arm_source_nrmse_delta": None
        if baseline["arm_source_error"]["nrmse"] is None
        or best["arm_source_error"]["nrmse"] is None
        else float(best["arm_source_error"]["nrmse"] - baseline["arm_source_error"]["nrmse"]),
        "score_delta": float(best["score"] - baseline["score"]),
    }


def main():
    args = parse_args()
    targets, _rows_by_frame, quality_action = quality_targets(
        args.quality_json,
        args.action,
        args.volume_threshold,
        args.top,
        args.target_flag,
        args,
    )
    target_frames = sorted({frame for _ratio, frame, _side in targets})
    if not target_frames:
        raise RuntimeError("No quality or source-compare target frames found.")

    source_payload = collect_source_payload(args, target_frames)

    print(f"[target] opening {args.target_blend}")
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.target_blend).resolve()))
    armature = bpy.data.objects.get(args.armature)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if armature is None or mesh_obj is None:
        raise RuntimeError("Target armature or mesh not found.")
    action = bpy.data.actions.get(args.action)
    if action is None:
        raise RuntimeError(f"Target action {args.action!r} not found.")
    qc.set_action(armature, action)
    physical_prior = None
    if args.mjx_prior_data_dir and args.mjx_prior_weight != 0.0:
        physical_prior = mjxprior.MJXForelimbPrior.from_data_dir(args.mjx_prior_data_dir)
        print(
            "[mjx] loaded "
            f"{len(physical_prior.records)} marker-site records from {args.mjx_prior_data_dir}"
        )

    paw_indices, paw_groups = qc.collect_paw_vertex_indices(mesh_obj, False, 0.001)
    arm_indices, arm_groups = qc.collect_arm_vertex_indices(mesh_obj, 0.001)
    obstacles, missing_obstacles = qc.obstacle_records(args.obstacle or DEFAULT_OBSTACLES)
    if not obstacles:
        raise RuntimeError("No obstacle BVHs available.")

    target_vertex_count = len(mesh_obj.data.vertices)
    source_vertex_count = source_payload["mesh_vertex_count"]
    if source_vertex_count != target_vertex_count:
        raise RuntimeError(
            f"Source/target mesh topology mismatch: {source_vertex_count} vs {target_vertex_count}."
        )

    volume_reference = quality_action["arm_volume_reference"]
    accepted = 0
    rejected = 0
    samples = []
    for index, (_ratio, frame, side) in enumerate(targets, start=1):
        print(f"[refine] {index}/{len(targets)} frame={frame} side={side}")
        qc.set_frame(frame)
        source_frame = source_payload["frames"].get(int(frame))
        if source_frame is None:
            rejected += 1
            samples.append(
                {
                    "frame": float(frame),
                    "side": side,
                    "decision": "rejected",
                    "reason": "missing_source_frame",
                }
            )
            continue
        row = refine_frame_side(
            armature,
            mesh_obj,
            arm_indices,
            paw_indices,
            obstacles,
            side,
            frame,
            source_frame,
            volume_reference[side],
            args,
            physical_prior,
        )
        if row["decision"] == "accepted":
            accepted += 1
        else:
            rejected += 1
        samples.append(row)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.resolve()))
    report = {
        "source": {
            key: value
            for key, value in source_payload.items()
            if key not in ("frames",)
        },
        "target": {
            "blend": str(Path(args.target_blend).resolve()),
            "output": str(output_path.resolve()),
            "armature": armature.name,
            "mesh": mesh_obj.name,
            "action": action.name,
            "mesh_vertex_count": target_vertex_count,
            "arm_vertex_counts": {
                side: {segment: len(values) for segment, values in segments.items()}
                for side, segments in arm_indices.items()
            },
            "paw_vertex_counts": {side: len(values) for side, values in paw_indices.items()},
            "arm_groups": arm_groups,
            "paw_groups": paw_groups,
            "obstacles": [record["name"] for record in obstacles],
            "missing_obstacles": missing_obstacles,
        },
        "settings": vars(args),
        "target_count": len(targets),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "samples": samples,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_SOURCE_MESH_REFINE_BEGIN")
    print(
        json.dumps(
            {
                "target_count": len(targets),
                "accepted_count": accepted,
                "rejected_count": rejected,
                "output": str(output_path.resolve()),
                "report": str(report_path.resolve()),
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_SOURCE_MESH_REFINE_END")


if __name__ == "__main__":
    main()

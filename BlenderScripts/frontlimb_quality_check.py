import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


DEFAULT_OBSTACLES = [
    "Part_MouseCupSide.001",
    "Part_MouseCupSide.002",
    "Part_PrintedMouseCup",
]

SIDE_SUFFIXES = ("_L", "_R")
FINGERTIP_BONES = ("IndexFinger3", "MiddleFinger3", "RingFinger3", "PinkyFinger3")


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check migrated mouse forelimbs for cup/paw collisions and elbow collapse."
    )
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--all-mig-actions", action="store_true")
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--obstacle", action="append", default=[])
    parser.add_argument("--include-wrist", action="store_true")
    parser.add_argument("--min-vertex-weight", type=float, default=0.001)
    parser.add_argument("--paw-polygon-mode", choices=("any", "all"), default="any")
    parser.add_argument("--clearance-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-threshold", type=int, default=50)
    parser.add_argument("--collapse-ratio-threshold", type=float, default=0.45)
    parser.add_argument("--elbow-angle-threshold", type=float, default=35.0)
    parser.add_argument("--arm-volume-ratio-threshold", type=float, default=0.70)
    parser.add_argument("--arm-volume-bins", type=int, default=6)
    parser.add_argument("--arm-volume-trim", type=float, default=0.12)
    parser.add_argument("--include-frame-details", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args(args_after_dash())


def action_fcurves(action):
    fcurves = []
    if getattr(action, "fcurves", None) is not None:
        fcurves.extend(action.fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def action_frames(action):
    frames = set()
    for fcurve in action_fcurves(action):
        for key in fcurve.keyframe_points:
            frames.add(round(float(key.co.x), 6))
    if frames:
        return sorted(frames)

    start, end = action.frame_range
    return [float(frame) for frame in range(math.ceil(start), math.floor(end) + 1)]


def find_armature(name):
    if name:
        armature = bpy.data.objects.get(name)
        if armature is None:
            raise RuntimeError(f"Armature {name!r} not found.")
        return armature

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found.")
    with_action = [
        obj
        for obj in armatures
        if obj.animation_data and obj.animation_data.action
    ]
    return with_action[0] if with_action else armatures[0]


def actions_to_check(armature, requested, all_mig):
    if requested:
        return [bpy.data.actions[name] for name in requested]
    if all_mig:
        actions = sorted(
            [action for action in bpy.data.actions if action.name.startswith("MIG_")],
            key=lambda action: action.name,
        )
        if actions:
            return actions
    if armature.animation_data and armature.animation_data.action:
        return [armature.animation_data.action]
    actions = [action for action in bpy.data.actions if action.name.startswith("MIG_")]
    if actions:
        return [sorted(actions, key=lambda action: action.name)[0]]
    raise RuntimeError("No action found.")


def set_action(armature, action):
    armature.animation_data_create()
    armature.animation_data.action = action


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def sampled_frames(action, explicit_frames, sample_step):
    if explicit_frames:
        return sorted(explicit_frames)
    frames = action_frames(action)
    step = max(int(sample_step), 1)
    return frames[::step]


def evaluated_mesh_world(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        vertices = [eval_obj.matrix_world @ vertex.co for vertex in eval_mesh.vertices]
        polygons = [tuple(poly.vertices) for poly in eval_mesh.polygons if len(poly.vertices) >= 3]
    finally:
        eval_obj.to_mesh_clear()
    return vertices, polygons


def make_bvh(vertices, polygons):
    if not vertices or not polygons:
        return None
    return BVHTree.FromPolygons(vertices, polygons)


def is_paw_vertex_group(name, include_wrist):
    if not name.endswith(SIDE_SUFFIXES):
        return False
    base = name[:-2]
    if "Toe" in base or base.startswith("Ankle"):
        return False
    if "Finger" in base:
        return True
    return include_wrist and base.startswith("Wrist")


def collect_paw_vertex_indices(mesh_obj, include_wrist, min_weight):
    group_by_index = {}
    groups_by_side = {"L": [], "R": []}
    for group in mesh_obj.vertex_groups:
        if not is_paw_vertex_group(group.name, include_wrist):
            continue
        side = group.name[-1]
        group_by_index[group.index] = side
        groups_by_side[side].append(group.name)

    indices_by_side = {"L": set(), "R": set()}
    for vertex in mesh_obj.data.vertices:
        for membership in vertex.groups:
            side = group_by_index.get(membership.group)
            if side and membership.weight >= min_weight:
                indices_by_side[side].add(vertex.index)
    return indices_by_side, groups_by_side


def collect_arm_vertex_indices(mesh_obj, min_weight):
    groups = {
        "L": {"upper": [], "lower": [], "excluded": []},
        "R": {"upper": [], "lower": [], "excluded": []},
    }
    group_kind_by_index = {}
    for group in mesh_obj.vertex_groups:
        if not group.name.endswith(SIDE_SUFFIXES):
            continue
        side = group.name[-1]
        base = group.name[:-2]
        if base == "Scapula":
            group_kind_by_index[group.index] = (side, "upper")
            groups[side]["upper"].append(group.name)
        elif base == "Wrist":
            group_kind_by_index[group.index] = (side, "lower")
            groups[side]["lower"].append(group.name)
        elif "Finger" in base:
            group_kind_by_index[group.index] = (side, "excluded")
            groups[side]["excluded"].append(group.name)

    indices = {"L": {"upper": set(), "lower": set()}, "R": {"upper": set(), "lower": set()}}
    for vertex in mesh_obj.data.vertices:
        excluded_sides = set()
        memberships = []
        for membership in vertex.groups:
            mapped = group_kind_by_index.get(membership.group)
            if not mapped or membership.weight < min_weight:
                continue
            side, kind = mapped
            if kind == "excluded":
                excluded_sides.add(side)
            else:
                memberships.append((side, kind))
        for side, kind in memberships:
            if side not in excluded_sides:
                indices[side][kind].add(vertex.index)
    return indices, groups


def submesh_for_indices(vertices, polygons, selected, mode):
    selected = set(selected)
    index_map = {}
    sub_vertices = []
    sub_polygons = []

    for polygon in polygons:
        poly_indices = list(polygon)
        hits = [index in selected for index in poly_indices]
        if (mode == "all" and not all(hits)) or (mode == "any" and not any(hits)):
            continue

        remapped = []
        for index in poly_indices:
            if index not in index_map:
                index_map[index] = len(sub_vertices)
                sub_vertices.append(vertices[index])
            remapped.append(index_map[index])
        sub_polygons.append(tuple(remapped))

    return sub_vertices, sub_polygons


def obstacle_records(names):
    records = []
    missing = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        vertices, polygons = evaluated_mesh_world(obj)
        bvh = make_bvh(vertices, polygons)
        if bvh is None:
            missing.append(name)
            continue
        records.append(
            {
                "name": name,
                "object": obj,
                "vertex_count": len(vertices),
                "polygon_count": len(polygons),
                "bvh": bvh,
            }
        )
    return records, missing


def closest_point_to_obstacles(points, obstacles):
    best = None
    for local_index, point in points:
        for obstacle in obstacles:
            nearest = obstacle["bvh"].find_nearest(point)
            if nearest is None or nearest[0] is None:
                continue
            location, normal, face_index, distance = nearest
            row = {
                "distance": float(distance),
                "obstacle": obstacle["name"],
                "vertex_index": int(local_index),
                "nearest_point": [float(value) for value in location],
                "nearest_normal": [float(value) for value in normal],
                "face_index": int(face_index),
            }
            if best is None or row["distance"] < best["distance"]:
                best = row
    return best


def paw_clearance_for_side(vertices, polygons, selected_indices, side, obstacles, polygon_mode):
    selected_points = [
        (index, vertices[index])
        for index in sorted(selected_indices)
        if index < len(vertices)
    ]
    sub_vertices, sub_polygons = submesh_for_indices(
        vertices, polygons, [index for index, _ in selected_points], polygon_mode
    )
    paw_bvh = make_bvh(sub_vertices, sub_polygons)

    overlap_by_obstacle = {}
    total_overlap_count = 0
    if paw_bvh is not None:
        for obstacle in obstacles:
            pairs = paw_bvh.overlap(obstacle["bvh"])
            overlap_count = len(pairs)
            overlap_by_obstacle[obstacle["name"]] = overlap_count
            total_overlap_count += overlap_count

    best = closest_point_to_obstacles(selected_points, obstacles)
    return {
        "side": side,
        "selected_vertex_count": len(selected_points),
        "selected_polygon_count": len(sub_polygons),
        "min_distance": None if best is None else best["distance"],
        "nearest": best,
        "overlap_count": int(total_overlap_count),
        "overlap_by_obstacle": overlap_by_obstacle,
    }


def pose_bone_head_world(armature, bone_name):
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return armature.matrix_world @ pose_bone.matrix.translation


def pose_bone_tail_world(armature, bone_name):
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return armature.matrix_world @ (
        pose_bone.matrix @ Vector((0.0, pose_bone.bone.length, 0.0))
    )


def angle_degrees(a, b):
    if a.length == 0.0 or b.length == 0.0:
        return None
    return math.degrees(a.angle(b))


def forelimb_pose_metrics(armature, side):
    scapula = f"Scapula_{side}"
    wrist = f"Wrist_{side}"
    fingers = f"Fingers1_{side}"

    shoulder = pose_bone_head_world(armature, scapula)
    elbow = pose_bone_tail_world(armature, scapula)
    palm = pose_bone_tail_world(armature, wrist)
    if palm is None:
        palm = pose_bone_head_world(armature, fingers)

    if shoulder is None or elbow is None or palm is None:
        return {"side": side, "available": False}

    upper = elbow - shoulder
    lower = palm - elbow
    reach = palm - shoulder
    chain_length = upper.length + lower.length
    reach_ratio = reach.length / chain_length if chain_length else None
    elbow_angle = angle_degrees(shoulder - elbow, palm - elbow)

    fingertips = []
    for base in FINGERTIP_BONES:
        tip = pose_bone_tail_world(armature, f"{base}_{side}")
        if tip is not None:
            fingertips.append(tip)
    fingertip_center = None
    if fingertips:
        fingertip_center = sum(fingertips, Vector()) / len(fingertips)

    return {
        "side": side,
        "available": True,
        "shoulder": [float(value) for value in shoulder],
        "elbow": [float(value) for value in elbow],
        "palm": [float(value) for value in palm],
        "fingertip_center": None
        if fingertip_center is None
        else [float(value) for value in fingertip_center],
        "upper_length": float(upper.length),
        "lower_length": float(lower.length),
        "reach": float(reach.length),
        "reach_ratio": None if reach_ratio is None else float(reach_ratio),
        "elbow_angle_deg": None if elbow_angle is None else float(elbow_angle),
    }


def forelimb_integrity_metrics(armature, side):
    prefixes = [
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
    rows = []
    for prefix in prefixes:
        pose_bone = armature.pose.bones.get(f"{prefix}_{side}")
        if pose_bone is None:
            continue
        scale_deviation = max(abs(float(value) - 1.0) for value in pose_bone.scale)
        rows.append(
            {
                "bone": pose_bone.name,
                "location_length": float(pose_bone.location.length),
                "max_scale_deviation": float(scale_deviation),
            }
        )
    return {
        "side": side,
        "max_location_length": max((row["location_length"] for row in rows), default=0.0),
        "max_scale_deviation": max((row["max_scale_deviation"] for row in rows), default=0.0),
    }


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * fraction)), 0), len(ordered) - 1)
    return ordered[index]


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def volume_proxy_for_segment(vertices, selected_indices, start, end, bins, trim):
    axis = end - start
    length = axis.length
    if length < 1.0e-8:
        return {
            "available": False,
            "point_count": 0,
            "volume_proxy": None,
            "min_bin_radius": None,
            "max_bin_radius": None,
        }

    axis_dir = axis.normalized()
    trim = max(0.0, min(trim, 0.45))
    bin_count = max(int(bins), 1)
    radial_bins = [[] for _ in range(bin_count)]
    kept_points = 0
    for index in selected_indices:
        if index >= len(vertices):
            continue
        relative = vertices[index] - start
        t = relative.dot(axis_dir) / length
        if t < trim or t > 1.0 - trim:
            continue
        radial = (relative - axis_dir * relative.dot(axis_dir)).length
        normalized = (t - trim) / max(1.0 - 2.0 * trim, 1.0e-8)
        bin_index = min(max(int(normalized * bin_count), 0), bin_count - 1)
        radial_bins[bin_index].append(radial)
        kept_points += 1

    radii = []
    for values in radial_bins:
        if not values:
            continue
        # The 70th percentile is more stable than max radius but still catches flattening.
        radii.append(percentile(values, 0.70))
    if not radii:
        return {
            "available": False,
            "point_count": kept_points,
            "volume_proxy": None,
            "min_bin_radius": None,
            "max_bin_radius": None,
        }

    bin_length = length * max(1.0 - 2.0 * trim, 1.0e-8) / len(radii)
    volume = sum(math.pi * radius * radius * bin_length for radius in radii)
    return {
        "available": True,
        "point_count": kept_points,
        "volume_proxy": float(volume),
        "min_bin_radius": float(min(radii)),
        "max_bin_radius": float(max(radii)),
        "bin_radii": [float(radius) for radius in radii],
    }


def forelimb_volume_metrics(vertices, armature, side, arm_indices, bins, trim):
    shoulder = pose_bone_head_world(armature, f"Scapula_{side}")
    elbow = pose_bone_tail_world(armature, f"Scapula_{side}")
    palm = pose_bone_tail_world(armature, f"Wrist_{side}")
    if shoulder is None or elbow is None or palm is None:
        return {"side": side, "available": False}

    upper = volume_proxy_for_segment(
        vertices,
        arm_indices[side]["upper"],
        shoulder,
        elbow,
        bins,
        trim,
    )
    lower = volume_proxy_for_segment(
        vertices,
        arm_indices[side]["lower"],
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
        "side": side,
        "available": available,
        "upper": upper,
        "lower": lower,
        "total_volume_proxy": float(total) if available else None,
    }


def warning_spans(warning_frames):
    spans = []
    current = None
    for row in warning_frames:
        frame = row["frame"]
        flags = set(row["flags"])
        if (
            current is None
            or frame > current["end_frame"] + 1.001
        ):
            if current is not None:
                current["flags"] = sorted(current["flags"])
                spans.append(current)
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "frame_count": 1,
                "flags": flags,
            }
        else:
            current["end_frame"] = frame
            current["frame_count"] += 1
            current["flags"].update(flags)
    if current is not None:
        current["flags"] = sorted(current["flags"])
        spans.append(current)
    return spans


def flag_counts(warning_frames):
    counts = {}
    for row in warning_frames:
        for flag in row["flags"]:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def spans_for_frames(frames):
    spans = []
    current = None
    for frame in sorted(frames):
        if current is None or frame > current["end_frame"] + 1.001:
            if current is not None:
                spans.append(current)
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "frame_count": 1,
            }
        else:
            current["end_frame"] = frame
            current["frame_count"] += 1
    if current is not None:
        spans.append(current)
    return spans


def flag_spans(warning_frames):
    frames_by_flag = {}
    for row in warning_frames:
        for flag in row["flags"]:
            frames_by_flag.setdefault(flag, []).append(row["frame"])
    return {
        flag: spans_for_frames(frames)
        for flag, frames in sorted(frames_by_flag.items(), key=lambda item: item[0])
    }


def evaluate_action(action, armature, mesh_obj, frames, selected_indices, arm_indices, obstacles, args):
    set_action(armature, action)

    frame_rows = []
    clearance_rows = []
    collapse_rows = []
    integrity_rows = []
    volume_rows = []

    for frame in frames:
        set_frame(frame)
        vertices, polygons = evaluated_mesh_world(mesh_obj)
        if len(vertices) != len(mesh_obj.data.vertices):
            raise RuntimeError(
                f"Evaluated mesh vertex count changed from {len(mesh_obj.data.vertices)} "
                f"to {len(vertices)}; vertex-group based paw checks need stable topology."
            )

        paw_by_side = {}
        pose_by_side = {}
        integrity_by_side = {}
        volume_by_side = {}
        flags = []

        for side in ("L", "R"):
            clearance = paw_clearance_for_side(
                vertices,
                polygons,
                selected_indices[side],
                side,
                obstacles,
                args.paw_polygon_mode,
            )
            paw_by_side[side] = clearance
            if clearance["min_distance"] is not None:
                clearance_row = {
                    "action": action.name,
                    "frame": float(frame),
                    "side": side,
                    "min_distance": clearance["min_distance"],
                    "nearest_obstacle": clearance["nearest"]["obstacle"],
                    "overlap_count": clearance["overlap_count"],
                    "selected_vertex_count": clearance["selected_vertex_count"],
                    "selected_polygon_count": clearance["selected_polygon_count"],
                }
                clearance_rows.append(clearance_row)
                if clearance["overlap_count"] >= args.overlap_threshold:
                    flags.append(f"{side}_paw_cup_overlap")
                elif clearance["min_distance"] < args.clearance_threshold:
                    flags.append(f"{side}_paw_cup_clearance")

            pose = forelimb_pose_metrics(armature, side)
            pose_by_side[side] = pose
            if pose.get("available"):
                collapse_row = {
                    "action": action.name,
                    "frame": float(frame),
                    "side": side,
                    "reach_ratio": pose["reach_ratio"],
                    "elbow_angle_deg": pose["elbow_angle_deg"],
                    "upper_length": pose["upper_length"],
                    "lower_length": pose["lower_length"],
                }
                collapse_rows.append(collapse_row)
                if (
                    pose["reach_ratio"] is not None
                    and pose["reach_ratio"] < args.collapse_ratio_threshold
                ) or (
                    pose["elbow_angle_deg"] is not None
                    and pose["elbow_angle_deg"] < args.elbow_angle_threshold
                ):
                    flags.append(f"{side}_forelimb_collapse")

            integrity = forelimb_integrity_metrics(armature, side)
            integrity_by_side[side] = integrity
            if (
                integrity["max_location_length"] > 1.0e-4
                or integrity["max_scale_deviation"] > 1.0e-4
            ):
                integrity_rows.append(
                    {
                        "action": action.name,
                        "frame": float(frame),
                        "side": side,
                        **integrity,
                    }
                )
                flags.append(f"{side}_forelimb_location_or_scale")

            volume = forelimb_volume_metrics(
                vertices,
                armature,
                side,
                arm_indices,
                args.arm_volume_bins,
                args.arm_volume_trim,
            )
            volume_by_side[side] = volume
            if volume.get("available"):
                volume_rows.append(
                    {
                        "action": action.name,
                        "frame": float(frame),
                        "side": side,
                        "upper_volume_proxy": volume["upper"]["volume_proxy"],
                        "lower_volume_proxy": volume["lower"]["volume_proxy"],
                        "total_volume_proxy": volume["total_volume_proxy"],
                    }
                )

        frame_rows.append(
            {
                "frame": float(frame),
                "flags": sorted(set(flags)),
                "paw_clearance": paw_by_side,
                "forelimb_pose": pose_by_side,
                "forelimb_integrity": integrity_by_side,
                "forelimb_volume": volume_by_side,
            }
        )

    volume_reference = {}
    for side in ("L", "R"):
        volume_reference[side] = {}
        for segment_key, row_key in (
            ("upper", "upper_volume_proxy"),
            ("lower", "lower_volume_proxy"),
            ("total", "total_volume_proxy"),
        ):
            values = [
                row[row_key]
                for row in volume_rows
                if row["side"] == side and row[row_key] is not None
            ]
            volume_reference[side][segment_key] = median(values)

    volume_warning_rows = []
    for frame_row in frame_rows:
        for side in ("L", "R"):
            volume = frame_row["forelimb_volume"].get(side)
            if not volume or not volume.get("available"):
                continue
            segment_ratios = {}
            worst_ratio = None
            worst_segment = None
            for segment_key, value in (
                ("upper", volume["upper"]["volume_proxy"]),
                ("lower", volume["lower"]["volume_proxy"]),
                ("total", volume["total_volume_proxy"]),
            ):
                reference = volume_reference[side].get(segment_key)
                ratio = None
                if reference and value is not None:
                    ratio = value / reference
                segment_ratios[segment_key] = None if ratio is None else float(ratio)
                if ratio is not None and (worst_ratio is None or ratio < worst_ratio):
                    worst_ratio = ratio
                    worst_segment = segment_key

            volume["volume_reference"] = {
                key: None if value is None else float(value)
                for key, value in volume_reference[side].items()
            }
            volume["volume_ratio"] = segment_ratios
            volume["worst_volume_ratio"] = None if worst_ratio is None else float(worst_ratio)
            volume["worst_volume_segment"] = worst_segment

            if worst_ratio is not None and worst_ratio < args.arm_volume_ratio_threshold:
                frame_row["flags"].append(f"{side}_arm_volume_collapse")
                frame_row["flags"] = sorted(set(frame_row["flags"]))
            volume_warning_rows.append(
                {
                    "action": action.name,
                    "frame": frame_row["frame"],
                    "side": side,
                    "worst_volume_ratio": None if worst_ratio is None else float(worst_ratio),
                    "worst_volume_segment": worst_segment,
                    "upper_volume_ratio": segment_ratios["upper"],
                    "lower_volume_ratio": segment_ratios["lower"],
                    "total_volume_ratio": segment_ratios["total"],
                    "upper_volume_proxy": volume["upper"]["volume_proxy"],
                    "lower_volume_proxy": volume["lower"]["volume_proxy"],
                    "total_volume_proxy": volume["total_volume_proxy"],
                }
            )

    clearance_rows.sort(
        key=lambda row: (
            0 if row["overlap_count"] > 0 else 1,
            -row["overlap_count"],
            row["min_distance"],
        )
    )
    collapse_rows.sort(
        key=lambda row: (
            row["reach_ratio"] if row["reach_ratio"] is not None else 999.0,
            row["elbow_angle_deg"] if row["elbow_angle_deg"] is not None else 999.0,
        )
    )
    volume_warning_rows.sort(
        key=lambda row: (
            row["worst_volume_ratio"]
            if row["worst_volume_ratio"] is not None
            else 999.0
        )
    )

    warning_frames = [
        {"frame": row["frame"], "flags": row["flags"]}
        for row in frame_rows
        if row["flags"]
    ]

    result = {
        "action": action.name,
        "frame_count": len(frames),
        "warning_frame_count": len(warning_frames),
        "warning_frames": warning_frames[: args.top],
        "warning_spans": warning_spans(warning_frames),
        "flag_counts": flag_counts(warning_frames),
        "flag_spans": flag_spans(warning_frames),
        "worst_clearance": clearance_rows[: args.top],
        "worst_collapse": collapse_rows[: args.top],
        "arm_volume_reference": volume_reference,
        "worst_arm_volume": volume_warning_rows[: args.top],
        "location_or_scale_warnings": integrity_rows[: args.top],
    }
    if args.include_frame_details:
        result["frames"] = frame_rows
    return result


def main():
    args = parse_args()
    armature = find_armature(args.armature)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if mesh_obj is None:
        raise RuntimeError(f"Mesh {args.mesh!r} not found.")

    selected_indices, groups_by_side = collect_paw_vertex_indices(
        mesh_obj, args.include_wrist, args.min_vertex_weight
    )
    arm_indices, arm_groups_by_side = collect_arm_vertex_indices(
        mesh_obj, args.min_vertex_weight
    )
    obstacle_names = args.obstacle or [
        name for name in DEFAULT_OBSTACLES if bpy.data.objects.get(name) is not None
    ]
    obstacles, missing_obstacles = obstacle_records(obstacle_names)
    if not obstacles:
        raise RuntimeError("No usable obstacle meshes found.")

    actions = actions_to_check(armature, args.action, args.all_mig_actions)
    results = []
    for action in actions:
        frames = sampled_frames(action, args.frame, args.sample_step)
        results.append(
            evaluate_action(
                action,
                armature,
                mesh_obj,
                frames,
                selected_indices,
                arm_indices,
                obstacles,
                args,
            )
        )

    report = {
        "file": bpy.data.filepath,
        "armature": armature.name,
        "mesh": mesh_obj.name,
        "settings": {
            "obstacles": [obstacle["name"] for obstacle in obstacles],
            "missing_obstacles": missing_obstacles,
            "include_wrist": args.include_wrist,
            "min_vertex_weight": args.min_vertex_weight,
            "paw_polygon_mode": args.paw_polygon_mode,
            "clearance_threshold": args.clearance_threshold,
            "overlap_threshold": args.overlap_threshold,
            "collapse_ratio_threshold": args.collapse_ratio_threshold,
            "elbow_angle_threshold": args.elbow_angle_threshold,
            "arm_volume_ratio_threshold": args.arm_volume_ratio_threshold,
            "arm_volume_bins": args.arm_volume_bins,
            "arm_volume_trim": args.arm_volume_trim,
            "sample_step": args.sample_step,
        },
        "paw_vertex_groups": groups_by_side,
        "paw_vertex_counts": {
            side: len(indices) for side, indices in selected_indices.items()
        },
        "arm_vertex_groups": arm_groups_by_side,
        "arm_vertex_counts": {
            side: {
                segment: len(indices)
                for segment, indices in segment_indices.items()
            }
            for side, segment_indices in arm_indices.items()
        },
        "obstacle_meshes": [
            {
                "name": obstacle["name"],
                "vertex_count": obstacle["vertex_count"],
                "polygon_count": obstacle["polygon_count"],
            }
            for obstacle in obstacles
        ],
        "actions": results,
    }

    text = json.dumps(report, indent=2)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print("MOUSE_GEO_FORGE_FRONTLIMB_QUALITY_BEGIN")
    print(text)
    print("MOUSE_GEO_FORGE_FRONTLIMB_QUALITY_END")


if __name__ == "__main__":
    main()

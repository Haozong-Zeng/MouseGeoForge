import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


DEFAULT_OBSTACLES = [
    "Part_PrintedMouseCup",
]

FINGER_GROUP_PREFIXES = (
    "Fingers1",
    "IndexFinger",
    "MiddleFinger",
    "RingFinger",
    "PinkyFinger",
)
SIDE_SUFFIXES = ("_L", "_R")


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--all-mig-actions", action="store_true")
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--obstacle", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--step", type=float, action="append", default=[])
    parser.add_argument("--min-reach-ratio", type=float, default=0.57)
    parser.add_argument("--clearance-threshold", type=float, default=0.25)
    parser.add_argument("--score-overlap-weight", type=float, default=100000.0)
    parser.add_argument("--arm-volume-ratio-threshold", type=float, default=0.72)
    parser.add_argument("--arm-volume-weight", type=float, default=350000.0)
    parser.add_argument("--arm-volume-bins", type=int, default=6)
    parser.add_argument("--arm-volume-trim", type=float, default=0.12)
    parser.add_argument("--score-reach-ratio-threshold", type=float, default=0.62)
    parser.add_argument("--score-reach-weight", type=float, default=250000.0)
    parser.add_argument("--wrist-roll-deg", type=float, action="append", default=[])
    parser.add_argument("--scapula-roll-deg", type=float, action="append", default=[])
    parser.add_argument("--keyframe-all-forelimb-bones", action="store_true")
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


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def actions_to_refine(armature, requested, all_mig):
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
    raise RuntimeError("No action found.")


def set_action(armature, action):
    armature.animation_data_create()
    armature.animation_data.action = action


def column_vector(matrix, index):
    return Vector((matrix[0][index], matrix[1][index], matrix[2][index]))


def matrix_with_columns(x_vec, y_vec, z_vec, translation):
    matrix = Matrix.Identity(4)
    for row, value in enumerate(x_vec):
        matrix[row][0] = value
    for row, value in enumerate(y_vec):
        matrix[row][1] = value
    for row, value in enumerate(z_vec):
        matrix[row][2] = value
    matrix[0][3] = translation.x
    matrix[1][3] = translation.y
    matrix[2][3] = translation.z
    return matrix


def aim_bone_tail_at(matrix, bone, tail_position):
    head = matrix.translation.copy()
    segment = tail_position - head
    if segment.length < 1e-8 or bone.length < 1e-8:
        return matrix

    y_dir = segment.normalized()
    x_vec = column_vector(matrix, 0)
    z_vec = column_vector(matrix, 2)
    x_scale = max(x_vec.length, 1e-8)
    y_scale = max(column_vector(matrix, 1).length, 1e-8)
    z_scale = max(z_vec.length, 1e-8)

    x_projected = x_vec - y_dir * x_vec.dot(y_dir)
    if x_projected.length < 1e-8:
        x_projected = z_vec.cross(y_dir)
    if x_projected.length < 1e-8:
        fallback = Vector((1.0, 0.0, 0.0))
        if abs(fallback.dot(y_dir)) > 0.95:
            fallback = Vector((0.0, 0.0, 1.0))
        x_projected = fallback - y_dir * fallback.dot(y_dir)

    x_dir = x_projected.normalized()
    z_dir = x_dir.cross(y_dir).normalized()
    if z_dir.dot(z_vec) < 0.0:
        x_dir.negate()
        z_dir = x_dir.cross(y_dir).normalized()

    return matrix_with_columns(x_dir * x_scale, y_dir * y_scale, z_dir * z_scale, head)


def set_pose_matrix_via_basis(armature, bone_name, desired_pose_matrix):
    pb = armature.pose.bones[bone_name]
    bone = pb.bone
    if bone.parent:
        parent_pb = armature.pose.bones[bone.parent.name]
        basis = bone.convert_local_to_pose(
            desired_pose_matrix,
            bone.matrix_local,
            parent_matrix=parent_pb.matrix,
            parent_matrix_local=bone.parent.matrix_local,
            invert=True,
        )
    else:
        basis = bone.convert_local_to_pose(
            desired_pose_matrix,
            bone.matrix_local,
            invert=True,
        )
    pb.matrix_basis = basis


def solve_two_bone_joint(root, preferred_joint, target, first_length, second_length, min_reach_ratio):
    to_target = target - root
    distance = to_target.length
    if distance < 1e-8:
        return preferred_joint.copy(), root.copy()

    direction = to_target.normalized()
    max_reach = max(first_length + second_length - 1e-6, 1e-6)
    min_reach = max(abs(first_length - second_length) + 1e-6, 1e-6)
    if min_reach_ratio > 0.0:
        min_reach = min(max(min_reach, (first_length + second_length) * min_reach_ratio), max_reach)
    clamped_distance = min(max(distance, min_reach), max_reach)
    clamped_target = root + direction * clamped_distance

    x_distance = (
        first_length * first_length
        + clamped_distance * clamped_distance
        - second_length * second_length
    ) / (2.0 * clamped_distance)
    height_sq = max(first_length * first_length - x_distance * x_distance, 0.0)

    pole = preferred_joint - (root + direction * (preferred_joint - root).dot(direction))
    if pole.length < 1e-8:
        fallback = Vector((0.0, 0.0, 1.0))
        if abs(fallback.dot(direction)) > 0.95:
            fallback = Vector((1.0, 0.0, 0.0))
        pole = fallback - direction * fallback.dot(direction)
    pole.normalize()

    joint = root + direction * x_distance + pole * math.sqrt(height_sq)
    return joint, clamped_target


def bone_tail_world(armature, bone_name):
    pb = armature.pose.bones[bone_name]
    return armature.matrix_world @ (pb.matrix @ Vector((0.0, pb.bone.length, 0.0)))


def bone_head_world(armature, bone_name):
    pb = armature.pose.bones.get(bone_name)
    if pb is None:
        return None
    return armature.matrix_world @ pb.matrix.translation


def safe_bone_tail_world(armature, bone_name):
    pb = armature.pose.bones.get(bone_name)
    if pb is None:
        return None
    return armature.matrix_world @ (pb.matrix @ Vector((0.0, pb.bone.length, 0.0)))


def forelimb_reach_ratio(armature, side):
    shoulder = bone_head_world(armature, f"Scapula_{side}")
    elbow = safe_bone_tail_world(armature, f"Scapula_{side}")
    palm = safe_bone_tail_world(armature, f"Wrist_{side}")
    if shoulder is None or elbow is None or palm is None:
        return None
    chain_length = (elbow - shoulder).length + (palm - elbow).length
    if chain_length < 1.0e-8:
        return None
    return float((palm - shoulder).length / chain_length)


def set_forelimb_palm_world(armature, side, target_palm_world, min_reach_ratio):
    scapula_name = f"Scapula_{side}"
    wrist_name = f"Wrist_{side}"
    if scapula_name not in armature.pose.bones or wrist_name not in armature.pose.bones:
        return False

    world_to_arm = armature.matrix_world.inverted()
    scapula_pb = armature.pose.bones[scapula_name]
    wrist_pb = armature.pose.bones[wrist_name]

    shoulder = scapula_pb.matrix.translation.copy()
    preferred_joint = world_to_arm @ bone_tail_world(armature, scapula_name)
    palm_target = world_to_arm @ target_palm_world
    first_length = scapula_pb.bone.length
    second_length = wrist_pb.bone.length

    wrist_position, palm_position = solve_two_bone_joint(
        shoulder,
        preferred_joint,
        palm_target,
        first_length,
        second_length,
        min_reach_ratio,
    )

    scapula_matrix = aim_bone_tail_at(
        scapula_pb.matrix.copy(),
        scapula_pb.bone,
        wrist_position,
    )
    set_pose_matrix_via_basis(armature, scapula_name, scapula_matrix)
    bpy.context.view_layer.update()

    wrist_matrix = wrist_pb.matrix.copy()
    wrist_matrix.translation = wrist_position
    wrist_matrix = aim_bone_tail_at(wrist_matrix, wrist_pb.bone, palm_position)
    set_pose_matrix_via_basis(armature, wrist_name, wrist_matrix)

    scapula_pb.scale = (1.0, 1.0, 1.0)
    wrist_pb.scale = (1.0, 1.0, 1.0)
    scapula_pb.location = (0.0, 0.0, 0.0)
    wrist_pb.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return True


def roll_pose_bone_around_local_y(armature, bone_name, angle_degrees):
    if abs(angle_degrees) < 1.0e-8 or bone_name not in armature.pose.bones:
        return

    pb = armature.pose.bones[bone_name]
    matrix = pb.matrix.copy()
    x_vec = column_vector(matrix, 0)
    y_vec = column_vector(matrix, 1)
    z_vec = column_vector(matrix, 2)
    x_scale = max(x_vec.length, 1.0e-8)
    y_scale = max(y_vec.length, 1.0e-8)
    z_scale = max(z_vec.length, 1.0e-8)
    x_dir = x_vec.normalized()
    z_dir = z_vec.normalized()
    angle = math.radians(angle_degrees)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rolled_x = (x_dir * cos_angle + z_dir * sin_angle).normalized() * x_scale
    rolled_z = (z_dir * cos_angle - x_dir * sin_angle).normalized() * z_scale
    rolled = matrix_with_columns(rolled_x, y_vec.normalized() * y_scale, rolled_z, matrix.translation)
    set_pose_matrix_via_basis(armature, bone_name, rolled)
    bpy.context.view_layer.update()


def mesh_bvh_world(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        vertices = [eval_obj.matrix_world @ vertex.co for vertex in eval_mesh.vertices]
        polygons = [tuple(poly.vertices) for poly in eval_mesh.polygons if len(poly.vertices) >= 3]
    finally:
        eval_obj.to_mesh_clear()
    return {
        "name": obj.name,
        "bvh": BVHTree.FromPolygons(vertices, polygons),
    }


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
        return {"available": False, "point_count": 0, "volume_proxy": None}

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

    radii = [percentile(values, 0.70) for values in radial_bins if values]
    if not radii:
        return {"available": False, "point_count": kept_points, "volume_proxy": None}

    bin_length = length * max(1.0 - 2.0 * trim, 1.0e-8) / len(radii)
    volume = sum(math.pi * radius * radius * bin_length for radius in radii)
    return {
        "available": True,
        "point_count": kept_points,
        "volume_proxy": float(volume),
        "min_bin_radius": float(min(radii)),
        "max_bin_radius": float(max(radii)),
    }


def forelimb_volume_metrics(vertices, armature, side, arm_indices, bins, trim):
    shoulder = bone_head_world(armature, f"Scapula_{side}")
    elbow = safe_bone_tail_world(armature, f"Scapula_{side}")
    palm = safe_bone_tail_world(armature, f"Wrist_{side}")
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


def build_volume_references(action_list, armature, mesh_obj, arm_indices, args):
    references = {}
    for action in action_list:
        set_action(armature, action)
        rows = {"L": {"upper": [], "lower": [], "total": []}, "R": {"upper": [], "lower": [], "total": []}}
        for frame in action_frames(action):
            set_frame(frame)
            vertices, _polygons = evaluated_mesh_world(mesh_obj)
            for side in ("L", "R"):
                volume = forelimb_volume_metrics(
                    vertices,
                    armature,
                    side,
                    arm_indices,
                    args.arm_volume_bins,
                    args.arm_volume_trim,
                )
                if not volume.get("available"):
                    continue
                for key, source in (
                    ("upper", volume["upper"].get("volume_proxy")),
                    ("lower", volume["lower"].get("volume_proxy")),
                    ("total", volume.get("total_volume_proxy")),
                ):
                    if source is not None:
                        rows[side][key].append(source)
        references[action.name] = {
            side: {segment: median(values) for segment, values in segments.items()}
            for side, segments in rows.items()
        }
    return references


def is_paw_group(name):
    if not name.endswith(("_L", "_R")):
        return False
    base = name[:-2]
    return any(base.startswith(prefix) for prefix in FINGER_GROUP_PREFIXES)


def collect_paw_indices(mesh_obj):
    group_side_by_index = {}
    for group in mesh_obj.vertex_groups:
        if is_paw_group(group.name):
            group_side_by_index[group.index] = group.name[-1]

    indices = {"L": set(), "R": set()}
    for vertex in mesh_obj.data.vertices:
        for membership in vertex.groups:
            side = group_side_by_index.get(membership.group)
            if side and membership.weight > 0.001:
                indices[side].add(vertex.index)
    return indices


def collect_arm_indices(mesh_obj, min_weight=0.001):
    group_kind_by_index = {}
    groups = {
        "L": {"upper": [], "lower": [], "excluded": []},
        "R": {"upper": [], "lower": [], "excluded": []},
    }
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


def submesh_for_indices(vertices, polygons, selected):
    selected = set(selected)
    index_map = {}
    sub_vertices = []
    sub_polygons = []
    for polygon in polygons:
        if not any(index in selected for index in polygon):
            continue
        remapped = []
        for index in polygon:
            if index not in index_map:
                index_map[index] = len(sub_vertices)
                sub_vertices.append(vertices[index])
            remapped.append(index_map[index])
        sub_polygons.append(tuple(remapped))
    return sub_vertices, sub_polygons


def nearest_distance(point, obstacle_bvhs):
    best = None
    for obstacle in obstacle_bvhs:
        nearest = obstacle["bvh"].find_nearest(point)
        if nearest is None or nearest[0] is None:
            continue
        distance = nearest[3]
        if best is None or distance < best:
            best = distance
    return best


def score_side(
    armature,
    mesh_obj,
    selected_indices,
    arm_indices,
    volume_reference,
    side,
    obstacle_bvhs,
    args,
):
    vertices, polygons = evaluated_mesh_world(mesh_obj)
    sub_vertices, sub_polygons = submesh_for_indices(vertices, polygons, selected_indices[side])
    if not sub_vertices or not sub_polygons:
        return {"score": float("inf"), "overlap_count": 0, "min_distance": None}

    paw_bvh = BVHTree.FromPolygons(sub_vertices, sub_polygons)
    overlap_count = 0
    for obstacle in obstacle_bvhs:
        overlap_count += len(paw_bvh.overlap(obstacle["bvh"]))

    min_distance = None
    for index in selected_indices[side]:
        if index >= len(vertices):
            continue
        distance = nearest_distance(vertices[index], obstacle_bvhs)
        if distance is not None and (min_distance is None or distance < min_distance):
            min_distance = distance

    clearance_penalty = 0.0
    if min_distance is not None and min_distance < args.clearance_threshold:
        clearance_penalty = args.clearance_threshold - min_distance

    volume = forelimb_volume_metrics(
        vertices,
        armature,
        side,
        arm_indices,
        args.arm_volume_bins,
        args.arm_volume_trim,
    )
    volume_ratios = {}
    worst_volume_ratio = None
    worst_volume_segment = None
    volume_penalty = 0.0
    if volume.get("available") and volume_reference:
        for segment, value in (
            ("upper", volume["upper"].get("volume_proxy")),
            ("lower", volume["lower"].get("volume_proxy")),
            ("total", volume.get("total_volume_proxy")),
        ):
            reference = volume_reference.get(segment)
            ratio = None
            if reference and value is not None:
                ratio = float(value / reference)
                if worst_volume_ratio is None or ratio < worst_volume_ratio:
                    worst_volume_ratio = ratio
                    worst_volume_segment = segment
                if ratio < args.arm_volume_ratio_threshold:
                    volume_penalty += (args.arm_volume_ratio_threshold - ratio) * args.arm_volume_weight
            volume_ratios[segment] = ratio

    reach_ratio = forelimb_reach_ratio(armature, side)
    reach_penalty = 0.0
    if reach_ratio is not None and reach_ratio < args.score_reach_ratio_threshold:
        reach_penalty = (args.score_reach_ratio_threshold - reach_ratio) * args.score_reach_weight

    score = (
        overlap_count * args.score_overlap_weight
        + clearance_penalty
        + volume_penalty
        + reach_penalty
    )
    return {
        "score": float(score),
        "overlap_count": int(overlap_count),
        "min_distance": None if min_distance is None else float(min_distance),
        "clearance_penalty": float(clearance_penalty),
        "volume_penalty": float(volume_penalty),
        "volume_ratios": volume_ratios,
        "worst_volume_ratio": None if worst_volume_ratio is None else float(worst_volume_ratio),
        "worst_volume_segment": worst_volume_segment,
        "reach_ratio": None if reach_ratio is None else float(reach_ratio),
        "reach_penalty": float(reach_penalty),
    }


def forelimb_bones(side):
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
    return [f"{prefix}_{side}" for prefix in prefixes]


def candidate_vectors(steps):
    directions = [
        Vector((0.0, 0.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
        Vector((0.0, -1.0, 0.0)),
        Vector((0.0, 0.0, 1.0)),
        Vector((0.0, 0.0, -1.0)),
        Vector((0.0, 1.0, 1.0)).normalized(),
        Vector((0.0, 1.0, -1.0)).normalized(),
        Vector((0.0, -1.0, 1.0)).normalized(),
        Vector((0.0, -1.0, -1.0)).normalized(),
        Vector((1.0, 0.0, 0.0)),
        Vector((-1.0, 0.0, 0.0)),
    ]
    vectors = [Vector((0.0, 0.0, 0.0))]
    for step in steps:
        for direction in directions[1:]:
            vectors.append(direction * step)
    return vectors


def keyframe_refined_bones(armature, side, frame, all_forelimb):
    names = forelimb_bones(side) if all_forelimb else [f"Scapula_{side}", f"Wrist_{side}"]
    for name in names:
        if name not in armature.pose.bones:
            continue
        pb = armature.pose.bones[name]
        pb.keyframe_insert(data_path="location", frame=frame)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pb.keyframe_insert(data_path="scale", frame=frame)


def snapshot_pose_basis(armature, names):
    return {
        name: armature.pose.bones[name].matrix_basis.copy()
        for name in names
        if name in armature.pose.bones
    }


def restore_pose_basis(armature, snapshot):
    for name, matrix_basis in snapshot.items():
        armature.pose.bones[name].matrix_basis = matrix_basis.copy()
    bpy.context.view_layer.update()


def refine_side_frame(
    armature,
    mesh_obj,
    selected_indices,
    arm_indices,
    volume_reference,
    obstacle_bvhs,
    side,
    args,
    vectors,
    scapula_rolls,
    wrist_rolls,
):
    current_palm = bone_tail_world(armature, f"Wrist_{side}")
    pose_snapshot = snapshot_pose_basis(armature, [f"Scapula_{side}", f"Wrist_{side}"])
    baseline = score_side(
        armature,
        mesh_obj,
        selected_indices,
        arm_indices,
        volume_reference,
        side,
        obstacle_bvhs,
        args,
    )
    best = baseline
    best_vector = Vector((0.0, 0.0, 0.0))
    best_palm = current_palm.copy()
    best_scapula_roll = 0.0
    best_wrist_roll = 0.0

    for vector in vectors:
        for scapula_roll in scapula_rolls:
            for wrist_roll in wrist_rolls:
                restore_pose_basis(armature, pose_snapshot)
                set_forelimb_palm_world(
                    armature,
                    side,
                    current_palm + vector,
                    args.min_reach_ratio,
                )
                roll_pose_bone_around_local_y(armature, f"Scapula_{side}", scapula_roll)
                roll_pose_bone_around_local_y(armature, f"Wrist_{side}", wrist_roll)
                candidate = score_side(
                    armature,
                    mesh_obj,
                    selected_indices,
                    arm_indices,
                    volume_reference,
                    side,
                    obstacle_bvhs,
                    args,
                )
                if candidate["score"] < best["score"]:
                    best = candidate
                    best_vector = vector.copy()
                    best_palm = current_palm + vector
                    best_scapula_roll = float(scapula_roll)
                    best_wrist_roll = float(wrist_roll)

    improved = best["score"] < baseline["score"]
    restore_pose_basis(armature, pose_snapshot)
    if improved:
        set_forelimb_palm_world(armature, side, best_palm, args.min_reach_ratio)
        roll_pose_bone_around_local_y(armature, f"Scapula_{side}", best_scapula_roll)
        roll_pose_bone_around_local_y(armature, f"Wrist_{side}", best_wrist_roll)
    return {
        "side": side,
        "improved": improved,
        "baseline": baseline,
        "best": best,
        "best_vector": [float(value) for value in best_vector],
        "best_scapula_roll_deg": best_scapula_roll,
        "best_wrist_roll_deg": best_wrist_roll,
    }


def main():
    args = parse_args()
    armature = bpy.data.objects[args.armature]
    mesh_obj = bpy.data.objects[args.mesh]
    action_list = actions_to_refine(armature, args.action, args.all_mig_actions)
    obstacle_names = args.obstacle or [
        name for name in DEFAULT_OBSTACLES if bpy.data.objects.get(name) is not None
    ]
    obstacle_bvhs = [
        mesh_bvh_world(bpy.data.objects[name])
        for name in obstacle_names
        if bpy.data.objects.get(name) is not None
    ]
    if not obstacle_bvhs:
        raise RuntimeError("No obstacle BVHs available.")

    selected_indices = collect_paw_indices(mesh_obj)
    arm_indices, arm_groups = collect_arm_indices(mesh_obj)
    steps = args.step or [0.5, 1.0, 1.5, 2.25, 3.0]
    vectors = candidate_vectors(steps)
    scapula_rolls = sorted(set([0.0] + [float(value) for value in args.scapula_roll_deg]))
    wrist_rolls = sorted(set([0.0] + [float(value) for value in args.wrist_roll_deg]))
    volume_references = build_volume_references(
        action_list,
        armature,
        mesh_obj,
        arm_indices,
        args,
    ) if args.arm_volume_weight > 0.0 else {}

    report = {
        "file": bpy.data.filepath,
        "output": str(Path(args.output).resolve()),
        "armature": armature.name,
        "mesh": mesh_obj.name,
        "obstacles": obstacle_names,
        "arm_vertex_groups": arm_groups,
        "arm_vertex_counts": {
            side: {segment: len(indices) for segment, indices in segment_indices.items()}
            for side, segment_indices in arm_indices.items()
        },
        "arm_volume_references": volume_references,
        "settings": {
            "iterations": args.iterations,
            "steps": steps,
            "min_reach_ratio": args.min_reach_ratio,
            "clearance_threshold": args.clearance_threshold,
            "score_overlap_weight": args.score_overlap_weight,
            "arm_volume_ratio_threshold": args.arm_volume_ratio_threshold,
            "arm_volume_weight": args.arm_volume_weight,
            "arm_volume_bins": args.arm_volume_bins,
            "arm_volume_trim": args.arm_volume_trim,
            "score_reach_ratio_threshold": args.score_reach_ratio_threshold,
            "score_reach_weight": args.score_reach_weight,
            "scapula_rolls": scapula_rolls,
            "wrist_rolls": wrist_rolls,
        },
        "actions": [],
    }

    for action in action_list:
        set_action(armature, action)
        frames = sorted(args.frame) if args.frame else action_frames(action)
        action_report = {
            "action": action.name,
            "frame_count": len(frames),
            "improved_count": 0,
            "samples": [],
        }
        print(f"[refine] {action.name}, {len(frames)} frames")
        for frame_index, frame in enumerate(frames, start=1):
            set_frame(frame)
            frame_report = {"frame": float(frame), "iterations": []}
            for iteration in range(args.iterations):
                iteration_rows = []
                for side in ("L", "R"):
                    row = refine_side_frame(
                        armature,
                        mesh_obj,
                        selected_indices,
                        arm_indices,
                        volume_references.get(action.name, {}).get(side, {}),
                        obstacle_bvhs,
                        side,
                        args,
                        vectors,
                        scapula_rolls,
                        wrist_rolls,
                    )
                    if row["improved"]:
                        action_report["improved_count"] += 1
                    iteration_rows.append(row)
                    keyframe_refined_bones(
                        armature,
                        side,
                        frame,
                        args.keyframe_all_forelimb_bones,
                    )
                frame_report["iterations"].append(iteration_rows)
            if frame_index <= 8 or any(
                row["improved"]
                for iteration_rows in frame_report["iterations"]
                for row in iteration_rows
            ):
                action_report["samples"].append(frame_report)
            if frame_index == 1 or frame_index == len(frames) or frame_index % 50 == 0:
                print(f"  refined {frame_index}/{len(frames)} frames")
        report["actions"].append(action_report)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.resolve()))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[done]")
    print(f"output={output_path.resolve()}")
    print(f"report={report_path.resolve()}")


if __name__ == "__main__":
    main()

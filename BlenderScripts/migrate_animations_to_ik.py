import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import numpy as np


DEFAULT_TARGET = "PyMouseStudio-BrowmFur-WaterReaching-IK.blend"
DEFAULT_TARGET_ARMATURE = "DeformationSystem"
DEFAULT_SOURCE_ARMATURE = "DeformationSystem"
HEAD_REFERENCE_BONES = ("Head1_M", "HeadEnd_M", "Jaw_M", "JawEnd_M")
BODY_TAIL_ALIGNMENT_BONES = (
    "Root_M",
    "RootPart1_M",
    "RootPart2_M",
    "Spine1_M",
    "Spine2_M",
    "Spine3_M",
    "Spine4_M",
    "Chest_M",
    "Neck3_M",
    "Neck1_M",
    "Neck2_M",
    "Head1_M",
    "Hip_L",
    "Hip_R",
    "Knee_L",
    "Knee_R",
    "Ankle_L",
    "Ankle_R",
    "Tail0_M",
    "Tail1_M",
    "Tail2_M",
    "Tail3_M",
)


def action_fcurves(action):
    fcurves = []
    if not action:
        return fcurves
    legacy_fcurves = getattr(action, "fcurves", None)
    if legacy_fcurves is not None:
        fcurves.extend(legacy_fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def action_keyframes(action):
    frames = set()
    interpolation = Counter()
    for fc in action_fcurves(action):
        for key in fc.keyframe_points:
            frames.add(round(float(key.co.x), 6))
            interpolation[key.interpolation] += 1

    if frames:
        return sorted(frames), interpolation.most_common(1)[0][0] if interpolation else "BEZIER"

    start, end = action.frame_range
    return [float(frame) for frame in range(math.ceil(start), math.floor(end) + 1)], "BEZIER"


def set_scene_frame(scene, frame):
    whole = math.floor(frame)
    scene.frame_set(whole, subframe=frame - whole)


def find_armature(name=None, require_action=False, bone_count=None):
    candidates = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if name:
        exact = [obj for obj in candidates if obj.name == name]
        if exact:
            candidates = exact

    if require_action:
        with_actions = [
            obj
            for obj in candidates
            if obj.animation_data and obj.animation_data.action
        ]
        if with_actions:
            candidates = with_actions

    if bone_count is not None:
        matching_count = [obj for obj in candidates if len(obj.data.bones) == bone_count]
        if matching_count:
            candidates = matching_count

    if not candidates:
        raise RuntimeError(f"No armature found for name={name!r}, require_action={require_action}")

    return candidates[0]


def parent_first_bone_names(armature, allowed_names):
    def depth(bone):
        level = 0
        current = bone
        while current.parent:
            level += 1
            current = current.parent
        return level

    bones = [armature.data.bones[name] for name in allowed_names if name in armature.data.bones]
    bones.sort(key=lambda bone: (depth(bone), bone.name))
    return [bone.name for bone in bones]


def bone_depth_groups(armature, allowed_names):
    def depth(bone):
        level = 0
        current = bone
        while current.parent:
            level += 1
            current = current.parent
        return level

    groups = {}
    for name in allowed_names:
        if name not in armature.data.bones:
            continue
        bone = armature.data.bones[name]
        groups.setdefault(depth(bone), []).append(name)
    return [sorted(groups[level]) for level in sorted(groups)]


def descendant_bone_names(armature, root_names):
    names = set()

    def visit(bone):
        names.add(bone.name)
        for child in bone.children:
            visit(child)

    for root_name in root_names:
        if root_name in armature.data.bones:
            visit(armature.data.bones[root_name])
    return names


def sanitize_action_name(source_path):
    stem = Path(source_path).stem
    stem = stem.replace("PyMouseStudio-", "")
    stem = stem.replace("WaterReaching-FrameToFrame-", "")
    stem = stem.replace("WaterReaching-", "")
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_")
    return f"MIG_{stem}"[:60]


def unique_action_name(base_name):
    existing = {action.name for action in bpy.data.actions}
    if base_name not in existing:
        return base_name
    index = 1
    while f"{base_name}.{index:03d}" in existing:
        index += 1
    return f"{base_name}.{index:03d}"


def clear_pose(armature):
    for pb in armature.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def hide_reference_mouse_objects():
    for name in ("DeformationSystem.001", "MS_Mouse.001"):
        obj = bpy.data.objects.get(name)
        if obj:
            try:
                obj.hide_set(True)
            except RuntimeError:
                pass
            obj.hide_viewport = True
            obj.hide_render = True


def matrix_error(a, b):
    return max(abs(a[row][col] - b[row][col]) for row in range(4) for col in range(4))


def bone_tail_from_matrix(matrix, bone):
    return matrix @ Vector((0.0, bone.length, 0.0))


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


def matrix_from_rotation_scale_translation(rotation, scale, translation):
    matrix = Matrix.Identity(4)
    for row in range(3):
        for col in range(3):
            matrix[row][col] = float(rotation[row, col]) * scale
    matrix[0][3] = float(translation[0])
    matrix[1][3] = float(translation[1])
    matrix[2][3] = float(translation[2])
    return matrix


def similarity_matrix_from_points(source_points, target_points, allow_scale=True):
    if len(source_points) < 3 or len(target_points) < 3:
        return Matrix.Identity(4), 1.0, 0

    source = np.array([[point.x, point.y, point.z] for point in source_points], dtype=np.float64)
    target = np.array([[point.x, point.y, point.z] for point in target_points], dtype=np.float64)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered.T @ target_centered / len(source)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T

    scale = 1.0
    source_variance = float((source_centered * source_centered).sum() / len(source))
    if allow_scale and source_variance > 1e-12:
        scale = float(singular_values.sum() / source_variance)

    translation = target_centroid - scale * rotation @ source_centroid
    return matrix_from_rotation_scale_translation(rotation, scale, translation), scale, len(source)


def similarity_matrix_from_matrices(source_matrices, target_matrices, bone_names, allow_scale=True):
    source_points = []
    target_points = []
    for name in bone_names:
        if name not in source_matrices or name not in target_matrices:
            continue
        source_value = source_matrices[name]
        target_value = target_matrices[name]
        source_points.append(
            source_value.translation if hasattr(source_value, "translation") else source_value
        )
        target_points.append(
            target_value.translation if hasattr(target_value, "translation") else target_value
        )
    return similarity_matrix_from_points(source_points, target_points, allow_scale)


def body_frame_from_matrices(matrices):
    required = ["Root_M", "Chest_M", "Hip_L", "Hip_R", "Scapula_L", "Scapula_R"]
    missing = [name for name in required if name not in matrices]
    if missing:
        raise RuntimeError(f"Cannot build body frame, missing bones: {missing}")

    root = matrices["Root_M"].translation.copy()
    chest = matrices["Chest_M"].translation.copy()
    left = (matrices["Hip_L"].translation + matrices["Scapula_L"].translation) * 0.5
    right = (matrices["Hip_R"].translation + matrices["Scapula_R"].translation) * 0.5

    forward = chest - root
    if forward.length < 1e-8:
        raise RuntimeError("Cannot build body frame: Root_M and Chest_M overlap.")
    forward.normalize()

    lateral = right - left
    lateral = lateral - forward * lateral.dot(forward)
    if lateral.length < 1e-8:
        raise RuntimeError("Cannot build body frame: lateral axis is degenerate.")
    lateral.normalize()

    up = lateral.cross(forward)
    if up.length < 1e-8:
        raise RuntimeError("Cannot build body frame: up axis is degenerate.")
    up.normalize()
    forward = up.cross(lateral)
    forward.normalize()

    return matrix_with_columns(lateral, forward, up, root)


def aim_bone_tail_at(matrix, bone, tail_position, preserve_length):
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

    x_new = x_dir * x_scale
    y_new = y_dir * y_scale if preserve_length else segment / bone.length
    z_new = z_dir * z_scale
    return matrix_with_columns(x_new, y_new, z_new, head)


def remove_cross_section_scale(matrix):
    x_vec = column_vector(matrix, 0)
    y_vec = column_vector(matrix, 1)
    z_vec = column_vector(matrix, 2)
    if y_vec.length < 1e-8:
        return matrix

    y_dir = y_vec.normalized()
    y_new = y_vec.copy()
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

    return matrix_with_columns(x_dir, y_new, z_dir, matrix.translation.copy())


def sanitize_scale(matrix, mode):
    if mode == "full":
        return matrix
    if mode == "keep-y":
        return remove_cross_section_scale(matrix)
    if mode == "none":
        sanitized = remove_cross_section_scale(matrix)
        y_vec = column_vector(sanitized, 1)
        if y_vec.length < 1e-8:
            return sanitized
        return matrix_with_columns(
            column_vector(sanitized, 0),
            y_vec.normalized(),
            column_vector(sanitized, 2),
            sanitized.translation.copy(),
        )
    raise RuntimeError(f"Unknown scale mode: {mode}")


def translate_matrices_in_place(matrices, bone_names, delta):
    for name in bone_names:
        matrices[name].translation = matrices[name].translation + delta


def aim_connected_parents(desired_local, target, ordered_bones, connected_aim_mode):
    aimed = []
    preserve_length = connected_aim_mode == "rotate"
    for child_name in ordered_bones:
        child_bone = target.data.bones[child_name]
        if not child_bone.use_connect or not child_bone.parent:
            continue
        parent_name = child_bone.parent.name
        if parent_name not in desired_local or child_name not in desired_local:
            continue
        child_head = desired_local[child_name].translation
        desired_local[parent_name] = aim_bone_tail_at(
            desired_local[parent_name],
            target.data.bones[parent_name],
            child_head,
            preserve_length,
        )
        aimed.append(parent_name)
    return aimed


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


def set_pose_matrix_rotation_only(
    armature,
    bone_name,
    desired_pose_matrix,
    reference_channels,
    allow_location,
    preserve_pose_scale=False,
):
    set_pose_matrix_via_basis(armature, bone_name, desired_pose_matrix)
    pb = armature.pose.bones[bone_name]
    if not allow_location:
        pb.location = reference_channels[bone_name]["location"].copy()
    if not preserve_pose_scale:
        pb.scale = reference_channels[bone_name]["scale"].copy()


def lock_pose_by_translating_root(armature, root_bone_name, lock_bone_name, target_position, reference_channels):
    if root_bone_name not in armature.pose.bones or lock_bone_name not in armature.pose.bones:
        return
    actual_position = armature.pose.bones[lock_bone_name].matrix.translation
    delta = target_position - actual_position
    root_matrix = armature.pose.bones[root_bone_name].matrix.copy()
    root_matrix.translation = root_matrix.translation + delta
    set_pose_matrix_via_basis(armature, root_bone_name, root_matrix)
    armature.pose.bones[root_bone_name].scale = reference_channels[root_bone_name]["scale"].copy()


def mesh_vertices_world(obj):
    if not obj or obj.type != "MESH":
        return []
    return [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]


def mesh_bvh_world(obj):
    if not obj or obj.type != "MESH":
        return None
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    polygons = [tuple(poly.vertices) for poly in obj.data.polygons if len(poly.vertices) >= 3]
    if not vertices or not polygons:
        return None
    return {
        "name": obj.name,
        "bvh": BVHTree.FromPolygons(vertices, polygons),
        "bbox_min": Vector((
            min(vertex.x for vertex in vertices),
            min(vertex.y for vertex in vertices),
            min(vertex.z for vertex in vertices),
        )),
        "bbox_max": Vector((
            max(vertex.x for vertex in vertices),
            max(vertex.y for vertex in vertices),
            max(vertex.z for vertex in vertices),
        )),
    }


def closest_vertex(point, vertices):
    if not vertices:
        return None, None
    best_vertex = None
    best_distance = None
    for vertex in vertices:
        distance = (point - vertex).length
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_vertex = vertex
    return best_vertex, best_distance


def closest_surface(point, bvh_records):
    best = None
    for record in bvh_records:
        nearest = record["bvh"].find_nearest(point)
        if nearest is None or nearest[0] is None:
            continue
        location, normal, face_index, distance = nearest
        row = {
            "object": record["name"],
            "location": location,
            "normal": normal,
            "face_index": face_index,
            "distance": distance,
        }
        if best is None or distance < best["distance"]:
            best = row
    return best


def cup_clearance_push(probe_points_world, cup_bvhs, clearance_distance, max_push):
    if not cup_bvhs or clearance_distance <= 0.0:
        return Vector(), None

    best_push = Vector()
    best_report = None
    for label, point in probe_points_world:
        nearest = closest_surface(point, cup_bvhs)
        if nearest is None or nearest["distance"] >= clearance_distance:
            continue

        direction = nearest["normal"].copy()
        if direction.length < 1e-8:
            direction = point - nearest["location"]
        if direction.length < 1e-8:
            continue
        direction.normalize()

        push_length = clearance_distance - nearest["distance"]
        if max_push > 0.0:
            push_length = min(push_length, max_push)
        push = direction * push_length
        if push.length > best_push.length:
            best_push = push
            best_report = {
                "probe": label,
                "object": nearest["object"],
                "distance": float(nearest["distance"]),
                "push_length": float(push.length),
                "normal": [float(value) for value in direction],
            }
    return best_push, best_report


def cup_front_y_push(probe_points_world, cup_bvhs, front_clearance, max_push):
    if not cup_bvhs or front_clearance <= 0.0:
        return Vector(), None

    front_y = max(record["bbox_max"].y for record in cup_bvhs)
    target_y = front_y + front_clearance
    best_push = 0.0
    best_report = None
    for label, point in probe_points_world:
        if label == "palm" or point.y >= target_y:
            continue
        push_length = target_y - point.y
        if max_push > 0.0:
            push_length = min(push_length, max_push)
        if push_length > best_push:
            best_push = push_length
            best_report = {
                "probe": label,
                "front_y": float(front_y),
                "distance_behind_front": float(target_y - point.y),
                "push_length": float(push_length),
            }
    return Vector((0.0, best_push, 0.0)), best_report


def solve_two_bone_joint(
    root,
    preferred_joint,
    target,
    first_length,
    second_length,
    min_reach_ratio=0.0,
):
    to_target = target - root
    distance = to_target.length
    if distance < 1e-8:
        return preferred_joint.copy(), root.copy()

    direction = to_target.normalized()
    max_reach = max(first_length + second_length - 1e-6, 1e-6)
    min_reach = max(abs(first_length - second_length) + 1e-6, 1e-6)
    if min_reach_ratio > 0.0:
        min_reach = max(min_reach, (first_length + second_length) * min_reach_ratio)
        min_reach = min(min_reach, max_reach)
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


def apply_forelimb_source_palm_ik(
    desired_local,
    source_aligned_local,
    target_base_local,
    target,
    source_world_heads=None,
    source_world_tails=None,
    source_to_target_world=None,
    pipe_vertices_world=None,
    pipe_snap_threshold=0.0,
    cup_bvhs=None,
    cup_clearance_distance=0.0,
    cup_clearance_max_push=0.0,
    cup_front_clearance_distance=0.0,
    cup_front_clearance_max_push=0.0,
    forelimb_min_reach_ratio=0.0,
):
    solved = []
    for side in ("L", "R"):
        scapula_name = f"Scapula_{side}"
        wrist_name = f"Wrist_{side}"
        fingers_name = f"Fingers1_{side}"
        required = (scapula_name, wrist_name, fingers_name)
        if not all(name in desired_local and name in source_aligned_local for name in required):
            continue

        shoulder = target_base_local[scapula_name].translation.copy()
        preferred_joint = source_aligned_local[wrist_name].translation.copy()
        palm_target = source_aligned_local[fingers_name].translation.copy()
        target_source = "local-deform"
        snap_distance = None
        if (
            source_world_heads
            and source_to_target_world
            and wrist_name in source_world_heads
            and fingers_name in source_world_heads
        ):
            target_to_local = target.matrix_world.inverted()
            preferred_joint = target_to_local @ (
                source_to_target_world @ source_world_heads[wrist_name]
            )
            palm_target = target_to_local @ (
                source_to_target_world @ source_world_heads[fingers_name]
            )
            target_source = "source-world"

            if source_world_tails and pipe_vertices_world and pipe_snap_threshold > 0.0:
                best_snap = None
                for finger_name in (
                    f"IndexFinger3_{side}",
                    f"MiddleFinger3_{side}",
                    f"RingFinger3_{side}",
                    f"PinkyFinger3_{side}",
                ):
                    if finger_name not in source_world_tails:
                        continue
                    target_tip_world = source_to_target_world @ source_world_tails[finger_name]
                    pipe_vertex, pipe_distance = closest_vertex(
                        target_tip_world,
                        pipe_vertices_world,
                    )
                    if pipe_vertex is None:
                        continue
                    if pipe_distance <= pipe_snap_threshold and (
                        best_snap is None or pipe_distance < best_snap["pipe_distance"]
                    ):
                        best_snap = {
                            "finger": finger_name,
                            "tip_world": target_tip_world,
                            "pipe_world": pipe_vertex,
                            "pipe_distance": pipe_distance,
                        }
                if best_snap:
                    target_palm_world = source_to_target_world @ source_world_heads[fingers_name]
                    tip_offset = best_snap["tip_world"] - target_palm_world
                    palm_target = target_to_local @ (best_snap["pipe_world"] - tip_offset)
                    target_source = f"source-world-pipe-snap:{best_snap['finger']}"
                    snap_distance = float(best_snap["pipe_distance"])

            if cup_bvhs and cup_clearance_distance > 0.0:
                target_palm_world = target.matrix_world @ palm_target
                probe_points_world = [("palm", target_palm_world)]
                if source_world_tails and fingers_name in source_world_heads:
                    source_palm_world = source_to_target_world @ source_world_heads[fingers_name]
                    for finger_name in (
                        f"IndexFinger3_{side}",
                        f"MiddleFinger3_{side}",
                        f"RingFinger3_{side}",
                        f"PinkyFinger3_{side}",
                    ):
                        if finger_name not in source_world_tails:
                            continue
                        tip_offset = (
                            source_to_target_world @ source_world_tails[finger_name]
                        ) - source_palm_world
                        probe_points_world.append((finger_name, target_palm_world + tip_offset))

                push_world, push_report = cup_clearance_push(
                    probe_points_world,
                    cup_bvhs,
                    cup_clearance_distance,
                    cup_clearance_max_push,
                )
                if push_report:
                    palm_target = target.matrix_world.inverted() @ (
                        target_palm_world + push_world
                    )
                    target_palm_world = target.matrix_world @ palm_target
                    target_source = f"{target_source}+cup-clearance:{push_report['probe']}"
                front_push_world, front_push_report = cup_front_y_push(
                    [
                        (label, point + push_world)
                        for label, point in probe_points_world
                    ],
                    cup_bvhs,
                    cup_front_clearance_distance,
                    cup_front_clearance_max_push,
                )
                if front_push_report:
                    palm_target = target.matrix_world.inverted() @ (
                        target_palm_world + front_push_world
                    )
                    target_source = (
                        f"{target_source}+cup-front-y:{front_push_report['probe']}"
                    )
        first_length = (
            target_base_local[wrist_name].translation
            - target_base_local[scapula_name].translation
        ).length
        second_length = (
            target_base_local[fingers_name].translation
            - target_base_local[wrist_name].translation
        ).length
        if first_length < 1e-8 or second_length < 1e-8:
            continue

        wrist_position, palm_position = solve_two_bone_joint(
            shoulder,
            preferred_joint,
            palm_target,
            first_length,
            second_length,
            forelimb_min_reach_ratio,
        )

        desired_local[scapula_name] = aim_bone_tail_at(
            target_base_local[scapula_name].copy(),
            target.data.bones[scapula_name],
            wrist_position,
            True,
        )
        wrist_matrix = target_base_local[wrist_name].copy()
        wrist_matrix.translation = wrist_position
        desired_local[wrist_name] = aim_bone_tail_at(
            wrist_matrix,
            target.data.bones[wrist_name],
            palm_position,
            True,
        )
        solved.append(
            {
                "side": side,
                "source_target_distance": float((palm_target - shoulder).length),
                "clamped_target_distance": float((palm_position - shoulder).length),
                "max_reach": float(first_length + second_length),
                "min_reach": float(
                    max(
                        abs(first_length - second_length) + 1e-6,
                        (first_length + second_length) * forelimb_min_reach_ratio
                        if forelimb_min_reach_ratio > 0.0
                        else 0.0,
                    )
                ),
                "target_source": target_source,
                "pipe_snap_distance": snap_distance,
            }
        )
    return solved


FINGER_CHAINS = {
    "Index": ("IndexFinger1", "IndexFinger2", "IndexFinger3"),
    "Middle": ("MiddleFinger1", "MiddleFinger2", "MiddleFinger3"),
    "Ring": ("RingFinger1", "RingFinger2", "RingFinger3"),
    "Pinky": ("PinkyFinger1", "PinkyFinger2", "PinkyFinger3"),
}


def local_target_from_source(source_world_points, source_to_target_world, target, name, endpoint):
    if not source_world_points or not source_to_target_world:
        return None
    point = source_world_points.get(name)
    if point is None:
        return None
    return target.matrix_world.inverted() @ (source_to_target_world @ point)


def mapped_source_head(source_world_heads, source_to_target_world, target, name):
    return local_target_from_source(source_world_heads, source_to_target_world, target, name, "head")


def mapped_source_tail(source_world_tails, source_to_target_world, target, name):
    return local_target_from_source(source_world_tails, source_to_target_world, target, name, "tail")


def matrix_with_translation(matrix, translation):
    out = matrix.copy()
    out.translation = translation
    return out


def pose_segment_length(matrix, bone):
    return (bone_tail_from_matrix(matrix, bone) - matrix.translation).length


def point_from_parent_rest_offset(target_base_local, desired_local, parent_name, child_name):
    if parent_name not in desired_local or parent_name not in target_base_local:
        return None
    if child_name not in target_base_local:
        return None
    parent_base_inv = target_base_local[parent_name].inverted()
    child_head_in_parent = parent_base_inv @ target_base_local[child_name].translation
    return desired_local[parent_name] @ child_head_in_parent


def vector_payload(vector):
    return [float(value) for value in vector]


def direction_or_fallback(start, end, fallback):
    vector = end - start
    if vector.length < 1e-8:
        vector = fallback.copy()
    if vector.length < 1e-8:
        vector = Vector((0.0, 1.0, 0.0))
    return vector.normalized()


def source_bone_direction_target(
    source_world_heads,
    source_world_tails,
    source_to_target_world,
    target,
    bone_name,
    head_target,
    segment_length,
    fallback_tail,
):
    source_head = mapped_source_head(
        source_world_heads,
        source_to_target_world,
        target,
        bone_name,
    )
    source_tail = mapped_source_tail(
        source_world_tails,
        source_to_target_world,
        target,
        bone_name,
    )
    if source_head is None or source_tail is None:
        return fallback_tail
    direction = direction_or_fallback(source_head, source_tail, fallback_tail - head_target)
    return head_target + direction * segment_length


def aim_pose_bone_from_source_endpoints(
    desired_local,
    target_base_local,
    target,
    bone_name,
    head_target,
    tail_target,
    preserve_length=True,
    set_head=True,
):
    if bone_name not in desired_local or bone_name not in target_base_local or tail_target is None:
        return None
    base = target_base_local[bone_name].copy()
    if set_head and head_target is not None:
        base.translation = head_target
    aimed = aim_bone_tail_at(base, target.data.bones[bone_name], tail_target, preserve_length)
    desired_local[bone_name] = aimed
    actual_tail = bone_tail_from_matrix(aimed, target.data.bones[bone_name])
    return {
        "bone": bone_name,
        "head_target": None if head_target is None else [float(value) for value in head_target],
        "tail_target": [float(value) for value in tail_target],
        "tail_error": float((actual_tail - tail_target).length),
        "set_head": bool(set_head),
    }


def apply_forelimb_source_endpoint_chain_ik(
    desired_local,
    target_base_local,
    target,
    source_world_heads,
    source_world_tails,
    source_to_target_world,
    forelimb_min_reach_ratio=0.0,
    palm_source_endpoint="fingers-head",
):
    """Drive old-IK forelimbs from source bone endpoint targets.

    The old IK file was made by editing the source bones into connected chains.
    For the main arm this means the IK chain should use source Wrist/Fingers
    endpoints as elbow/palm targets. Finger segments use source heads/tails
    directly so their local rotations no longer depend on the non-connected
    source rest orientation.
    """
    solved = []
    if not source_world_heads or not source_world_tails or not source_to_target_world:
        return solved
    for side in ("L", "R"):
        scapula_name = f"Scapula_{side}"
        wrist_name = f"Wrist_{side}"
        fingers_name = f"Fingers1_{side}"
        required = (scapula_name, wrist_name, fingers_name)
        if not all(name in desired_local and name in target_base_local for name in required):
            continue

        shoulder = target_base_local[scapula_name].translation.copy()
        elbow_target = mapped_source_head(source_world_heads, source_to_target_world, target, wrist_name)
        if palm_source_endpoint == "wrist-tail":
            palm_target = mapped_source_tail(
                source_world_tails,
                source_to_target_world,
                target,
                wrist_name,
            )
        else:
            palm_target = mapped_source_head(
                source_world_heads,
                source_to_target_world,
                target,
                fingers_name,
            )
        if elbow_target is None or palm_target is None:
            continue

        first_length = (target_base_local[wrist_name].translation - shoulder).length
        second_length = (
            target_base_local[fingers_name].translation - target_base_local[wrist_name].translation
        ).length
        if first_length < 1e-8 or second_length < 1e-8:
            continue
        wrist_position, palm_position = solve_two_bone_joint(
            shoulder,
            elbow_target,
            palm_target,
            first_length,
            second_length,
            forelimb_min_reach_ratio,
        )

        side_report = {
            "side": side,
            "mode": "source-endpoint-chain-ik",
            "palm_source_endpoint": palm_source_endpoint,
            "arm": [],
            "fingers": [],
            "source_elbow_distance": float((elbow_target - shoulder).length),
            "source_palm_distance": float((palm_target - shoulder).length),
            "clamped_palm_distance": float((palm_position - shoulder).length),
            "max_reach": float(first_length + second_length),
        }
        row = aim_pose_bone_from_source_endpoints(
            desired_local,
            target_base_local,
            target,
            scapula_name,
            shoulder,
            wrist_position,
            True,
            True,
        )
        if row:
            side_report["arm"].append(row)
        wrist_matrix = target_base_local[wrist_name].copy()
        wrist_matrix.translation = wrist_position
        desired_local[wrist_name] = aim_bone_tail_at(
            wrist_matrix,
            target.data.bones[wrist_name],
            palm_position,
            True,
        )
        actual_wrist_tail = bone_tail_from_matrix(desired_local[wrist_name], target.data.bones[wrist_name])
        side_report["arm"].append(
            {
                "bone": wrist_name,
                "head_target": [float(value) for value in wrist_position],
                "tail_target": [float(value) for value in palm_position],
                "tail_error": float((actual_wrist_tail - palm_position).length),
                "set_head": True,
            }
        )

        # Keep the palm bone anchored at the solved palm and aim it toward the
        # average first-finger base. This gives the fingers a stable parent frame.
        finger_base_targets = []
        for chain in FINGER_CHAINS.values():
            target_point = mapped_source_head(
                source_world_heads,
                source_to_target_world,
                target,
                f"{chain[0]}_{side}",
            )
            if target_point is not None:
                finger_base_targets.append(target_point)
        if finger_base_targets:
            avg_base = Vector((0.0, 0.0, 0.0))
            for point in finger_base_targets:
                avg_base += point
            avg_base /= len(finger_base_targets)
            row = aim_pose_bone_from_source_endpoints(
                desired_local,
                target_base_local,
                target,
                fingers_name,
                palm_position,
                avg_base,
                True,
                True,
            )
            if row:
                side_report["fingers"].append(row)

        for chain_name, chain in FINGER_CHAINS.items():
            for index, base_name in enumerate(chain):
                bone_name = f"{base_name}_{side}"
                if bone_name not in desired_local or bone_name not in target_base_local:
                    continue
                head_target = mapped_source_head(
                    source_world_heads,
                    source_to_target_world,
                    target,
                    bone_name,
                )
                if index + 1 < len(chain):
                    tail_source_name = f"{chain[index + 1]}_{side}"
                    tail_target = mapped_source_head(
                        source_world_heads,
                        source_to_target_world,
                        target,
                        tail_source_name,
                    )
                else:
                    tail_target = mapped_source_tail(
                        source_world_tails,
                        source_to_target_world,
                        target,
                        bone_name,
                    )
                row = aim_pose_bone_from_source_endpoints(
                    desired_local,
                    target_base_local,
                    target,
                    bone_name,
                    head_target,
                    tail_target,
                    True,
                    True,
                )
                if row:
                    row["chain"] = chain_name
                    side_report["fingers"].append(row)

        solved.append(side_report)
    return solved


def apply_forelimb_source_tail_shape_ik(
    desired_local,
    target_base_local,
    target,
    source_world_heads,
    source_world_tails,
    source_to_target_world,
    forelimb_min_reach_ratio=0.0,
):
    """Normalize source endpoint shape onto old-IK bone lengths.

    Source forelimb bones are disconnected, so absolute source tail distances can
    exceed the connected old-IK chain.  This mode keeps the source reach ratio
    and pole direction while solving inside the target chain's own lengths.
    """
    solved = []
    if not source_world_heads or not source_world_tails or not source_to_target_world:
        return solved

    for side in ("L", "R"):
        scapula_name = f"Scapula_{side}"
        wrist_name = f"Wrist_{side}"
        fingers_name = f"Fingers1_{side}"
        required = (scapula_name, wrist_name, fingers_name)
        if not all(name in desired_local and name in target_base_local for name in required):
            continue

        shoulder = target_base_local[scapula_name].translation.copy()
        source_shoulder = mapped_source_head(
            source_world_heads,
            source_to_target_world,
            target,
            scapula_name,
        )
        source_elbow = mapped_source_tail(
            source_world_tails,
            source_to_target_world,
            target,
            scapula_name,
        )
        source_wrist_tail = mapped_source_tail(
            source_world_tails,
            source_to_target_world,
            target,
            wrist_name,
        )
        if source_shoulder is None or source_elbow is None or source_wrist_tail is None:
            continue

        first_length = pose_segment_length(
            target_base_local[scapula_name],
            target.data.bones[scapula_name],
        )
        second_length = pose_segment_length(
            target_base_local[wrist_name],
            target.data.bones[wrist_name],
        )
        base_reach = first_length + second_length
        if first_length < 1e-8 or second_length < 1e-8 or base_reach < 1e-8:
            continue

        source_upper = (source_elbow - source_shoulder).length
        source_lower = (source_wrist_tail - source_elbow).length
        source_total = source_upper + source_lower
        source_reach = (source_wrist_tail - source_shoulder).length
        source_reach_ratio = (
            min(max(source_reach / source_total, 0.0), 1.0)
            if source_total > 1e-8
            else 1.0
        )
        if forelimb_min_reach_ratio > 0.0:
            source_reach_ratio = max(source_reach_ratio, forelimb_min_reach_ratio)
        wrist_direction = direction_or_fallback(
            source_shoulder,
            source_wrist_tail,
            bone_tail_from_matrix(target_base_local[wrist_name], target.data.bones[wrist_name]) - shoulder,
        )
        wrist_tail_target = shoulder + wrist_direction * (base_reach * source_reach_ratio)
        preferred_joint = shoulder + (source_elbow - source_shoulder)

        elbow_position, wrist_tail_position = solve_two_bone_joint(
            shoulder,
            preferred_joint,
            wrist_tail_target,
            first_length,
            second_length,
            forelimb_min_reach_ratio,
        )

        side_report = {
            "side": side,
            "mode": "source-tail-shape-ik",
            "arm": [],
            "fingers": [],
            "source_reach_ratio": float(source_reach_ratio),
            "source_wrist_tail_distance": float(source_reach),
            "clamped_wrist_tail_distance": float((wrist_tail_position - shoulder).length),
            "base_reach": float(base_reach),
            "max_reach": float(base_reach),
            "arm_stretch_factor": 1.0,
        }

        desired_local[scapula_name] = aim_bone_tail_at(
            target_base_local[scapula_name].copy(),
            target.data.bones[scapula_name],
            elbow_position,
            True,
        )
        actual_scapula_tail = bone_tail_from_matrix(
            desired_local[scapula_name],
            target.data.bones[scapula_name],
        )
        side_report["arm"].append(
            {
                "bone": scapula_name,
                "head_target": vector_payload(shoulder),
                "tail_target": vector_payload(elbow_position),
                "tail_error": float((actual_scapula_tail - elbow_position).length),
                "target_residual": 0.0,
                "set_head": False,
            }
        )

        wrist_head = bone_tail_from_matrix(
            desired_local[scapula_name],
            target.data.bones[scapula_name],
        )
        wrist_matrix = matrix_with_translation(target_base_local[wrist_name], wrist_head)
        desired_local[wrist_name] = aim_bone_tail_at(
            wrist_matrix,
            target.data.bones[wrist_name],
            wrist_tail_position,
            True,
        )
        actual_wrist_tail = bone_tail_from_matrix(
            desired_local[wrist_name],
            target.data.bones[wrist_name],
        )
        side_report["arm"].append(
            {
                "bone": wrist_name,
                "head_target": vector_payload(wrist_head),
                "tail_target": vector_payload(wrist_tail_position),
                "tail_error": float((actual_wrist_tail - wrist_tail_position).length),
                "target_residual": 0.0,
                "set_head": False,
            }
        )

        fingers_head = point_from_parent_rest_offset(
            target_base_local,
            desired_local,
            wrist_name,
            fingers_name,
        )
        if fingers_head is None:
            fingers_head = actual_wrist_tail.copy()
        fingers_length = pose_segment_length(
            target_base_local[fingers_name],
            target.data.bones[fingers_name],
        )
        fingers_tail = source_bone_direction_target(
            source_world_heads,
            source_world_tails,
            source_to_target_world,
            target,
            fingers_name,
            fingers_head,
            fingers_length,
            bone_tail_from_matrix(target_base_local[fingers_name], target.data.bones[fingers_name]),
        )
        fingers_matrix = matrix_with_translation(target_base_local[fingers_name], fingers_head)
        desired_local[fingers_name] = aim_bone_tail_at(
            fingers_matrix,
            target.data.bones[fingers_name],
            fingers_tail,
            True,
        )
        actual_fingers_tail = bone_tail_from_matrix(
            desired_local[fingers_name],
            target.data.bones[fingers_name],
        )
        side_report["fingers"].append(
            {
                "bone": fingers_name,
                "head_target": vector_payload(fingers_head),
                "tail_target": vector_payload(fingers_tail),
                "tail_error": float((actual_fingers_tail - fingers_tail).length),
                "target_residual": 0.0,
                "set_head": False,
            }
        )

        for chain_name, chain in FINGER_CHAINS.items():
            parent_name = fingers_name
            for base_name in chain:
                bone_name = f"{base_name}_{side}"
                if bone_name not in desired_local or bone_name not in target_base_local:
                    parent_name = bone_name
                    continue
                if parent_name not in desired_local:
                    parent_name = bone_name
                    continue

                head_target = point_from_parent_rest_offset(
                    target_base_local,
                    desired_local,
                    parent_name,
                    bone_name,
                )
                if head_target is None:
                    head_target = target_base_local[bone_name].translation.copy()
                length = pose_segment_length(
                    target_base_local[bone_name],
                    target.data.bones[bone_name],
                )
                fallback_tail = bone_tail_from_matrix(
                    target_base_local[bone_name],
                    target.data.bones[bone_name],
                )
                tail_target = source_bone_direction_target(
                    source_world_heads,
                    source_world_tails,
                    source_to_target_world,
                    target,
                    bone_name,
                    head_target,
                    length,
                    fallback_tail,
                )

                finger_matrix = matrix_with_translation(target_base_local[bone_name], head_target)
                desired_local[bone_name] = aim_bone_tail_at(
                    finger_matrix,
                    target.data.bones[bone_name],
                    tail_target,
                    True,
                )
                actual_tail = bone_tail_from_matrix(
                    desired_local[bone_name],
                    target.data.bones[bone_name],
                )
                side_report["fingers"].append(
                    {
                        "bone": bone_name,
                        "chain": chain_name,
                        "head_target": vector_payload(head_target),
                        "tail_target": vector_payload(tail_target),
                        "tail_error": float((actual_tail - tail_target).length),
                        "target_residual": 0.0,
                        "set_head": False,
                    }
                )
                parent_name = bone_name

        solved.append(side_report)
    return solved


def apply_forelimb_source_tail_constrained_ik(
    desired_local,
    target_base_local,
    target,
    source_world_tails,
    source_to_target_world,
    forelimb_min_reach_ratio=0.0,
    forelimb_max_arm_stretch=1.0,
):
    """Drive old-IK forelimbs from source bone tails without moving free bone heads.

    The source rig has disconnected forelimb bones, while the old IK rig was
    edited so many bone tails are semantic IK endpoints.  This mode therefore
    uses source tails as targets, keeps child heads at their target-rest offset
    from the solved parent, and preserves deform bone lengths.
    """
    solved = []
    if not source_world_tails or not source_to_target_world:
        return solved

    for side in ("L", "R"):
        scapula_name = f"Scapula_{side}"
        wrist_name = f"Wrist_{side}"
        fingers_name = f"Fingers1_{side}"
        required = (scapula_name, wrist_name, fingers_name)
        if not all(name in desired_local and name in target_base_local for name in required):
            continue

        shoulder = target_base_local[scapula_name].translation.copy()
        elbow_target = mapped_source_tail(
            source_world_tails,
            source_to_target_world,
            target,
            scapula_name,
        )
        wrist_tail_target = mapped_source_tail(
            source_world_tails,
            source_to_target_world,
            target,
            wrist_name,
        )
        fingers_tail_target = mapped_source_tail(
            source_world_tails,
            source_to_target_world,
            target,
            fingers_name,
        )
        if elbow_target is None or wrist_tail_target is None:
            continue

        first_length = pose_segment_length(
            target_base_local[scapula_name],
            target.data.bones[scapula_name],
        )
        second_length = pose_segment_length(
            target_base_local[wrist_name],
            target.data.bones[wrist_name],
        )
        if first_length < 1e-8 or second_length < 1e-8:
            continue

        requested_distance = (wrist_tail_target - shoulder).length
        base_reach = first_length + second_length
        stretch_factor = 1.0
        max_arm_stretch = max(float(forelimb_max_arm_stretch), 1.0)
        if base_reach > 1e-8 and requested_distance > base_reach:
            stretch_factor = min(max_arm_stretch, requested_distance / base_reach)
        solve_first_length = first_length * stretch_factor
        solve_second_length = second_length * stretch_factor

        elbow_position, wrist_tail_position = solve_two_bone_joint(
            shoulder,
            elbow_target,
            wrist_tail_target,
            solve_first_length,
            solve_second_length,
            forelimb_min_reach_ratio,
        )

        side_report = {
            "side": side,
            "mode": "source-tail-constrained-ik",
            "arm": [],
            "fingers": [],
            "source_elbow_distance": float((elbow_target - shoulder).length),
            "source_wrist_tail_distance": float((wrist_tail_target - shoulder).length),
            "clamped_wrist_tail_distance": float((wrist_tail_position - shoulder).length),
            "max_reach": float(solve_first_length + solve_second_length),
            "base_reach": float(base_reach),
            "arm_stretch_factor": float(stretch_factor),
        }
        preserve_arm_length = abs(stretch_factor - 1.0) < 1e-6

        desired_local[scapula_name] = aim_bone_tail_at(
            target_base_local[scapula_name].copy(),
            target.data.bones[scapula_name],
            elbow_position,
            preserve_arm_length,
        )
        actual_scapula_tail = bone_tail_from_matrix(
            desired_local[scapula_name],
            target.data.bones[scapula_name],
        )
        side_report["arm"].append(
            {
                "bone": scapula_name,
                "head_target": vector_payload(shoulder),
                "tail_target": vector_payload(elbow_target),
                "solved_tail": vector_payload(elbow_position),
                "tail_error": float((actual_scapula_tail - elbow_target).length),
                "target_residual": float((elbow_position - elbow_target).length),
                "set_head": False,
            }
        )

        wrist_head = bone_tail_from_matrix(
            desired_local[scapula_name],
            target.data.bones[scapula_name],
        )
        wrist_matrix = matrix_with_translation(target_base_local[wrist_name], wrist_head)
        desired_local[wrist_name] = aim_bone_tail_at(
            wrist_matrix,
            target.data.bones[wrist_name],
            wrist_tail_position,
            preserve_arm_length,
        )
        actual_wrist_tail = bone_tail_from_matrix(
            desired_local[wrist_name],
            target.data.bones[wrist_name],
        )
        side_report["arm"].append(
            {
                "bone": wrist_name,
                "head_target": vector_payload(wrist_head),
                "tail_target": vector_payload(wrist_tail_target),
                "solved_tail": vector_payload(wrist_tail_position),
                "tail_error": float((actual_wrist_tail - wrist_tail_target).length),
                "target_residual": float((wrist_tail_position - wrist_tail_target).length),
                "set_head": False,
            }
        )

        fingers_head = point_from_parent_rest_offset(
            target_base_local,
            desired_local,
            wrist_name,
            fingers_name,
        )
        if fingers_head is None:
            fingers_head = actual_wrist_tail.copy()
        fingers_tail = fingers_tail_target if fingers_tail_target is not None else actual_wrist_tail
        fingers_matrix = matrix_with_translation(target_base_local[fingers_name], fingers_head)
        desired_local[fingers_name] = aim_bone_tail_at(
            fingers_matrix,
            target.data.bones[fingers_name],
            fingers_tail,
            True,
        )
        actual_fingers_tail = bone_tail_from_matrix(
            desired_local[fingers_name],
            target.data.bones[fingers_name],
        )
        side_report["fingers"].append(
            {
                "bone": fingers_name,
                "head_target": vector_payload(fingers_head),
                "tail_target": vector_payload(fingers_tail),
                "tail_error": float((actual_fingers_tail - fingers_tail).length),
                "set_head": False,
            }
        )

        for chain_name, chain in FINGER_CHAINS.items():
            parent_name = fingers_name
            for base_name in chain:
                bone_name = f"{base_name}_{side}"
                if bone_name not in desired_local or bone_name not in target_base_local:
                    parent_name = bone_name
                    continue
                if parent_name not in desired_local:
                    parent_name = bone_name
                    continue

                head_target = point_from_parent_rest_offset(
                    target_base_local,
                    desired_local,
                    parent_name,
                    bone_name,
                )
                if head_target is None:
                    head_target = target_base_local[bone_name].translation.copy()
                tail_target = mapped_source_tail(
                    source_world_tails,
                    source_to_target_world,
                    target,
                    bone_name,
                )
                if tail_target is None:
                    tail_target = bone_tail_from_matrix(
                        target_base_local[bone_name],
                        target.data.bones[bone_name],
                    )

                finger_matrix = matrix_with_translation(target_base_local[bone_name], head_target)
                desired_local[bone_name] = aim_bone_tail_at(
                    finger_matrix,
                    target.data.bones[bone_name],
                    tail_target,
                    True,
                )
                actual_tail = bone_tail_from_matrix(
                    desired_local[bone_name],
                    target.data.bones[bone_name],
                )
                side_report["fingers"].append(
                    {
                        "bone": bone_name,
                        "chain": chain_name,
                        "head_target": vector_payload(head_target),
                        "tail_target": vector_payload(tail_target),
                        "tail_error": float((actual_tail - tail_target).length),
                        "set_head": False,
                    }
                )
                parent_name = bone_name

        solved.append(side_report)
    return solved


def summarize_forelimb_endpoint_rows(frame, solutions):
    rows = []
    for solution in solutions or []:
        side = solution.get("side")
        for section in ("arm", "fingers"):
            for item in solution.get(section, []):
                rows.append(
                    {
                        "frame": float(frame),
                        "side": side,
                        "section": section,
                        "bone": item.get("bone"),
                        "chain": item.get("chain"),
                        "tail_error": float(item.get("tail_error", 0.0)),
                        "target_residual": float(item.get("target_residual", 0.0)),
                    }
                )
    return rows


def summarize_forelimb_solution_rows(frame, solutions):
    rows = []
    for solution in solutions or []:
        rows.append(
            {
                "frame": float(frame),
                "side": solution.get("side"),
                "mode": solution.get("mode"),
                "source_wrist_tail_distance": float(
                    solution.get("source_wrist_tail_distance", 0.0)
                ),
                "clamped_wrist_tail_distance": float(
                    solution.get("clamped_wrist_tail_distance", 0.0)
                ),
                "base_reach": float(solution.get("base_reach", 0.0)),
                "max_reach": float(solution.get("max_reach", 0.0)),
                "arm_stretch_factor": float(solution.get("arm_stretch_factor", 1.0)),
            }
        )
    return rows


def numeric_summary(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(q):
        index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
        return float(ordered[index])

    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
    }


def collect_source_clip(source_path, source_armature_name, frame_step=1, max_frames=0):
    print(f"[collect] {source_path}")
    bpy.ops.wm.open_mainfile(filepath=str(source_path))

    scene = bpy.context.scene
    source = find_armature(source_armature_name, require_action=True)
    action = source.animation_data.action
    frames, interpolation = action_keyframes(action)
    frame_step = max(int(frame_step), 1)
    if frame_step > 1:
        frames = frames[::frame_step]
    if max_frames and max_frames > 0:
        frames = frames[: int(max_frames)]

    source_bones = set(source.data.bones.keys())
    source_rest_local = {
        name: source.data.bones[name].matrix_local.copy()
        for name in source_bones
    }

    per_frame = []
    for frame_index, frame in enumerate(frames, start=1):
        set_scene_frame(scene, frame)
        bpy.context.view_layer.update()

        deforms = {}
        world_heads = {}
        world_tails = {}
        for name in source_bones:
            pose_bone = source.pose.bones[name]
            pose_local = pose_bone.matrix
            deforms[name] = (pose_local @ source_rest_local[name].inverted()).copy()
            world_heads[name] = (source.matrix_world @ pose_local.translation).copy()
            world_tails[name] = (
                source.matrix_world
                @ (pose_local @ Vector((0.0, pose_bone.bone.length, 0.0)))
            ).copy()

        per_frame.append(
            {
                "frame": frame,
                "deforms": deforms,
                "world_heads": world_heads,
                "world_tails": world_tails,
            }
        )
        if frame_index == 1 or frame_index == len(frames) or frame_index % 50 == 0:
            print(f"  collected {frame_index}/{len(frames)} frames")

    return {
        "source": str(source_path),
        "source_armature": source.name,
        "source_action": action.name,
        "frames": frames,
        "interpolation": interpolation,
        "bone_names": sorted(source_bones),
        "per_frame": per_frame,
    }


def set_action_interpolation(action, interpolation):
    for fc in action_fcurves(action):
        for key in fc.keyframe_points:
            key.interpolation = interpolation


def bake_clip_to_target(
    clip,
    target,
    target_reference_local,
    target_reference_channels,
    report,
    scale_mode,
    alignment_anchor,
    position_lock_bone,
    preserve_tail,
    preserve_head,
    allow_global_scale,
    connected_aim_mode,
    rotation_only,
    animated_bone_set,
    forelimb_target_mode,
    pipe_snap_object,
    pipe_snap_threshold,
    cup_clearance_objects,
    cup_clearance_distance,
    cup_clearance_max_push,
    cup_front_clearance_distance,
    cup_front_clearance_max_push,
    forelimb_min_reach_ratio,
    forelimb_max_arm_stretch,
    forelimb_palm_source_endpoint,
):
    target.animation_data_create()
    action_name = unique_action_name(sanitize_action_name(clip["source"]))
    action = bpy.data.actions.new(action_name)
    action.use_fake_user = True
    target.animation_data.action = action

    clear_pose(target)
    bpy.context.view_layer.update()

    target_bones = set(target.data.bones.keys())
    source_bones = set(clip["bone_names"])
    common_bones = sorted(source_bones & target_bones)
    missing_in_target = sorted(source_bones - target_bones)
    extra_in_target = sorted(target_bones - source_bones)
    ordered_bones = parent_first_bone_names(target, common_bones)
    ordered_bone_groups = bone_depth_groups(target, common_bones)
    if animated_bone_set == "forelimbs":
        animated_bones = descendant_bone_names(target, ("Scapula_L", "Scapula_R")) & set(common_bones)
    elif animated_bone_set == "all":
        animated_bones = set(common_bones)
    else:
        raise RuntimeError(f"Unknown animated bone set: {animated_bone_set}")

    target_base_local = {
        name: target_reference_local[name].copy()
        for name in ordered_bones
    }
    target_base_world_heads = {
        name: target.matrix_world @ target_base_local[name].translation
        for name in ordered_bones
    }
    target_body_frame = body_frame_from_matrices(target_base_local)
    target_head_frame = target_base_local["Head1_M"].copy()
    clip_body_alignment = None
    clip_alignment_scale = 1.0
    clip_alignment_point_count = 0
    source_to_target_world = None
    source_to_target_world_scale = 1.0
    source_to_target_world_point_count = 0
    forelimb_ik_solutions = []
    pipe_vertices_world = (
        mesh_vertices_world(bpy.data.objects.get(pipe_snap_object))
        if pipe_snap_threshold > 0.0
        else []
    )
    cup_bvhs = [
        record
        for record in (
            mesh_bvh_world(bpy.data.objects.get(name))
            for name in cup_clearance_objects
        )
        if record is not None
    ]

    first_world_heads = clip["per_frame"][0].get("world_heads", {})
    source_to_target_world, source_to_target_world_scale, source_to_target_world_point_count = (
        similarity_matrix_from_matrices(
            first_world_heads,
            target_base_world_heads,
            BODY_TAIL_ALIGNMENT_BONES,
            allow_global_scale,
        )
    )

    max_assignment_error = 0.0
    max_assignment_error_at = None
    max_head_error = 0.0
    max_head_error_at = None
    max_tail_error = 0.0
    max_tail_error_at = None
    aimed_parent_names = []
    forelimb_endpoint_rows = []
    forelimb_solution_rows = []
    forelimb_pose_scale_bones = set()
    if (
        rotation_only
        and forelimb_target_mode == "source-tail-constrained-ik"
        and forelimb_max_arm_stretch > 1.0
    ):
        forelimb_pose_scale_bones = {
            f"{base}_{side}"
            for side in ("L", "R")
            for base in ("Scapula", "Wrist")
        }
    scene = bpy.context.scene
    print(f"[bake] {clip['source']} -> {action.name}, {len(clip['frames'])} frames, {len(ordered_bones)} bones")

    for frame_index, frame_payload in enumerate(clip["per_frame"], start=1):
        frame = frame_payload["frame"]
        set_scene_frame(scene, frame)

        desired_local = {}
        for name in ordered_bones:
            desired_local[name] = frame_payload["deforms"][name] @ target_base_local[name]
        if clip_body_alignment is None:
            if alignment_anchor == "body":
                raw_frame = body_frame_from_matrices(desired_local)
                target_frame = target_body_frame
                clip_alignment_scale = 1.0
                clip_alignment_point_count = 6
                clip_body_alignment = target_frame @ raw_frame.inverted()
            elif alignment_anchor == "body-tail":
                clip_body_alignment, clip_alignment_scale, clip_alignment_point_count = (
                    similarity_matrix_from_matrices(
                        desired_local,
                        target_base_local,
                        BODY_TAIL_ALIGNMENT_BONES,
                        allow_global_scale,
                    )
                )
                print(
                    f"  body-tail alignment: points={clip_alignment_point_count}, "
                    f"scale={clip_alignment_scale:.6f}"
                )
                raw_frame = None
                target_frame = None
            elif alignment_anchor == "head":
                raw_frame = desired_local["Head1_M"]
                target_frame = target_head_frame
                clip_alignment_scale = 1.0
                clip_alignment_point_count = 1
                clip_body_alignment = target_frame @ raw_frame.inverted()
            elif alignment_anchor == "none":
                raw_frame = Matrix.Identity(4)
                target_frame = Matrix.Identity(4)
                clip_alignment_scale = 1.0
                clip_alignment_point_count = 0
                clip_body_alignment = target_frame @ raw_frame.inverted()
            else:
                raise RuntimeError(f"Unknown alignment anchor: {alignment_anchor}")
        for name in ordered_bones:
            desired_local[name] = sanitize_scale(
                clip_body_alignment @ desired_local[name],
                scale_mode,
            )
        source_aligned_local = {
            name: desired_local[name].copy()
            for name in ordered_bones
        }
        if position_lock_bone:
            if position_lock_bone not in desired_local:
                raise RuntimeError(f"Position lock bone not found: {position_lock_bone}")
            delta = (
                target_base_local[position_lock_bone].translation
                - desired_local[position_lock_bone].translation
            )
            translate_matrices_in_place(desired_local, ordered_bones, delta)
        if preserve_tail:
            for tail_index in range(12):
                tail_name = f"Tail{tail_index}_M"
                if tail_name in desired_local and tail_name in target_base_local:
                    desired_local[tail_name] = target_base_local[tail_name].copy()
        if preserve_head:
            for head_name in HEAD_REFERENCE_BONES:
                if head_name in desired_local and head_name in target_base_local:
                    desired_local[head_name] = target_base_local[head_name].copy()
        if animated_bone_set == "forelimbs":
            for name in ordered_bones:
                if name not in animated_bones:
                    desired_local[name] = target_base_local[name].copy()
        if forelimb_target_mode in ("source-palm-ik", "source-world-palm-ik"):
            source_world_heads = (
                frame_payload.get("world_heads")
                if forelimb_target_mode == "source-world-palm-ik"
                else None
            )
            source_world_tails = (
                frame_payload.get("world_tails")
                if forelimb_target_mode == "source-world-palm-ik"
                else None
            )
            forelimb_ik_solutions = apply_forelimb_source_palm_ik(
                desired_local,
                source_aligned_local,
                target_base_local,
                target,
                source_world_heads,
                source_world_tails,
                source_to_target_world,
                pipe_vertices_world,
                pipe_snap_threshold,
                cup_bvhs,
                cup_clearance_distance,
                cup_clearance_max_push,
                cup_front_clearance_distance,
                cup_front_clearance_max_push,
                forelimb_min_reach_ratio,
            )
        elif forelimb_target_mode == "source-endpoint-chain-ik":
            forelimb_ik_solutions = apply_forelimb_source_endpoint_chain_ik(
                desired_local,
                target_base_local,
                target,
                frame_payload.get("world_heads"),
                frame_payload.get("world_tails"),
                source_to_target_world,
                forelimb_min_reach_ratio,
                forelimb_palm_source_endpoint,
            )
        elif forelimb_target_mode == "source-tail-constrained-ik":
            forelimb_ik_solutions = apply_forelimb_source_tail_constrained_ik(
                desired_local,
                target_base_local,
                target,
                frame_payload.get("world_tails"),
                source_to_target_world,
                forelimb_min_reach_ratio,
                forelimb_max_arm_stretch,
            )
        elif forelimb_target_mode == "source-tail-shape-ik":
            forelimb_ik_solutions = apply_forelimb_source_tail_shape_ik(
                desired_local,
                target_base_local,
                target,
                frame_payload.get("world_heads"),
                frame_payload.get("world_tails"),
                source_to_target_world,
                forelimb_min_reach_ratio,
            )
        elif forelimb_target_mode == "rotation":
            forelimb_ik_solutions = []
        else:
            raise RuntimeError(f"Unknown forelimb target mode: {forelimb_target_mode}")
        forelimb_endpoint_rows.extend(summarize_forelimb_endpoint_rows(frame, forelimb_ik_solutions))
        forelimb_solution_rows.extend(summarize_forelimb_solution_rows(frame, forelimb_ik_solutions))
        aimed_parent_names = aim_connected_parents(
            desired_local,
            target,
            ordered_bones,
            connected_aim_mode,
        )

        for bone_group in ordered_bone_groups:
            for name in bone_group:
                if rotation_only:
                    set_pose_matrix_rotation_only(
                        target,
                        name,
                        desired_local[name],
                        target_reference_channels,
                        name == "Root_M",
                        name in forelimb_pose_scale_bones,
                    )
                else:
                    set_pose_matrix_via_basis(target, name, desired_local[name])
            bpy.context.view_layer.update()
        if rotation_only and position_lock_bone:
            lock_pose_by_translating_root(
                target,
                "Root_M",
                position_lock_bone,
                target_base_local[position_lock_bone].translation,
                target_reference_channels,
            )
            bpy.context.view_layer.update()

        for name in ordered_bones:
            actual = target.pose.bones[name].matrix
            error = matrix_error(actual, desired_local[name])
            if error > max_assignment_error:
                max_assignment_error = error
                max_assignment_error_at = {"frame": frame, "bone": name}

            desired_head = desired_local[name].translation
            actual_head = actual.translation
            head_error = (actual_head - desired_head).length
            if head_error > max_head_error:
                max_head_error = head_error
                max_head_error_at = {"frame": frame, "bone": name}

            bone = target.data.bones[name]
            desired_tail = bone_tail_from_matrix(desired_local[name], bone)
            actual_tail = bone_tail_from_matrix(actual, bone)
            tail_error = (actual_tail - desired_tail).length
            if tail_error > max_tail_error:
                max_tail_error = tail_error
                max_tail_error_at = {"frame": frame, "bone": name}

        for name in ordered_bones:
            pb = target.pose.bones[name]
            pb.keyframe_insert(data_path="location", frame=frame)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            pb.keyframe_insert(data_path="scale", frame=frame)

        if frame_index == 1 or frame_index == len(clip["frames"]) or frame_index % 50 == 0:
            print(f"  baked {frame_index}/{len(clip['frames'])} frames")

    set_action_interpolation(action, clip["interpolation"])
    forelimb_tail_errors = [row["tail_error"] for row in forelimb_endpoint_rows]
    forelimb_target_residuals = [
        row["target_residual"]
        for row in forelimb_endpoint_rows
        if row.get("target_residual", 0.0) > 0.0
    ]
    forelimb_arm_stretch_factors = [
        row["arm_stretch_factor"]
        for row in forelimb_solution_rows
        if row.get("mode") in {"source-tail-constrained-ik", "source-tail-shape-ik"}
    ]
    forelimb_top_endpoint_errors = sorted(
        forelimb_endpoint_rows,
        key=lambda row: row["tail_error"],
        reverse=True,
    )[:40]
    forelimb_top_target_residuals = sorted(
        forelimb_endpoint_rows,
        key=lambda row: row.get("target_residual", 0.0),
        reverse=True,
    )[:40]
    forelimb_top_arm_stretches = sorted(
        forelimb_solution_rows,
        key=lambda row: row.get("arm_stretch_factor", 1.0),
        reverse=True,
    )[:40]
    report["clips"].append(
        {
            "source": clip["source"],
            "source_armature": clip["source_armature"],
            "source_action": clip["source_action"],
            "target_armature": target.name,
            "target_action": action.name,
            "frames": [clip["frames"][0], clip["frames"][-1]],
            "frame_count": len(clip["frames"]),
            "common_bone_count": len(common_bones),
            "missing_in_target": missing_in_target,
            "extra_in_target": extra_in_target,
            "interpolation": clip["interpolation"],
            "max_assignment_matrix_error": max_assignment_error,
            "max_assignment_matrix_error_at": max_assignment_error_at,
            "max_head_error": max_head_error,
            "max_head_error_at": max_head_error_at,
            "max_tail_error": max_tail_error,
            "max_tail_error_at": max_tail_error_at,
            "aimed_parent_count": len(aimed_parent_names),
            "aimed_parent_names": aimed_parent_names,
            "scale_mode": scale_mode,
            "alignment_anchor": alignment_anchor,
            "position_lock_bone": position_lock_bone,
            "preserve_tail": preserve_tail,
            "preserve_head": preserve_head,
            "allow_global_scale": allow_global_scale,
            "global_alignment_scale": clip_alignment_scale,
            "global_alignment_point_count": clip_alignment_point_count,
            "source_to_target_world_scale": source_to_target_world_scale,
            "source_to_target_world_point_count": source_to_target_world_point_count,
            "connected_aim_mode": connected_aim_mode,
            "rotation_only": rotation_only,
            "animated_bone_set": animated_bone_set,
            "animated_bone_count": len(animated_bones),
            "forelimb_target_mode": forelimb_target_mode,
            "forelimb_endpoint_error_summary": numeric_summary(forelimb_tail_errors),
            "forelimb_target_residual_summary": numeric_summary(forelimb_target_residuals),
            "forelimb_arm_stretch_summary": numeric_summary(forelimb_arm_stretch_factors),
            "forelimb_top_endpoint_errors": forelimb_top_endpoint_errors,
            "forelimb_top_target_residuals": forelimb_top_target_residuals,
            "forelimb_top_arm_stretches": forelimb_top_arm_stretches,
            "forelimb_ik_solutions_last_frame": forelimb_ik_solutions,
            "pipe_snap_object": pipe_snap_object,
            "pipe_snap_threshold": pipe_snap_threshold,
            "cup_clearance_objects": cup_clearance_objects,
            "cup_clearance_distance": cup_clearance_distance,
            "cup_clearance_max_push": cup_clearance_max_push,
            "cup_front_clearance_distance": cup_front_clearance_distance,
            "cup_front_clearance_max_push": cup_front_clearance_max_push,
            "forelimb_min_reach_ratio": forelimb_min_reach_ratio,
            "forelimb_max_arm_stretch": forelimb_max_arm_stretch,
            "body_alignment_matrix": [
                float(value) for row in clip_body_alignment for value in row
            ]
            if clip_body_alignment
            else None,
        }
    )
    return action


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--target-armature", default=DEFAULT_TARGET_ARMATURE)
    parser.add_argument("--source-armature", default=DEFAULT_SOURCE_ARMATURE)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--scale-mode",
        choices=["full", "keep-y", "none"],
        default="none",
        help="How to transfer non-uniform pose scale. keep-y preserves bone-length scale while clearing cross-section squash.",
    )
    parser.add_argument(
        "--alignment-anchor",
        choices=["body", "body-tail", "head", "none"],
        default="body-tail",
        help="Constant clip alignment anchor. body-tail fits torso, hips, head base, and tail base while excluding forelimbs.",
    )
    parser.add_argument(
        "--position-lock-bone",
        default="Tail0_M",
        help="After alignment, translate the whole clip so this bone matches the IK target rest position. Use 'none' to disable.",
    )
    parser.add_argument(
        "--preserve-tail",
        action="store_true",
        help="Keep Tail0_M..Tail11_M in the IK target rest pose so it stays threaded through the cup slot.",
    )
    parser.add_argument(
        "--preserve-head",
        action="store_true",
        help="Keep Head1_M/HeadEnd_M/Jaw in the IK target reference pose so the cranial window relationship stays fixed.",
    )
    parser.add_argument(
        "--no-global-scale",
        action="store_true",
        help="Disable uniform scale in the body-tail similarity fit.",
    )
    parser.add_argument(
        "--connected-aim-mode",
        choices=["rotate", "stretch"],
        default="rotate",
        help="For connected parents, rotate toward children without changing bone length, or preserve the older stretching behavior.",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["rotation-only", "full-pose"],
        default="rotation-only",
        help="rotation-only keeps non-root bone location and scale at the IK reference pose to avoid claw and toe stretching.",
    )
    parser.add_argument(
        "--animated-bone-set",
        choices=["all", "forelimbs"],
        default="all",
        help="forelimbs transfers only Scapula_L/R descendants and keeps torso, hips, head, hindlimbs, and tail at the IK reference pose.",
    )
    parser.add_argument(
        "--forelimb-target-mode",
        choices=[
            "rotation",
            "source-palm-ik",
            "source-world-palm-ik",
            "source-endpoint-chain-ik",
            "source-tail-constrained-ik",
            "source-tail-shape-ik",
        ],
        default="rotation",
        help="source-tail-shape-ik normalizes source bone-tail shape onto old-IK lengths; constrained mode chases tail positions.",
    )
    parser.add_argument("--pipe-snap-object", default="Part_WaterPipe")
    parser.add_argument(
        "--pipe-snap-threshold",
        type=float,
        default=0.0,
        help="When positive, source-world finger tips within this world-space distance of the water pipe snap to the nearest pipe vertex.",
    )
    parser.add_argument(
        "--cup-clearance-object",
        action="append",
        default=[],
        help="Cup or rail mesh used to push front-paw IK probe points out of collision.",
    )
    parser.add_argument(
        "--cup-clearance-distance",
        type=float,
        default=0.0,
        help="World-space clearance for palm/fingertip probe points around cup geometry. Disabled at 0.",
    )
    parser.add_argument(
        "--cup-clearance-max-push",
        type=float,
        default=0.0,
        help="Maximum world-space palm-target correction per frame. 0 means uncapped.",
    )
    parser.add_argument(
        "--cup-front-clearance-distance",
        type=float,
        default=0.0,
        help="Require fingertip probe points to stay this far in front of the cup max-Y rail plane.",
    )
    parser.add_argument(
        "--cup-front-clearance-max-push",
        type=float,
        default=0.0,
        help="Maximum +Y palm-target correction from the front rail plane. 0 means uncapped.",
    )
    parser.add_argument(
        "--forelimb-min-reach-ratio",
        type=float,
        default=0.0,
        help="Minimum shoulder-to-palm distance as a fraction of the two-bone forelimb chain length.",
    )
    parser.add_argument(
        "--forelimb-max-arm-stretch",
        type=float,
        default=1.0,
        help="Maximum Scapula/Wrist pose-length scale for source-tail-constrained-ik. 1 disables arm stretch.",
    )
    parser.add_argument(
        "--forelimb-palm-source-endpoint",
        choices=["fingers-head", "wrist-tail"],
        default="fingers-head",
        help="Palm target endpoint for source-endpoint-chain-ik. fingers-head preserves V42 behavior; wrist-tail tests the source Wrist tail as old-IK palm.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    target_path = Path(args.target) if args.target else script_dir / DEFAULT_TARGET
    output_path = (
        Path(args.output)
        if args.output
        else target_path.with_name(target_path.stem + "-migrated-actions.blend")
    )
    report_path = (
        Path(args.report)
        if args.report
        else target_path.with_name(target_path.stem + "-migration-report.json")
    )

    if args.source:
        source_paths = [Path(path) for path in args.source]
    else:
        source_paths = sorted(
            path
            for path in script_dir.glob("*.blend")
            if path.name != target_path.name and "FrameToFrame" in path.name
        )

    if not source_paths:
        raise RuntimeError("No source .blend files found.")

    clips = [
        collect_source_clip(source_path.resolve(), args.source_armature, args.frame_step, args.max_frames)
        for source_path in source_paths
    ]

    print(f"[target] {target_path}")
    bpy.ops.wm.open_mainfile(filepath=str(target_path.resolve()))
    target = find_armature(args.target_armature)
    if target.name != args.target_armature:
        print(f"[target] requested {args.target_armature!r}, using {target.name!r}")
    target_reference_local = {
        pb.name: pb.matrix.copy()
        for pb in target.pose.bones
    }
    target_reference_channels = {
        pb.name: {
            "location": pb.location.copy(),
            "scale": pb.scale.copy(),
        }
        for pb in target.pose.bones
    }

    report = {
        "target": str(target_path.resolve()),
        "output": str(output_path.resolve()),
        "target_armature": target.name,
        "retarget_space": "ARMATURE_LOCAL",
        "scale_mode": args.scale_mode,
        "alignment_anchor": args.alignment_anchor,
        "position_lock_bone": None
        if args.position_lock_bone.lower() == "none"
        else args.position_lock_bone,
        "preserve_tail": args.preserve_tail,
        "preserve_head": args.preserve_head,
        "allow_global_scale": not args.no_global_scale,
        "connected_aim_mode": args.connected_aim_mode,
        "transfer_mode": args.transfer_mode,
        "animated_bone_set": args.animated_bone_set,
        "forelimb_target_mode": args.forelimb_target_mode,
        "pipe_snap_object": args.pipe_snap_object,
        "pipe_snap_threshold": args.pipe_snap_threshold,
        "cup_clearance_objects": args.cup_clearance_object,
        "cup_clearance_distance": args.cup_clearance_distance,
        "cup_clearance_max_push": args.cup_clearance_max_push,
        "cup_front_clearance_distance": args.cup_front_clearance_distance,
        "cup_front_clearance_max_push": args.cup_front_clearance_max_push,
        "forelimb_min_reach_ratio": args.forelimb_min_reach_ratio,
        "forelimb_max_arm_stretch": args.forelimb_max_arm_stretch,
        "forelimb_palm_source_endpoint": args.forelimb_palm_source_endpoint,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "clips": [],
    }

    first_action = None
    position_lock_bone = (
        None if args.position_lock_bone.lower() == "none" else args.position_lock_bone
    )
    for clip in clips:
        action = bake_clip_to_target(
            clip,
            target,
            target_reference_local,
            target_reference_channels,
            report,
            args.scale_mode,
            args.alignment_anchor,
            position_lock_bone,
            args.preserve_tail,
            args.preserve_head,
            not args.no_global_scale,
            args.connected_aim_mode,
            args.transfer_mode == "rotation-only",
            args.animated_bone_set,
            args.forelimb_target_mode,
            args.pipe_snap_object,
            args.pipe_snap_threshold,
            args.cup_clearance_object,
            args.cup_clearance_distance,
            args.cup_clearance_max_push,
            args.cup_front_clearance_distance,
            args.cup_front_clearance_max_push,
            args.forelimb_min_reach_ratio,
            args.forelimb_max_arm_stretch,
            args.forelimb_palm_source_endpoint,
        )
        if first_action is None:
            first_action = action

    if first_action:
        target.animation_data.action = first_action
        bpy.context.scene.frame_start = int(math.floor(min(c["frames"][0] for c in clips)))
        bpy.context.scene.frame_end = int(math.ceil(max(c["frames"][-1] for c in clips)))

    hide_reference_mouse_objects()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.resolve()))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done]")
    print(f"output={output_path.resolve()}")
    print(f"report={report_path.resolve()}")


if __name__ == "__main__":
    main()

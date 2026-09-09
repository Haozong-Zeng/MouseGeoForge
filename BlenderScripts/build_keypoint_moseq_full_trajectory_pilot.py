#!/usr/bin/env python3
"""Build a short full-trajectory Keypoint-MoSeq retargeting pilot in Blender."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


DEFAULT_KEYPOINTS = Path(
    "/home/haozong/projects/MouseGeoForge/dataset/Cylinder/"
    "11_01_22_12wk_m0.keypoints.npy"
)
DEFAULT_OUTPUT_NAME = (
    "PyMouseStudio-WhiteFur-Controller-moseq-"
    "FullTrajectoryPilot-f21060-21239.blend"
)

ARMATURE_NAME = "DeformationSystem.001"
MOUSE_MESH_NAME = "MS_Mouse.001"
CONTROL_COLLECTION = "MGF_FullTrajectory_Controls"
SOURCE_START = 21060
SOURCE_FRAMES = 180
SOURCE_FPS = 30.0
FLOOR_Z = 339.5
CYLINDER_DIAMETER = 380.0
CYLINDER_INNER_RADIUS = 187.0
HORIZONTAL_CAMERA_LENS = 30.0
CAMERA_FRAME_MARGIN = 0.03
SUPPORT_COLLECTION = "MGF_Redesigned_CameraSupport"
SUPPORT_Z_CENTER = 322.5
SUPPORT_THICKNESS = 10.0
HIND_MAX_REACH = 28.0
HIND_MIN_LATERAL = 8.0
HIND_MAX_LATERAL = 18.0
HIND_MIN_FORWARD = -8.0
HIND_MAX_FORWARD = 22.0
HIND_MAX_HEIGHT = FLOOR_Z + 3.5

CAMERA_NAMES = tuple(f"Camera.{index:03d}" for index in range(1, 9))
RIG_COLLECTION_NAMES = {
    1: "MGF_Rig_Camera.001_Horizontal_Front",
    2: "MGF_Rig_Camera.002_Horizontal_BackLeft",
    3: "MGF_Rig_Camera.003_Horizontal_BackRight",
    4: "MGF_Rig_Camera.004_Horizontal_Back",
    5: "MGF_Rig_Camera.005_Horizontal_FrontRight",
    6: "MGF_Rig_Camera.006_Horizontal_FrontLeft",
    7: "MGF_Rig_Camera.007_Bottom",
}

SPINE_SOURCE_INDICES = (3, 2, 1, 0, 5, 8)
SPINE_CURVE_NAME = "Spine_NURBSCurve.001"
TAIL_CURVE_NAME = "Tail_NURBSCurve.001"


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-start", type=int, default=SOURCE_START)
    parser.add_argument("--frames", type=int, default=SOURCE_FRAMES)
    parser.add_argument("--source-stride", type=int, default=1)
    return parser.parse_args(args_after_dash())


def smooth_time(values: np.ndarray) -> np.ndarray:
    """Apply a short symmetric binomial filter without shifting endpoints."""
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    padded = np.pad(values, ((2, 2), (0, 0), (0, 0)), mode="edge")
    result = np.zeros_like(values, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        result += weight * padded[offset : offset + len(values)]
    return result


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    collection.hide_render = True
    collection.hide_viewport = False
    return collection


def remove_previous_controls() -> None:
    collection = bpy.data.collections.get(CONTROL_COLLECTION)
    if collection is not None:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)
    for curve_name in (SPINE_CURVE_NAME, TAIL_CURVE_NAME):
        curve = bpy.data.objects.get(curve_name)
        if curve is None:
            continue
        for modifier in list(curve.modifiers):
            if modifier.name.startswith("MGF_FT_Hook_"):
                curve.modifiers.remove(modifier)


def make_curve_visual_controls(curve_name: str, prefix: str) -> list[bpy.types.Object]:
    curve = bpy.data.objects[curve_name]
    spline = curve.data.splines[0]
    collection = ensure_collection(CONTROL_COLLECTION)
    controls = []
    for index, point in enumerate(spline.points):
        control = bpy.data.objects.new(f"MGF_FT_{prefix}_{index:02d}", None)
        collection.objects.link(control)
        control.empty_display_type = "SPHERE"
        control.empty_display_size = 2.5
        control.show_in_front = True
        control.matrix_world.translation = curve.matrix_world @ point.co.to_3d()
        controls.append(control)
    bpy.context.view_layer.update()
    return controls


def set_world_location(obj: bpy.types.Object, world: Vector, frame: int) -> None:
    if obj.parent is None:
        obj.location = world - obj.delta_location
    else:
        local = obj.parent.matrix_world.inverted() @ world
        obj.location = local - obj.delta_location
    obj.keyframe_insert(data_path="location", frame=frame)


def set_empty_world_location(obj: bpy.types.Object, world: Vector, frame: int) -> None:
    obj.location = world
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


def map_source(point: np.ndarray, arena_center: np.ndarray) -> Vector:
    return Vector(
        (
            float(point[0] - arena_center[0]),
            float(point[1] - arena_center[1]),
            float(FLOOR_Z + point[2]),
        )
    )


def safe_direction(start: Vector, end: Vector, fallback: Vector) -> Vector:
    direction = end - start
    if direction.length < 1e-6:
        return fallback.normalized()
    return direction.normalized()


def resample_polyline(points: list[Vector], fractions: list[float]) -> list[Vector]:
    lengths = [(points[index + 1] - points[index]).length for index in range(len(points) - 1)]
    total = sum(lengths)
    if total < 1e-6:
        return [points[0].copy() for _ in fractions]
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    result = []
    for fraction in fractions:
        target = max(0.0, min(1.0, fraction)) * total
        segment = next(
            (index for index in range(len(lengths)) if cumulative[index + 1] >= target),
            len(lengths) - 1,
        )
        segment_length = max(lengths[segment], 1e-9)
        alpha = (target - cumulative[segment]) / segment_length
        result.append(points[segment].lerp(points[segment + 1], alpha))
    return result


def normalized_spine_controls(
    frame_points: np.ndarray,
    arena_center: np.ndarray,
    fixed_source_lengths: list[float],
    curve_fractions: list[float],
) -> list[Vector]:
    raw = [map_source(frame_points[index], arena_center) for index in SPINE_SOURCE_INDICES]
    rebuilt = [raw[0]]
    fallback = Vector((-1.0, 0.0, 0.0))
    for index, length in enumerate(fixed_source_lengths):
        direction = safe_direction(raw[index], raw[index + 1], fallback)
        rebuilt.append(rebuilt[-1] + direction * length)
        fallback = direction
    controls = resample_polyline(rebuilt, curve_fractions)
    root_lift = max(0.0, FLOOR_Z + 12.0 - controls[0].z)
    if root_lift:
        for point in controls:
            point.z += root_lift
    for index, point in enumerate(controls):
        point.z = max(point.z, FLOOR_Z + (5.0 if index == 0 else 2.0))
    return controls


def limit_xy(point: Vector, radius: float, audit: dict[str, int]) -> Vector:
    radial = math.hypot(point.x, point.y)
    if radial <= radius or radial < 1e-9:
        return point
    scale = radius / radial
    audit["wall_clamps"] += 1
    return Vector((point.x * scale, point.y * scale, point.z))


def floor_clamp(point: Vector, minimum_z: float, audit: dict[str, int]) -> Vector:
    if point.z >= minimum_z:
        return point
    audit["floor_clamps"] += 1
    return Vector((point.x, point.y, minimum_z))


def closest_point_on_segment(point: Vector, start: Vector, end: Vector) -> Vector:
    segment = end - start
    denominator = segment.length_squared
    if denominator < 1e-9:
        return start.copy()
    alpha = max(0.0, min(1.0, (point - start).dot(segment) / denominator))
    return start + segment * alpha


def push_wrist_from_body(
    wrist: Vector,
    spine: list[Vector],
    audit: dict[str, int],
) -> Vector:
    if wrist.z <= FLOOR_Z + 7.0:
        return wrist
    nearest = min(
        (closest_point_on_segment(wrist, spine[index], spine[index + 1]) for index in range(len(spine) - 1)),
        key=lambda point: (wrist - point).length,
    )
    offset = wrist - nearest
    minimum_distance = 8.0
    if offset.length >= minimum_distance:
        return wrist
    if offset.length < 1e-6:
        tangent = spine[-1] - spine[0]
        offset = Vector((-tangent.y, tangent.x, 0.0)).normalized()
    else:
        offset.normalize()
    audit["body_proxy_pushes"] += 1
    return nearest + offset * minimum_distance


def constrain_hind_target(
    target: Vector,
    root: Vector,
    forward: Vector,
    lateral: Vector,
    side_sign: float,
    audit: dict[str, int],
) -> Vector:
    relative = target - root
    forward_value = max(HIND_MIN_FORWARD, min(HIND_MAX_FORWARD, relative.dot(forward)))
    lateral_value = relative.dot(lateral)
    if side_sign > 0.0:
        constrained_lateral = max(HIND_MIN_LATERAL, min(HIND_MAX_LATERAL, lateral_value))
    else:
        constrained_lateral = min(-HIND_MIN_LATERAL, max(-HIND_MAX_LATERAL, lateral_value))
    if abs(constrained_lateral - lateral_value) > 1e-5:
        audit["hind_lateral_clamps"] += 1

    height = max(FLOOR_Z + 3.0, min(HIND_MAX_HEIGHT, target.z))
    if abs(height - target.z) > 1e-5:
        audit["hind_height_clamps"] += 1
    constrained = root + forward * forward_value + lateral * constrained_lateral
    constrained.z = height
    offset = constrained - root
    if offset.length > HIND_MAX_REACH:
        constrained = root + offset.normalized() * HIND_MAX_REACH
        audit["hind_reach_clamps"] += 1
    return constrained


def build_tail_controls(
    base: Vector,
    tip_raw: Vector,
    body_heading: Vector,
    target_length: float,
    previous_relative_mid: Vector | None,
    audit: dict[str, int],
) -> tuple[list[Vector], Vector]:
    tip = floor_clamp(tip_raw, FLOOR_Z + 0.8, audit)
    chord = tip - base
    if chord.length > target_length:
        tip = base + chord.normalized() * target_length
        chord = tip - base
        audit["tail_length_clamps"] += 1
    if chord.length < 1e-6:
        chord = -body_heading.normalized() * target_length
        tip = base + chord

    half_chord = chord.length * 0.5
    half_length = target_length * 0.5
    bend = math.sqrt(max(0.0, half_length * half_length - half_chord * half_chord))
    perpendicular = Vector((-chord.y, chord.x, 0.0))
    if perpendicular.length < 1e-6:
        perpendicular = Vector((0.0, 1.0, 0.0))
    perpendicular.normalize()
    midpoint = base + chord * 0.5
    candidates = [midpoint + perpendicular * bend, midpoint - perpendicular * bend]
    if previous_relative_mid is None:
        head_reference = base + body_heading.normalized() * 30.0
        middle = max(candidates, key=lambda point: (point - head_reference).length)
    else:
        middle = min(candidates, key=lambda point: ((point - base) - previous_relative_mid).length)
    middle.z = max(middle.z, FLOOR_Z + 0.8)
    tip = limit_xy(tip, CYLINDER_INNER_RADIUS - 2.0, audit)
    middle = limit_xy(middle, CYLINDER_INNER_RADIUS - 2.0, audit)
    return [base, middle, tip], middle - base


def curve_world_points(curve: bpy.types.Object) -> list[Vector]:
    return [curve.matrix_world @ point.co.to_3d() for point in curve.data.splines[0].points]


def configure_tail_and_ik(armature: bpy.types.Object) -> dict[str, object]:
    tail_names = ["TailStart_M"] + [f"Tail{index}_M" for index in range(12)]
    tail_arc = sum(
        ((armature.matrix_world @ armature.pose.bones[name].tail) - (armature.matrix_world @ armature.pose.bones[name].head)).length
        for name in tail_names
    )
    ik_chain_counts = {}
    pole_angles = {"L": 3.0 * math.pi / 4.0, "R": math.pi}
    pole_objects = {}
    control_collection = ensure_collection(CONTROL_COLLECTION)
    for side in ("L", "R"):
        pole = bpy.data.objects.new(f"MGF_FT_KneePole_{side}", None)
        control_collection.objects.link(pole)
        pole.empty_display_type = "SPHERE"
        pole.empty_display_size = 3.0
        pole.show_in_front = True
        pole_objects[side] = pole
    for name in ("Wrist_L", "Wrist_R", "Ankle_L", "Ankle_R"):
        for constraint in armature.pose.bones[name].constraints:
            if constraint.type == "IK":
                constraint.use_stretch = False
                constraint.mute = False
                constraint.influence = 1.0
                if name.startswith("Ankle_"):
                    # The hind limb is exactly ankle -> knee -> hip.  The donor
                    # value of six reaches through Root_M and lets a lifted paw
                    # flip the leg across the back while also moving the torso.
                    constraint.chain_count = 3
                    side = name[-1]
                    constraint.pole_target = pole_objects[side]
                    constraint.pole_subtarget = ""
                    constraint.pole_angle = pole_angles[side]
                ik_chain_counts[name] = constraint.chain_count
    tail_constraint = next(
        constraint
        for constraint in armature.pose.bones["Tail11_M"].constraints
        if constraint.type == "SPLINE_IK"
    )
    tail_constraint.y_scale_mode = "FIT_CURVE"
    tail_constraint.xz_scale_mode = "NONE"
    tail_constraint.use_curve_radius = True
    tail_constraint.mute = False
    return {
        "original_tail_arc": tail_arc,
        "ik_chain_counts": ik_chain_counts,
        "ik_stretch_disabled": True,
        "hind_knee_poles": {side: obj.name for side, obj in pole_objects.items()},
        "hind_knee_pole_angles": pole_angles,
    }


def scale_xy_to_dimensions(obj: bpy.types.Object, x: float, y: float) -> None:
    if obj.dimensions.x > 1e-6:
        obj.scale.x *= x / obj.dimensions.x
    if obj.dimensions.y > 1e-6:
        obj.scale.y *= y / obj.dimensions.y


def mesh_world_points(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]


def radial_bounds(obj: bpy.types.Object) -> tuple[float, float]:
    radii = [math.hypot(point.x, point.y) for point in mesh_world_points(obj)]
    return min(radii), max(radii)


def configure_environment() -> dict[str, object]:
    cylinder = bpy.data.objects["Part_GlassCylinder160mm"]
    square_glass = bpy.data.objects["Part_AntiReflectiveGlass"]
    circle_glass = bpy.data.objects["Part_AntiReflectiveGlass_Circle"]

    scale_xy_to_dimensions(cylinder, CYLINDER_DIAMETER, CYLINDER_DIAMETER)
    scale_xy_to_dimensions(square_glass, 420.0, 420.0)
    scale_xy_to_dimensions(circle_glass, 600.0, 600.0)
    for obj in (cylinder, square_glass, circle_glass):
        obj["MGF_full_trajectory_scaled"] = True
    cylinder["MGF_source_arena"] = "300 mm square Keypoint-MoSeq open field"
    cylinder["MGF_pilot_outer_diameter_mm"] = CYLINDER_DIAMETER
    bpy.context.view_layer.update()
    return {
        "cylinder_dimensions": list(cylinder.dimensions),
        "square_glass_dimensions": list(square_glass.dimensions),
        "circle_glass_dimensions": list(circle_glass.dimensions),
    }


def set_object_tree_visibility(name: str, visible: bool) -> list[str]:
    root = bpy.data.objects.get(name)
    if root is None:
        return []
    objects = [root, *root.children_recursive]
    for obj in objects:
        obj.hide_viewport = not visible
        obj.hide_render = not visible
        obj.hide_set(not visible)
    return [obj.name for obj in objects]


def configure_cables() -> dict[str, object]:
    # The bottom USB-C and Arduino leads are continuous. The remaining power and
    # Ethernet assemblies retain endpoints from the old, smaller camera layout.
    keep_roots = [
        "UsbC.001",
        "Curve_PowerCable.001",
        "Part_PowerCable.001",
        "ArduinoUsb.001",
        "ArduinoUsb.002",
        "Curve_ArduinoUSB",
    ]
    hide_roots = [
        "UsbC.002", "UsbC.003", "UsbC.004", "UsbC.005", "UsbC.006", "UsbC.007",
        "Curve_PowerCable.002", "Part_PowerCable.002",
        "Curve_PowerCable.003", "Part_PowerCable.003",
        "StandardUsb.002",
        "EthernetHead1.001", "EthernetHead1.002",
        "EthernetHead1.003", "EthernetHead1.004",
        "NurbsPath.001", "NurbsPath.002", "BezierCircle.001", "BezierCircle.002",
        "EthernetHead7.001", "EthernetHead7.002",
    ]
    kept = []
    hidden = []
    for name in keep_roots:
        kept.extend(set_object_tree_visibility(name, True))
    for name in hide_roots:
        hidden.extend(set_object_tree_visibility(name, False))
    return {
        "policy": "keep only cable assemblies with continuous, physically connected endpoints",
        "kept_roots": keep_roots,
        "hidden_roots": hide_roots,
        "kept_object_count": len(set(kept)),
        "hidden_object_count": len(set(hidden)),
        "mipi_cables_unchanged": True,
    }


def replace_collection(name: str) -> bpy.types.Collection:
    old = bpy.data.collections.get(name)
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def support_material() -> bpy.types.Material | None:
    old = bpy.data.objects.get("Part_CNC-10mmGlassSupport")
    if old and old.data.materials:
        return old.data.materials[0]
    return None


def create_annulus(
    collection: bpy.types.Collection,
    name: str,
    inner_radius: float,
    outer_radius: float,
    material: bpy.types.Material | None,
    segments: int = 128,
) -> bpy.types.Object:
    half_z = SUPPORT_THICKNESS * 0.5
    vertices = []
    for z in (-half_z, half_z):
        for radius in (inner_radius, outer_radius):
            for index in range(segments):
                angle = 2.0 * math.pi * index / segments
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    faces = []
    bottom_inner = 0
    bottom_outer = segments
    top_inner = segments * 2
    top_outer = segments * 3
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(
            [
                (bottom_inner + index, bottom_inner + nxt, bottom_outer + nxt, bottom_outer + index),
                (top_inner + index, top_outer + index, top_outer + nxt, top_inner + nxt),
                (bottom_inner + index, top_inner + index, top_inner + nxt, bottom_inner + nxt),
                (bottom_outer + index, bottom_outer + nxt, top_outer + nxt, top_outer + index),
            ]
        )
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location.z = SUPPORT_Z_CENTER
    if material is not None:
        obj.data.materials.append(material)
    return obj


def create_support_box(
    collection: bpy.types.Collection,
    name: str,
    center: Vector,
    dimensions: Vector,
    angle: float,
    material: bpy.types.Material | None,
) -> bpy.types.Object:
    x, y, z = dimensions.x * 0.5, dimensions.y * 0.5, dimensions.z * 0.5
    vertices = [
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7),
    ]
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.matrix_world = Matrix.Translation(center) @ Matrix.Rotation(angle, 4, "Z")
    if material is not None:
        obj.data.materials.append(material)
    return obj


def bbox_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def configure_support_structure() -> dict[str, object]:
    for name in ["Part_CNC-10mmGlassSupport"] + [
        f"Part_CNC-CameraSupport.{index:03d}" for index in range(1, 5)
    ]:
        old = bpy.data.objects.get(name)
        if old is not None:
            old.hide_render = True
            old.hide_viewport = True

    collection = replace_collection(SUPPORT_COLLECTION)
    material = support_material()
    bottom_posts = [bpy.data.objects[f"Part_12inchOpticPost.{index:03d}"] for index in range(1, 5)]
    side_posts = [bpy.data.objects[f"Part_8inchOpticPost.{index:03d}"] for index in range(1, 7)]
    bottom_centers = [object_bbox_center(obj) for obj in bottom_posts]
    side_centers = [object_bbox_center(obj) for obj in side_posts]
    bottom_radius = sum(math.hypot(p.x, p.y) for p in bottom_centers) / len(bottom_centers)
    side_radii = [math.hypot(p.x, p.y) for p in side_centers]
    rings = [(CYLINDER_DIAMETER * 0.5 + 0.5, bottom_radius + 20.0), (285.0, 315.0)]
    rings.append((min(side_radii) - 20.0, max(side_radii) + 20.0))
    ring_objects = []
    for index, (inner, outer) in enumerate(rings, start=1):
        ring_objects.append(
            create_annulus(collection, f"MGF_SupportRing_{index:02d}", inner, outer, material)
        )

    spoke_start = rings[0][1]
    spoke_objects = []
    pad_objects = []
    ring_top = SUPPORT_Z_CENTER + SUPPORT_THICKNESS * 0.5
    for index, (post, center, radius) in enumerate(zip(side_posts, side_centers, side_radii, strict=True), start=1):
        direction = Vector((center.x, center.y, 0.0)).normalized()
        spoke_end = radius + 20.0
        spoke_center = direction * ((spoke_start + spoke_end) * 0.5)
        spoke_center.z = SUPPORT_Z_CENTER
        spoke_objects.append(
            create_support_box(
                collection,
                f"MGF_RadialSpoke_{index:02d}",
                spoke_center,
                Vector((spoke_end - spoke_start, 26.0, SUPPORT_THICKNESS)),
                math.atan2(direction.y, direction.x),
                material,
            )
        )
        post_min, _ = bbox_bounds(post)
        if post_min.z > ring_top:
            pad_height = post_min.z - ring_top + 2.0
            pad_objects.append(
                create_support_box(
                    collection,
                    f"MGF_PostPad_{index:02d}",
                    Vector((center.x, center.y, ring_top + pad_height * 0.5)),
                    Vector((24.0, 24.0, pad_height)),
                    0.0,
                    material,
                )
            )

    return {
        "design": "nested rings with six radial spokes",
        "old_supports_hidden": True,
        "central_clearance_diameter": CYLINDER_DIAMETER + 1.0,
        "bottom_post_radius": bottom_radius,
        "side_post_radii": side_radii,
        "ring_ranges": [list(pair) for pair in rings],
        "rings": [obj.name for obj in ring_objects],
        "spokes": [obj.name for obj in spoke_objects],
        "post_pads": [obj.name for obj in pad_objects],
    }


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def collection_roots(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    members = set(collection.all_objects)
    return [obj for obj in members if obj.parent not in members]


def object_bbox_center(obj: bpy.types.Object) -> Vector:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(corners, Vector()) / len(corners)


def align_stage_lights_to_posts() -> dict[str, object]:
    pairs = {
        "LED.001": "Part_8inchOpticPost.001",
        "LED.002": "Part_8inchOpticPost.003",
        "LED.003": "Part_8inchOpticPost.002",
    }
    report = {}
    for collection_name, post_name in pairs.items():
        collection = bpy.data.collections[collection_name]
        collection.hide_render = False
        collection.hide_viewport = False
        stage_objects = list(collection.all_objects)
        for obj in stage_objects:
            obj.hide_render = False
            obj.hide_viewport = False
        stage_center = sum(
            (obj.matrix_world.translation for obj in stage_objects),
            Vector(),
        ) / len(stage_objects)
        post_center = object_bbox_center(bpy.data.objects[post_name])
        delta = Vector((post_center.x - stage_center.x, post_center.y - stage_center.y, 0.0))
        for obj in collection_roots(collection):
            obj.matrix_world = Matrix.Translation(delta) @ obj.matrix_world
        report[collection_name] = {
            "post": post_name,
            "translation": list(delta),
            "post_xy": [post_center.x, post_center.y],
        }
    return report


def camera_projection_bounds(camera: bpy.types.Object, obj: bpy.types.Object) -> list[float]:
    coordinates = [
        world_to_camera_view(bpy.context.scene, camera, point)
        for point in mesh_world_points(obj)
    ]
    return [
        min(point.x for point in coordinates),
        max(point.x for point in coordinates),
        min(point.y for point in coordinates),
        max(point.y for point in coordinates),
    ]


def fit_camera_to_cylinder(camera: bpy.types.Object) -> dict[str, object]:
    cylinder = bpy.data.objects["Part_GlassCylinder160mm"]
    low, high = 1.0, 100.0
    for _ in range(32):
        middle = (low + high) * 0.5
        camera.data.lens = middle
        bounds = camera_projection_bounds(camera, cylinder)
        fits = (
            bounds[0] >= CAMERA_FRAME_MARGIN
            and bounds[1] <= 1.0 - CAMERA_FRAME_MARGIN
            and bounds[2] >= CAMERA_FRAME_MARGIN
            and bounds[3] <= 1.0 - CAMERA_FRAME_MARGIN
        )
        if fits:
            low = middle
        else:
            high = middle
    camera.data.lens = low
    return {
        "lens": camera.data.lens,
        "projected_bounds": camera_projection_bounds(camera, cylinder),
        "frame_margin": CAMERA_FRAME_MARGIN,
    }


def fit_side_camera_distance(
    camera: bpy.types.Object,
    direction: Vector,
    height: float,
    target: Vector,
) -> dict[str, object]:
    camera.data.lens = HORIZONTAL_CAMERA_LENS
    low, high = CYLINDER_DIAMETER * 0.5 + 25.0, 1500.0
    for _ in range(32):
        middle = (low + high) * 0.5
        camera.location = direction * middle
        camera.location.z = height
        look_at(camera, target)
        bpy.context.view_layer.update()
        bounds = camera_projection_bounds(camera, bpy.data.objects["Part_GlassCylinder160mm"])
        fits = (
            bounds[0] >= CAMERA_FRAME_MARGIN
            and bounds[1] <= 1.0 - CAMERA_FRAME_MARGIN
            and bounds[2] >= CAMERA_FRAME_MARGIN
            and bounds[3] <= 1.0 - CAMERA_FRAME_MARGIN
        )
        if fits:
            high = middle
        else:
            low = middle
    camera.location = direction * high
    camera.location.z = height
    look_at(camera, target)
    bpy.context.view_layer.update()
    return {
        "lens": camera.data.lens,
        "radius": high,
        "projected_bounds": camera_projection_bounds(
            camera,
            bpy.data.objects["Part_GlassCylinder160mm"],
        ),
        "frame_margin": CAMERA_FRAME_MARGIN,
    }


def configure_cameras() -> dict[str, object]:
    cylinder = bpy.data.objects["Part_GlassCylinder160mm"]
    target = object_bbox_center(cylinder)
    report = {}
    for index in range(1, 7):
        camera = bpy.data.objects[f"Camera.{index:03d}"]
        old_location = camera.location.copy()
        direction = Vector((old_location.x, old_location.y, 0.0))
        if direction.length < 1e-6:
            angle = math.radians((index - 1) * 60.0)
            direction = Vector((math.cos(angle), math.sin(angle), 0.0))
        direction.normalize()
        camera.data.sensor_width = 25.4
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.clip_start = 1.0
        camera.data.clip_end = 3000.0
        fit_report = fit_side_camera_distance(
            camera,
            direction,
            FLOOR_Z + (43.0 if index <= 3 else 92.0),
            target,
        )

        collection = bpy.data.collections.get(RIG_COLLECTION_NAMES[index])
        if collection is not None:
            delta = camera.location - old_location
            for root in collection_roots(collection):
                root.matrix_world = Matrix.Translation(delta) @ root.matrix_world
            collection["MGF_full_trajectory_translation"] = list(delta)
        report[camera.name] = {"location": list(camera.location), **fit_report}

    stage_light_report = align_stage_lights_to_posts()

    bottom = bpy.data.objects["Camera.007"]
    bottom.location = Vector((0.0, 0.0, 40.0))
    bottom.data.sensor_width = 25.4
    bottom.data.sensor_fit = "HORIZONTAL"
    bottom.data.clip_start = 1.0
    bottom.data.clip_end = 1500.0
    look_at(bottom, target)
    report[bottom.name] = {"location": list(bottom.location)}

    top = bpy.data.objects["Camera.008"]
    top.location = Vector((0.0, 0.0, 850.0))
    top.data.sensor_width = 25.4
    top.data.sensor_fit = "HORIZONTAL"
    top.data.clip_start = 1.0
    top.data.clip_end = 2000.0
    look_at(top, target)
    report[top.name] = {"location": list(top.location)}

    bpy.context.view_layer.update()
    for camera_name in ("Camera.007", "Camera.008"):
        report[camera_name].update(fit_camera_to_cylinder(bpy.data.objects[camera_name]))
    report["stage_lights"] = stage_light_report

    ceiling = bpy.data.objects.get("Part_TopCover")
    if ceiling is not None:
        ceiling.location.z = 1200.0
    return report


def configure_lighting() -> dict[str, float]:
    top = bpy.data.objects.get("日光")
    if top is not None and top.type == "LIGHT":
        top.data.energy = 2.5
        top.hide_render = False
    bottom = bpy.data.objects.get("MGF_BottomFill")
    if bottom is not None and bottom.type == "LIGHT":
        bottom.data.energy = 4.0
        bottom.rotation_euler = (math.pi, 0.0, 0.0)
        bottom.hide_render = False

    side = bpy.data.objects.get("MGF_FullTrajectory_SideFill")
    if side is None:
        light_data = bpy.data.lights.new("MGF_FullTrajectory_SideFill", type="SUN")
        side = bpy.data.objects.new("MGF_FullTrajectory_SideFill", light_data)
        (bpy.data.collections.get("Collection") or bpy.context.scene.collection).objects.link(side)
    side.data.energy = 1.6
    side.data.angle = 0.25
    side.rotation_euler = (math.radians(65.0), 0.0, math.radians(35.0))
    side.hide_render = False

    world_strength = 0.8
    world = bpy.context.scene.world
    if world and world.use_nodes and world.node_tree:
        background = next(
            (node for node in world.node_tree.nodes if node.type == "BACKGROUND"),
            None,
        )
        if background is not None:
            background.inputs[0].default_value = (0.10, 0.11, 0.12, 1.0)
            background.inputs[1].default_value = world_strength
    return {
        "top_sun": float(top.data.energy) if top and top.type == "LIGHT" else 0.0,
        "bottom_sun": float(bottom.data.energy) if bottom and bottom.type == "LIGHT" else 0.0,
        "side_sun": float(side.data.energy),
        "world_strength": world_strength,
    }


def install_pilot_render_helper() -> None:
    text = bpy.data.texts.get("MGF_FullTrajectory_Render8Views.py")
    if text is None:
        text = bpy.data.texts.new("MGF_FullTrajectory_Render8Views.py")
    text.clear()
    text.write(
        '''from pathlib import Path
import bpy

scene = bpy.context.scene
root = Path(bpy.path.abspath("//full_trajectory_multiview")) / f"frame_{scene.frame_current:06d}"
root.mkdir(parents=True, exist_ok=True)
rig_names = {
    1: "MGF_Rig_Camera.001_Horizontal_Front",
    2: "MGF_Rig_Camera.002_Horizontal_BackLeft",
    3: "MGF_Rig_Camera.003_Horizontal_BackRight",
    4: "MGF_Rig_Camera.004_Horizontal_Back",
    5: "MGF_Rig_Camera.005_Horizontal_FrontRight",
    6: "MGF_Rig_Camera.006_Horizontal_FrontLeft",
    7: "MGF_Rig_Camera.007_Bottom",
}
for index in range(1, 9):
    camera = bpy.data.objects[f"Camera.{index:03d}"]
    collection = bpy.data.collections.get(rig_names.get(index, ""))
    old_collection = collection.hide_render if collection else None
    try:
        if collection:
            collection.hide_render = True
        scene.camera = camera
        scene.render.filepath = str(root / f"view_{index - 1:02d}_{camera.name}.png")
        bpy.ops.render.render(write_still=True)
    finally:
        if collection:
            collection.hide_render = old_collection
'''
    )


def animate_pilot(
    data: np.ndarray,
    source_start: int,
    frame_count: int,
    source_stride: int,
) -> dict[str, object]:
    armature = bpy.data.objects[ARMATURE_NAME]
    controls = {
        "root": bpy.data.objects["Root_M.001"],
        "spine2": bpy.data.objects["Spine2_M.001"],
        "wrist_l": bpy.data.objects["Wrist_L.001"],
        "wrist_r": bpy.data.objects["Wrist_R.001"],
        "ankle_l": bpy.data.objects["Ankle_L.001"],
        "ankle_r": bpy.data.objects["Ankle_R.001"],
        "knee_pole_l": bpy.data.objects["MGF_FT_KneePole_L"],
        "knee_pole_r": bpy.data.objects["MGF_FT_KneePole_R"],
    }
    for control in controls.values():
        control.animation_data_clear()

    spine_curve = bpy.data.objects[SPINE_CURVE_NAME]
    tail_curve = bpy.data.objects[TAIL_CURVE_NAME]
    spine_curve.data.animation_data_clear()
    tail_curve.data.animation_data_clear()
    spine_controls = make_curve_visual_controls(SPINE_CURVE_NAME, "Spine")
    tail_controls = make_curve_visual_controls(TAIL_CURVE_NAME, "Tail")

    smoothed = smooth_time(data)
    robust_center = np.median(smoothed, axis=1)
    arena_center = np.median(robust_center[:, :2], axis=0)

    chain_distances = []
    for first, second in zip(SPINE_SOURCE_INDICES[:-1], SPINE_SOURCE_INDICES[1:]):
        distance = np.linalg.norm(smoothed[:, second] - smoothed[:, first], axis=1)
        chain_distances.append(float(np.median(distance)))

    current_curve_points = curve_world_points(spine_curve)
    current_curve_lengths = [
        (current_curve_points[index + 1] - current_curve_points[index]).length
        for index in range(len(current_curve_points) - 1)
    ]
    current_curve_arc = sum(current_curve_lengths)
    source_chain_arc = sum(chain_distances)
    fixed_source_lengths = [length * current_curve_arc / source_chain_arc for length in chain_distances]
    curve_cumulative = [0.0]
    for length in current_curve_lengths:
        curve_cumulative.append(curve_cumulative[-1] + length)
    curve_fractions = [value / current_curve_arc for value in curve_cumulative]

    tail_distance = np.linalg.norm(smoothed[:, 4] - smoothed[:, 3], axis=1)
    target_tail_length = float(np.median(tail_distance))
    audit = {
        "floor_clamps": 0,
        "wall_clamps": 0,
        "body_proxy_pushes": 0,
        "tail_length_clamps": 0,
        "hind_lateral_clamps": 0,
        "hind_height_clamps": 0,
        "hind_reach_clamps": 0,
    }
    previous_tail_mid = None

    for scene_frame in range(1, frame_count + 1):
        source_frame = source_start + (scene_frame - 1) * source_stride
        points = smoothed[source_frame]
        spine = normalized_spine_controls(
            points,
            arena_center,
            fixed_source_lengths,
            curve_fractions,
        )
        spine = [limit_xy(point, CYLINDER_INNER_RADIUS - 9.0, audit) for point in spine]
        for index, (control, point) in enumerate(zip(spine_controls, spine, strict=True)):
            set_curve_point_world(spine_curve, index, point, scene_frame)
            set_empty_world_location(control, point, scene_frame)
        set_world_location(controls["root"], spine[0], scene_frame)
        set_world_location(controls["spine2"], spine[1], scene_frame)

        body_forward = spine[-1] - spine[0]
        body_forward.z = 0.0
        if body_forward.length < 1e-6:
            body_forward = Vector((1.0, 0.0, 0.0))
        else:
            body_forward.normalize()
        body_lateral = Vector((-body_forward.y, body_forward.x, 0.0))
        set_empty_world_location(
            controls["knee_pole_l"],
            spine[0] + body_forward * 22.0 + body_lateral * 14.0 + Vector((0.0, 0.0, -1.0)),
            scene_frame,
        )
        set_empty_world_location(
            controls["knee_pole_r"],
            spine[0] + body_forward * 22.0 - body_lateral * 14.0 + Vector((0.0, 0.0, -1.0)),
            scene_frame,
        )

        wrist_l = map_source(points[13], arena_center)
        wrist_r = map_source(points[14], arena_center)
        ankle_l = map_source(points[9], arena_center)
        ankle_r = map_source(points[11], arena_center)

        wrist_l = floor_clamp(wrist_l, FLOOR_Z + 4.5, audit)
        wrist_r = floor_clamp(wrist_r, FLOOR_Z + 4.5, audit)
        ankle_l = floor_clamp(ankle_l, FLOOR_Z + 3.0, audit)
        ankle_r = floor_clamp(ankle_r, FLOOR_Z + 3.0, audit)
        ankle_l = constrain_hind_target(
            ankle_l,
            spine[0],
            body_forward,
            body_lateral,
            1.0,
            audit,
        )
        ankle_r = constrain_hind_target(
            ankle_r,
            spine[0],
            body_forward,
            body_lateral,
            -1.0,
            audit,
        )
        wrist_l = push_wrist_from_body(wrist_l, spine, audit)
        wrist_r = push_wrist_from_body(wrist_r, spine, audit)
        wrist_l = limit_xy(wrist_l, CYLINDER_INNER_RADIUS - 3.0, audit)
        wrist_r = limit_xy(wrist_r, CYLINDER_INNER_RADIUS - 3.0, audit)
        ankle_l = limit_xy(ankle_l, CYLINDER_INNER_RADIUS - 3.0, audit)
        ankle_r = limit_xy(ankle_r, CYLINDER_INNER_RADIUS - 3.0, audit)

        set_world_location(controls["wrist_l"], wrist_l, scene_frame)
        set_world_location(controls["wrist_r"], wrist_r, scene_frame)
        set_world_location(controls["ankle_l"], ankle_l, scene_frame)
        set_world_location(controls["ankle_r"], ankle_r, scene_frame)

        tail_tip = map_source(points[4], arena_center)
        tail, previous_tail_mid = build_tail_controls(
            spine[0],
            tail_tip,
            body_forward,
            target_tail_length,
            previous_tail_mid,
            audit,
        )
        for index, (control, point) in enumerate(zip(tail_controls, tail, strict=True)):
            set_curve_point_world(tail_curve, index, point, scene_frame)
            set_empty_world_location(control, point, scene_frame)

    bpy.context.view_layer.update()
    return {
        "arena_center_xy": arena_center.tolist(),
        "source_chain_median_lengths": chain_distances,
        "normalized_spine_arc": current_curve_arc,
        "target_tail_length": target_tail_length,
        "collision_audit": audit,
        "control_objects": {key: value.name for key, value in controls.items()},
        "spine_curve_visual_controls": [obj.name for obj in spine_controls],
        "tail_curve_visual_controls": [obj.name for obj in tail_controls],
    }


def evaluated_mesh_points(mesh: bpy.types.Object) -> list[Vector]:
    evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    try:
        return [evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def shift_curve_at_frame(curve: bpy.types.Object, delta_z: float, frame: int) -> None:
    for index, point in enumerate(curve.data.splines[0].points):
        world = curve.matrix_world @ point.co.to_3d()
        world.z += delta_z
        set_curve_point_world(curve, index, world, frame)


def refine_floor_penetration(frame_count: int) -> dict[str, object]:
    """Use the nearest active IK/root controller to resolve mesh-floor penetration."""
    mesh = bpy.data.objects[MOUSE_MESH_NAME]
    spine_curve = bpy.data.objects[SPINE_CURVE_NAME]
    tail_curve = bpy.data.objects[TAIL_CURVE_NAME]
    root = bpy.data.objects["Root_M.001"]
    spine2 = bpy.data.objects["Spine2_M.001"]
    movable = {
        "wrist_l": bpy.data.objects["Wrist_L.001"],
        "wrist_r": bpy.data.objects["Wrist_R.001"],
        "ankle_l": bpy.data.objects["Ankle_L.001"],
        "ankle_r": bpy.data.objects["Ankle_R.001"],
        "root": root,
    }
    corrections = {name: 0 for name in movable}
    corrected_frames = set()
    maximum_depth = 0.0

    for frame in range(1, frame_count + 1):
        bpy.context.scene.frame_set(frame)
        for _ in range(3):
            bpy.context.view_layer.update()
            points = evaluated_mesh_points(mesh)
            minimum_z = min(point.z for point in points)
            depth = FLOOR_Z + 0.15 - minimum_z
            maximum_depth = max(maximum_depth, depth)
            if depth <= 0.0:
                break
            lowest = sorted(points, key=lambda point: point.z)[:40]
            centroid = sum(lowest, Vector()) / len(lowest)
            nearest_name = min(
                movable,
                key=lambda name: (centroid - movable[name].matrix_world.translation).length,
            )
            delta_z = min(depth + 0.1, 5.0)
            if nearest_name == "root":
                shift_curve_at_frame(spine_curve, delta_z, frame)
                shift_curve_at_frame(tail_curve, delta_z, frame)
                set_world_location(root, root.matrix_world.translation + Vector((0, 0, delta_z)), frame)
                set_world_location(spine2, spine2.matrix_world.translation + Vector((0, 0, delta_z)), frame)
            else:
                control = movable[nearest_name]
                set_world_location(
                    control,
                    control.matrix_world.translation + Vector((0, 0, delta_z)),
                    frame,
                )
            corrections[nearest_name] += 1
            corrected_frames.add(frame)

    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    residual = {}
    for frame in (1, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count):
        bpy.context.scene.frame_set(max(1, frame))
        bpy.context.view_layer.update()
        residual[str(max(1, frame))] = min(point.z for point in evaluated_mesh_points(mesh)) - FLOOR_Z
    return {
        "corrected_frame_count": len(corrected_frames),
        "controller_corrections": corrections,
        "maximum_detected_penetration": maximum_depth,
        "sampled_residual_clearance": residual,
    }


def configure_scene(scene: bpy.types.Scene, frame_count: int, source_stride: int) -> None:
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.frame_set(1)
    scene.render.fps = int(SOURCE_FPS)
    scene.render.fps_base = float(source_stride)
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene["MGF_animation_mode"] = "full_trajectory_pilot"
    scene["MGF_source_fps"] = SOURCE_FPS
    scene["MGF_source_stride"] = source_stride
    scene["MGF_sampled_fps"] = SOURCE_FPS / source_stride
    scene["MGF_source_arena_mm"] = [300.0, 300.0]
    scene["MGF_bottom_view_clean_support"] = False


def main() -> None:
    args = parse_args()
    if not args.keypoints.exists():
        raise FileNotFoundError(args.keypoints)
    if args.source_stride < 1:
        raise ValueError("--source-stride must be at least 1")

    source_blend = Path(bpy.data.filepath)
    output_path = args.output or source_blend.parent / DEFAULT_OUTPUT_NAME
    data = np.load(args.keypoints).astype(np.float64)
    if data.ndim != 3 or data.shape[1:] != (15, 3):
        raise ValueError(f"Expected (frames, 15, 3), got {data.shape}")
    last_source_frame = args.source_start + (args.frames - 1) * args.source_stride
    if args.source_start < 0 or args.frames < 1 or last_source_frame >= len(data):
        raise ValueError(
            f"Requested sampled range [{args.source_start}, {last_source_frame}] "
            f"is outside the {len(data)}-frame sample"
        )

    remove_previous_controls()
    armature = bpy.data.objects[ARMATURE_NAME]
    rig_report = configure_tail_and_ik(armature)
    configure_scene(bpy.context.scene, args.frames, args.source_stride)
    environment_report = configure_environment()
    camera_report = configure_cameras()
    cable_report = configure_cables()
    environment_report["support_structure"] = configure_support_structure()
    lighting_report = configure_lighting()
    animation_report = animate_pilot(
        data,
        args.source_start,
        args.frames,
        args.source_stride,
    )
    animation_report["floor_refinement"] = refine_floor_penetration(args.frames)
    install_pilot_render_helper()

    scene = bpy.context.scene
    scene["MGF_source_keypoints"] = str(args.keypoints)
    scene["MGF_source_frame_start"] = args.source_start
    scene["MGF_source_frame_end"] = last_source_frame
    scene["MGF_tail_target_length"] = animation_report["target_tail_length"]
    scene.frame_set(1)
    bpy.context.view_layer.update()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), check_existing=False)
    manifest = {
        "format": "mousegeoforge_keypoint_moseq_full_trajectory_pilot_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(source_blend),
        "source_keypoints": str(args.keypoints),
        "source_frame_range": [args.source_start, last_source_frame],
        "source_window": [args.source_start, args.source_start + args.frames * args.source_stride - 1],
        "source_stride": args.source_stride,
        "scene_frame_range": [1, args.frames],
        "source_fps": SOURCE_FPS,
        "fps": SOURCE_FPS / args.source_stride,
        "output_blend": str(output_path),
        "rig": rig_report,
        "environment": environment_report,
        "cameras": camera_report,
        "cables": cable_report,
        "lighting": lighting_report,
        "animation": animation_report,
        "notes": [
            "Paper specifies a 300 mm square transparent arena and six Azure Kinect views, but not per-camera calibration matrices.",
            "This pilot retains eight project cameras and moves the six side rigs outward.",
            "The six horizontal cameras use 30 mm lenses and move radially until the cylinder retains a three-percent frame margin.",
            "Legacy ring/hash supports are replaced by nested rings and six spokes fitted to six side posts and four bottom posts.",
            "Three stage-light assemblies are aligned to the matching front, back-left, and back-right optic posts.",
            "Only the continuous bottom USB-C and Arduino USB cable assemblies remain visible; disconnected power and Ethernet assemblies are hidden.",
            "Hind-paw targets are clamped in the body frame and knee pole controls preserve a forward/outward bend.",
            "Spine arc length is normalized; wrist and ankle IK have stretching disabled.",
            "Tail Spline IK fits a median 3D tail length instead of scaling the whole mouse.",
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


REFERENCE_CAMERA_POSITIONS = {
    "CameraA": (-29.8610000610, 124.4000015260, 70.3685302734),
    "CameraB": (68.4729995728, 194.4100036621, 63.7130012512),
    "CameraC": (169.1732635498, 125.0, 70.5),
    "CameraD": (126.8616256714, 151.1414337158, 66.0),
    "CameraE": (16.1541938782, 149.1687164307, 66.0),
}

WAYPOINTS = (
    (0.00, 0.0, 0.0, "CENTER_START"),
    (0.20, -27.0, 3.0, "LEFT"),
    (0.40, -7.0, 15.0, "UP"),
    (0.60, 31.0, 4.0, "RIGHT"),
    (0.80, 8.0, -9.0, "DOWN"),
    (1.00, 0.0, 0.0, "CENTER_END"),
)


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate the V19 CamB physical rig over a constrained hemisphere patch."
    )
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--start-frame", type=int, default=63)
    parser.add_argument("--end-frame", type=int, default=163)
    parser.add_argument("--camera", default="Camera.001")
    parser.add_argument("--camera-model-root", default="CameraItems.001")
    parser.add_argument("--camera-lens-model", default="Part_CamLen.001")
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--face-bone", default="Head1_M")
    parser.add_argument("--face-head-tail", type=float, default=0.55)
    parser.add_argument("--model-clearance", type=float, default=44.0)
    return parser.parse_args(args_after_dash())


def ensure_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def link_object(collection, obj):
    if obj.name not in collection.objects:
        collection.objects.link(obj)


def world_bone_point(armature, bone_name, head_tail):
    bone = armature.pose.bones[bone_name]
    local = bone.head.lerp(bone.tail, max(0.0, min(1.0, head_tail)))
    return armature.matrix_world @ local


def camera_forward(camera):
    return (camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()


def look_at(obj, target):
    direction = target - obj.location
    if direction.length < 1.0e-8:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def parent_preserve_world(obj, parent):
    world = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted_safe()
    obj.matrix_world = world


def descendants(obj):
    result = []
    pending = list(obj.children)
    while pending:
        child = pending.pop()
        result.append(child)
        pending.extend(child.children)
    return result


def assembly_world_matrix(obj, cache=None):
    """Rebuild transforms without relying on stale matrices of hidden objects."""
    if cache is None:
        cache = {}
    if obj.name in cache:
        return cache[obj.name].copy()
    if obj.parent is None:
        matrix = obj.matrix_basis.copy()
    else:
        matrix = (
            assembly_world_matrix(obj.parent, cache)
            @ obj.matrix_parent_inverse
            @ obj.matrix_basis
        )
    cache[obj.name] = matrix.copy()
    return matrix


def prototype_forward_extent(objects, matrices):
    forward_extent = 0.0
    for obj in objects:
        if obj.type != "MESH":
            continue
        matrix = matrices[obj.name]
        for corner in obj.bound_box:
            point = matrix @ Vector(corner)
            forward_extent = max(forward_extent, -point.z)
    return forward_extent


def combine_prototype_meshes(objects, matrices, collection):
    vertices = []
    faces = []
    face_materials = []
    materials = []
    material_indices = {}
    source_names = []

    for obj in objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        source_names.append(obj.name)
        matrix = matrices[obj.name]
        vertex_offset = len(vertices)
        vertices.extend(matrix @ vertex.co for vertex in obj.data.vertices)

        slot_map = []
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                slot_map.append(0)
                continue
            if material.name not in material_indices:
                material_indices[material.name] = len(materials)
                materials.append(material)
            slot_map.append(material_indices[material.name])

        flip_faces = matrix.to_3x3().determinant() < 0.0
        for polygon in obj.data.polygons:
            indices = [vertex_offset + index for index in polygon.vertices]
            if flip_faces:
                indices.reverse()
            faces.append(indices)
            if slot_map:
                source_index = min(polygon.material_index, len(slot_map) - 1)
                face_materials.append(slot_map[source_index])
            else:
                face_materials.append(0)

    if not vertices or not faces:
        raise RuntimeError("CamB prototype meshes are empty")

    mesh = bpy.data.meshes.new("MGF_CamB_PhysicalModel_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    for material in materials:
        mesh.materials.append(material)
    if materials:
        for polygon, material_index in zip(mesh.polygons, face_materials):
            polygon.material_index = material_index

    combined = bpy.data.objects.new("MGF_CamB_PhysicalModel", mesh)
    link_object(collection, combined)
    combined.color = (0.16, 0.19, 0.22, 1.0)
    combined["source_parts"] = json.dumps(source_names)
    combined["rigid_combined_mesh"] = True
    return combined, source_names


def spherical_position(focus, radius, radial, horizontal, vertical, azimuth_deg, elevation_deg):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    direction = (
        math.cos(elevation) * math.cos(azimuth) * radial
        + math.cos(elevation) * math.sin(azimuth) * horizontal
        + math.sin(elevation) * vertical
    ).normalized()
    return focus + radius * direction


def add_viewport_path(collection, name, points):
    curve_data = bpy.data.curves.new(f"{name}_Data", "CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 0.35
    curve_data.bevel_resolution = 3
    spline = curve_data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, position in zip(spline.points, points):
        point.co = (*position, 1.0)
    path = bpy.data.objects.new(name, curve_data)
    link_object(collection, path)
    path.color = (0.08, 0.8, 1.0, 1.0)
    path.hide_render = True
    path.show_in_front = True
    return path


def add_seed_markers(collection):
    markers = []
    for name, coordinates in REFERENCE_CAMERA_POSITIONS.items():
        marker = bpy.data.objects.new(f"MGF_RigSeed_{name}", None)
        link_object(collection, marker)
        marker.location = coordinates
        marker.empty_display_type = "SPHERE"
        marker.empty_display_size = 2.5
        marker.color = (1.0, 0.45, 0.05, 1.0)
        marker.hide_render = True
        marker.show_in_front = True
        marker["source"] = "PyMouseStudio-BrowmFur-WaterReaching-FrameToFrame-FGcamB.blend"
        markers.append(marker)
    return markers


def set_linear_interpolation(obj):
    if obj.animation_data is None or obj.animation_data.action is None:
        return
    action = obj.animation_data.action
    if hasattr(action, "fcurves"):
        fcurves = action.fcurves
    else:
        fcurves = [
            fcurve
            for layer in action.layers
            for strip in layer.strips
            for channelbag in strip.channelbags
            for fcurve in channelbag.fcurves
        ]
    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def eased_angles(frame, start_frame, end_frame):
    fraction = (frame - start_frame) / max(end_frame - start_frame, 1)
    for index in range(len(WAYPOINTS) - 1):
        left = WAYPOINTS[index]
        right = WAYPOINTS[index + 1]
        if fraction <= right[0] or index == len(WAYPOINTS) - 2:
            local = (fraction - left[0]) / max(right[0] - left[0], 1.0e-8)
            local = max(0.0, min(1.0, local))
            weight = 0.5 - 0.5 * math.cos(math.pi * local)
            azimuth = left[1] + weight * (right[1] - left[1])
            elevation = left[2] + weight * (right[2] - left[2])
            return azimuth, elevation
    return WAYPOINTS[-1][1], WAYPOINTS[-1][2]


def add_timeline_markers(scene, keyframes):
    for marker in list(scene.timeline_markers):
        if marker.name.startswith("MGF_CAMB_"):
            scene.timeline_markers.remove(marker)
    for row in keyframes:
        scene.timeline_markers.new(f"MGF_CAMB_{row['label']}", frame=row["frame"])


def frame_audit(scene, camera, root, target, frames, initial_radius, tracked_objects):
    rows = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        camera_position = camera.matrix_world.translation.copy()
        target_position = target.matrix_world.translation.copy()
        radius = (camera_position - target_position).length
        ndc = world_to_camera_view(scene, camera, target_position)
        alignment = camera_forward(camera).dot((target_position - camera_position).normalized())
        rows.append(
            {
                "frame": frame,
                "camera_location": list(camera_position),
                "target_location": list(target_position),
                "radius": float(radius),
                "radius_error": float(abs(radius - initial_radius)),
                "target_ndc": [float(ndc.x), float(ndc.y), float(ndc.z)],
                "forward_alignment": float(alignment),
                "root_location": list(root.matrix_world.translation),
                "tracked_world_locations": {
                    obj.name: list(obj.matrix_world.translation) for obj in tracked_objects
                },
                "tracked_root_local_matrices": {
                    obj.name: [list(matrix_row) for matrix_row in (root.matrix_world.inverted_safe() @ obj.matrix_world)]
                    for obj in tracked_objects
                },
            }
        )
    return rows


def main():
    args = parse_args()
    if args.end_frame <= args.start_frame:
        raise ValueError("end-frame must be greater than start-frame")

    source_blend = bpy.data.filepath
    scene = bpy.context.scene
    scene.frame_set(args.start_frame)
    camera = bpy.data.objects.get(args.camera)
    model_root = bpy.data.objects.get(args.camera_model_root)
    lens_model = bpy.data.objects.get(args.camera_lens_model)
    armature = bpy.data.objects.get(args.armature)
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Camera {args.camera!r} was not found")
    if model_root is None or lens_model is None:
        raise RuntimeError("CamB physical model roots were not found")
    if armature is None or args.face_bone not in armature.pose.bones:
        raise RuntimeError("Face target armature/bone was not found")

    demo_collection = ensure_collection("MGF_CamB_HemisphereDemo")
    focus = world_bone_point(armature, args.face_bone, args.face_head_tail)
    initial_camera_matrix = camera.matrix_world.copy()
    physical_model_objects = [model_root] + descendants(model_root)
    if lens_model not in physical_model_objects:
        physical_model_objects.append(lens_model)
    assembly_cache = {}
    camera_inverse = initial_camera_matrix.inverted_safe()
    initial_model_matrices = {
        obj.name: camera_inverse @ assembly_world_matrix(obj, assembly_cache)
        for obj in physical_model_objects
    }
    model_forward_extent = prototype_forward_extent(
        physical_model_objects,
        initial_model_matrices,
    )
    initial_position = initial_camera_matrix.translation.copy()
    radial = (initial_position - focus).normalized()
    horizontal = Vector((1.0, 0.0, 0.0))
    horizontal = (horizontal - radial * horizontal.dot(radial)).normalized()
    vertical = horizontal.cross(radial).normalized()
    if vertical.z < 0.0:
        vertical.negate()
    original_radius = (initial_position - focus).length
    reference_b_radius = (Vector(REFERENCE_CAMERA_POSITIONS["CameraB"]) - focus).length
    initial_radius = max(
        original_radius,
        reference_b_radius,
        model_forward_extent + max(args.model_clearance, 0.0),
    )

    target = bpy.data.objects.new("MGF_CamB_FaceTarget", None)
    link_object(demo_collection, target)
    target.location = focus
    target.empty_display_type = "SPHERE"
    target.empty_display_size = 1.5
    target.color = (1.0, 0.1, 0.1, 1.0)
    target.hide_render = True
    target.show_in_front = True
    copy_location = target.constraints.new("COPY_LOCATION")
    copy_location.name = "MGF_FollowHeadEnd"
    copy_location.target = armature
    copy_location.subtarget = args.face_bone
    if hasattr(copy_location, "head_tail"):
        copy_location.head_tail = max(0.0, min(1.0, args.face_head_tail))
    copy_location.target_space = "WORLD"
    copy_location.owner_space = "WORLD"

    rig_root = bpy.data.objects.new("MGF_CamB_AnimatedRig", None)
    link_object(demo_collection, rig_root)
    rig_root.matrix_world = initial_camera_matrix
    rig_root.empty_display_type = "ARROWS"
    rig_root.empty_display_size = 5.0
    rig_root.show_in_front = True
    rig_root["description"] = "CamB rigid rig moving on a five-camera-bounded hemisphere patch"
    rig_root["frame_range"] = [args.start_frame, args.end_frame]
    rig_root["focus_bone"] = args.face_bone
    rig_root["focus_head_tail"] = args.face_head_tail
    rig_root["constant_radius"] = initial_radius
    rig_root["original_camera_radius"] = original_radius
    rig_root["model_forward_extent"] = model_forward_extent
    rig_root["model_clearance"] = args.model_clearance
    rig_root["reference_camera_positions"] = json.dumps(REFERENCE_CAMERA_POSITIONS)

    camera.parent = rig_root
    camera.matrix_parent_inverse = Matrix.Identity(4)
    camera.matrix_basis = Matrix.Identity(4)
    combined_model, combined_source_names = combine_prototype_meshes(
        physical_model_objects,
        initial_model_matrices,
        demo_collection,
    )
    combined_model.parent = rig_root
    combined_model.matrix_parent_inverse = Matrix.Identity(4)
    combined_model.matrix_basis = Matrix.Identity(4)
    combined_model.hide_viewport = False
    # The physical barrel surrounds the optical camera and must not occlude
    # synthetic renders made from CameraB itself.
    combined_model.hide_render = True
    try:
        combined_model.hide_set(False)
    except RuntimeError:
        pass
    for model in physical_model_objects:
        model.hide_viewport = True
        model.hide_render = True
        try:
            model.hide_set(True)
        except RuntimeError:
            pass
    fixed_middle_support = bpy.data.objects.get("cam_sup_middle")
    if fixed_middle_support is not None:
        fixed_middle_support.hide_viewport = True
        fixed_middle_support.hide_render = True

    track = rig_root.constraints.new("TRACK_TO")
    track.name = "MGF_TrackMouseFace"
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    track.use_target_z = False

    keyframes = []
    duration = args.end_frame - args.start_frame
    for fraction, azimuth, elevation, label in WAYPOINTS:
        frame = int(round(args.start_frame + fraction * duration))
        position = spherical_position(
            focus,
            initial_radius,
            radial,
            horizontal,
            vertical,
            azimuth,
            elevation,
        )
        keyframes.append(
            {
                "frame": frame,
                "label": label,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "location": list(position),
            }
        )
    for frame in range(args.start_frame, args.end_frame + 1):
        azimuth, elevation = eased_angles(frame, args.start_frame, args.end_frame)
        rig_root.location = spherical_position(
            focus,
            initial_radius,
            radial,
            horizontal,
            vertical,
            azimuth,
            elevation,
        )
        rig_root.keyframe_insert(data_path="location", frame=frame, group="MGF_CamB_Hemisphere")
    set_linear_interpolation(rig_root)

    sampled_path = []
    for frame in range(args.start_frame, args.end_frame + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        sampled_path.append(camera.matrix_world.translation.copy())
    path = add_viewport_path(demo_collection, "MGF_CamB_HemispherePath", sampled_path)
    path["frame_range"] = [args.start_frame, args.end_frame]
    add_seed_markers(demo_collection)
    add_timeline_markers(scene, keyframes)

    overview_seed = bpy.data.objects.get("Camera.002")
    if overview_seed is None or overview_seed.type != "CAMERA":
        raise RuntimeError("Camera.002 is required as the validated overview seed")
    overview_data = overview_seed.data.copy()
    overview_data.name = "MGF_CamB_OverviewCamera_Data"
    overview_data.lens = max(overview_seed.data.lens * 0.48, 8.0)
    overview_data.clip_start = 0.1
    overview_data.clip_end = 2000.0
    overview = bpy.data.objects.new("MGF_CamB_OverviewCamera", overview_data)
    link_object(demo_collection, overview)
    overview.matrix_world = overview_seed.matrix_world.copy()
    overview["description"] = "Static inspection camera showing the animated CamB physical rig"
    overview["seed_camera"] = overview_seed.name

    scene.use_preview_range = True
    scene.frame_preview_start = args.start_frame
    scene.frame_preview_end = args.end_frame
    scene.camera = camera
    camera.data.clip_start = min(camera.data.clip_start, 0.1)

    audit_frames = sorted(
        set([row["frame"] for row in keyframes] + list(range(args.start_frame, args.end_frame + 1, 5)))
    )
    tracked_objects = [camera, combined_model]
    audit = frame_audit(
        scene,
        camera,
        rig_root,
        target,
        audit_frames,
        initial_radius,
        tracked_objects,
    )
    max_radius_error = max(row["radius_error"] for row in audit)
    min_alignment = min(row["forward_alignment"] for row in audit)
    max_center_offset = max(
        math.hypot(row["target_ndc"][0] - 0.5, row["target_ndc"][1] - 0.5)
        for row in audit
    )
    if max_radius_error > 1.0e-3:
        raise RuntimeError(f"Camera escaped the constant-radius surface: {max_radius_error}")
    if min_alignment < 0.999:
        raise RuntimeError(f"Camera stopped tracking the face: {min_alignment}")
    if max_center_offset > 0.01:
        raise RuntimeError(f"Face target is not centered: {max_center_offset}")

    scene.frame_set(args.start_frame)
    output_blend = Path(args.output_blend).resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))

    moved_meshes = [combined_model.name]
    manifest = {
        "source_blend": source_blend,
        "output_blend": str(output_blend),
        "frame_range": [args.start_frame, args.end_frame],
        "camera": camera.name,
        "physical_model_roots": [combined_model.name],
        "physical_model_source_parts": combined_source_names,
        "physical_model_rigid_combined_mesh": True,
        "physical_model_viewport_visible": True,
        "physical_model_render_visible": False,
        "moved_meshes": sorted(set(moved_meshes)),
        "rig_root": rig_root.name,
        "face_target": target.name,
        "overview_camera": overview.name,
        "face_bone": args.face_bone,
        "face_head_tail": args.face_head_tail,
        "focus_at_start": list(focus),
        "initial_radius": initial_radius,
        "original_camera_radius": original_radius,
        "reference_camera_b_radius": reference_b_radius,
        "model_forward_extent": model_forward_extent,
        "model_clearance": args.model_clearance,
        "waypoints": keyframes,
        "reference_camera_positions": REFERENCE_CAMERA_POSITIONS,
        "safety": {
            "constant_radius": True,
            "max_radius_error": max_radius_error,
            "min_forward_alignment": min_alignment,
            "max_target_center_offset_ndc": max_center_offset,
            "azimuth_bounds_deg": [-27.0, 31.0],
            "elevation_bounds_deg": [-9.0, 15.0],
            "note": "Radius is no smaller than the five-camera CamB reference and the physical lens forward extent plus clearance; motion stays inside the five-camera horizontal envelope.",
        },
        "audit": audit,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("MGF_CAMB_HEMISPHERE_DEMO_BEGIN")
    print(
        json.dumps(
            {
                "output_blend": str(output_blend),
                "manifest": str(manifest_path),
                "frame_range": [args.start_frame, args.end_frame],
                "waypoints": keyframes,
                "max_radius_error": max_radius_error,
                "min_forward_alignment": min_alignment,
                "max_target_center_offset_ndc": max_center_offset,
            },
            indent=2,
        )
    )
    print("MGF_CAMB_HEMISPHERE_DEMO_END")


if __name__ == "__main__":
    main()

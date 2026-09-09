#!/usr/bin/env python3
"""Convert legacy Cylinder renders into BlenderPyMouse Core-23 datasets.

Run with the matching source animation project loaded in Blender. Existing
renders are reused; this script only evaluates the rig, camera calibration,
and deformed mouse vertices.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
import Imath
import numpy as np
import OpenEXR
from mathutils import Matrix
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_vertex_keypoint_map as support_builder  # noqa: E402


FORMAT_NAME = "blenderpymouse_core23_v1"
KEYPOINT_FORMAT = "freemoving_core23_bone_head_v1"
CAMERA_NAMES = [f"Camera.{index:03d}" for index in range(1, 9)]

# Every joint coordinate is evaluated from pose_bone.head. This is important
# for these manually animated, disconnected bones.
CORE23_SPECS = [
    {"name": "nose", "bone": "HeadEnd_M", "group": "axial", "support_mode": "surface_cap"},
    {"name": "head_center", "bone": "Head1_M", "group": "axial", "support_mode": "joint_ring"},
    {"name": "left_ear", "bone": "Ear01_L", "group": "axial", "support_mode": "joint_patch"},
    {"name": "right_ear", "bone": "Ear01_R", "group": "axial", "support_mode": "joint_patch"},
    {"name": "neck", "bone": "Neck3_M", "group": "axial", "support_mode": "joint_ring"},
    {"name": "spine_center", "bone": "Spine3_M", "group": "axial", "support_mode": "joint_ring"},
    {"name": "lumbar_spine", "bone": "Spine1_M", "group": "axial", "support_mode": "joint_ring"},
    {"name": "tail_base", "bone": "Tail0_M", "group": "tail", "support_mode": "joint_ring"},
    {"name": "tail_q1", "bone": "Tail2_M", "group": "tail", "support_mode": "joint_ring"},
    {"name": "tail_q2", "bone": "Tail4_M", "group": "tail", "support_mode": "joint_ring"},
    {"name": "tail_q3", "bone": "Tail6_M", "group": "tail", "support_mode": "joint_ring"},
    {"name": "tail_q4", "bone": "Tail8_M", "group": "tail", "support_mode": "joint_ring"},
    {"name": "tail_q5", "bone": "Tail10_M", "group": "tail", "support_mode": "joint_ring"},
    {"name": "tail_tip", "bone": "Tail11_M", "group": "tail", "support_mode": "surface_cap"},
    {"name": "left_forepaw_center", "bone": "Fingers1_L", "group": "paw", "support_mode": "joint_patch"},
    {"name": "right_forepaw_center", "bone": "Fingers1_R", "group": "paw", "support_mode": "joint_patch"},
    {"name": "left_hindpaw_center", "bone": "Toes1_L", "group": "paw", "support_mode": "joint_patch"},
    {"name": "right_hindpaw_center", "bone": "Toes1_R", "group": "paw", "support_mode": "joint_patch"},
    {"name": "left_elbow", "bone": "Wrist_L", "group": "limb_joint", "support_mode": "joint_patch"},
    {"name": "right_elbow", "bone": "Wrist_R", "group": "limb_joint", "support_mode": "joint_patch"},
    {"name": "left_knee", "bone": "Knee_L", "group": "limb_joint", "support_mode": "joint_patch"},
    {"name": "right_knee", "bone": "Knee_R", "group": "limb_joint", "support_mode": "joint_patch"},
    {"name": "abdomen_center", "bone": "Spine3_M", "group": "bottom", "support_mode": "ventral_patch"},
]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--source-prefix",
        required=True,
        choices=("walk_around", "wash_face", "climb_the_wall_brown"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--frame-count", type=int, default=91)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=0)
    parser.add_argument("--vertices-per-keypoint", type=int, default=16)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-window-radius", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--depth-num", type=int, default=192)
    parser.add_argument(
        "--invalid-depth-cameras",
        default="",
        help="Comma-separated cameras whose legacy EXR is excluded from depth supervision.",
    )
    parser.add_argument(
        "--camera-world-z-overrides",
        default="Camera.008=650",
        help="Comma-separated render-time camera world-Z overrides, for example Camera.008=650.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(args_after_dash())


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_exr(path: Path) -> np.ndarray:
    handle = OpenEXR.InputFile(str(path))
    header = handle.header()
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    channels = list(header["channels"].keys())
    selected = next((name for name in ("Z", "Depth", "V", "Y", "R") if name in channels), None)
    if selected is None:
        if not channels:
            raise ValueError(f"No channels in EXR: {path}")
        selected = channels[0]
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth = np.frombuffer(handle.channel(selected, pixel_type), dtype=np.float32).reshape(height, width)
    return depth


def write_pfm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32)
    image = np.where(np.isfinite(image) & (image > 0), image, 0.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        handle.write(b"-1.0\n")
        np.flipud(image).tofile(handle)


def write_cam_file(path: Path, extrinsic: np.ndarray, intrinsic: np.ndarray, depth: np.ndarray, depth_num: int) -> None:
    valid = depth[np.isfinite(depth) & (depth > 1e-8)]
    if valid.size:
        depth_min = float(np.percentile(valid, 0.1))
        depth_max = float(np.percentile(valid, 99.9))
        if depth_max <= depth_min:
            depth_max = depth_min + 1.0
    else:
        depth_min, depth_max = 0.1, 10.0
    interval = (depth_max - depth_min) / max(depth_num - 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("extrinsic\n")
        for row in extrinsic:
            handle.write(" ".join(f"{value:.10g}" for value in row) + "\n")
        handle.write("\nintrinsic\n")
        for row in intrinsic:
            handle.write(" ".join(f"{value:.10g}" for value in row) + "\n")
        handle.write(f"\n{depth_min:.10g} {interval:.10g} {depth_num:d} {depth_max:.10g}\n")


def write_pair_file(path: Path, view_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{view_count}\n")
        for reference in range(view_count):
            sources = [index for index in range(view_count) if index != reference]
            handle.write(f"{reference}\n")
            values = [str(len(sources))]
            for rank, source in enumerate(sources):
                values.extend((str(source), f"{1.0 / (rank + 1):.6f}"))
            handle.write(" ".join(values) + "\n")


def camera_intrinsic(camera_obj: bpy.types.Object, width: int, height: int) -> np.ndarray:
    scene = bpy.context.scene
    camera = camera_obj.data
    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    sensor_fit = camera.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= pixel_aspect * height else "VERTICAL"
    sensor_size = camera.sensor_height if sensor_fit == "VERTICAL" else camera.sensor_width
    view_fac = pixel_aspect * height if sensor_fit == "VERTICAL" else width
    fx = camera.lens * view_fac / sensor_size
    fy = fx / pixel_aspect
    cx = width / 2.0 - camera.shift_x * view_fac
    cy = height / 2.0 + camera.shift_y * view_fac / pixel_aspect
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64)


def parse_camera_z_overrides(value: str) -> dict[str, float]:
    overrides = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, raw_z = item.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"Invalid camera Z override: {item!r}")
        overrides[name.strip()] = float(raw_z)
    return overrides


def camera_extrinsic(camera_obj: bpy.types.Object, world_z_override: float | None = None) -> np.ndarray:
    blender_to_opencv = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    # Camera objects in the Cylinder projects inherit a 20x parent scale. A
    # direct matrix inverse therefore puts 0.05 scale into the extrinsic
    # rotation. Decompose first so the exported rotation stays orthonormal and
    # translation/depth remain in Blender world units.
    location, rotation_world, _scale = camera_obj.matrix_world.decompose()
    if world_z_override is not None:
        location.z = world_z_override
    world_to_camera_rotation = rotation_world.to_matrix().transposed()
    world_to_camera_translation = -(world_to_camera_rotation @ location)
    rotation = blender_to_opencv @ world_to_camera_rotation
    translation = blender_to_opencv @ world_to_camera_translation
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def project_points(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = points.shape
    flat = points.reshape(-1, 3)
    camera = (extrinsic[:3, :3] @ flat.T).T + extrinsic[:3, 3]
    projected = (intrinsic @ camera.T).T
    uv = np.full((len(flat), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(projected).all(axis=1) & (camera[:, 2] > 1e-9)
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return camera.reshape(shape), uv.reshape(*shape[:-1], 2)


def support_visibility(
    camera: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    visible = np.zeros(camera.shape[:-1], dtype=bool)
    residual = np.full(camera.shape[:-1], np.nan, dtype=np.float64)
    for index in np.ndindex(visible.shape):
        z = float(camera[index][2])
        u, v = (float(value) for value in uv[index])
        if not (math.isfinite(z) and z > 0 and math.isfinite(u) and math.isfinite(v)):
            continue
        x, y = int(round(u)), int(round(v))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if not np.any(mask[y0:y1, x0:x1]):
            continue
        values = depth[y0:y1, x0:x1]
        values = values[np.isfinite(values) & (values > 1e-8)]
        if not values.size:
            continue
        minimum = float(np.min(np.abs(values.astype(np.float64) - z)))
        residual[index] = minimum
        visible[index] = minimum <= max(absolute_tolerance, relative_tolerance * abs(z))
    return visible, residual


def build_support_map(
    mesh_obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    vertices_per_keypoint: int,
) -> tuple[dict[str, list[int]], dict]:
    all_weights = support_builder.vertex_group_weights(mesh_obj)
    support_ids: dict[str, list[int]] = {}
    audit = {
        "format": "mousegeoforge_core23_vertex_support_v1",
        "topology": support_builder.mesh_topology_signature(mesh_obj),
        "keypoint_format": KEYPOINT_FORMAT,
        "keypoints": [],
    }
    for spec in CORE23_SPECS:
        bone = armature_obj.data.bones.get(spec["bone"])
        if bone is None:
            raise KeyError(f"Missing Core-23 bone: {spec['bone']}")
        builder_mode = "joint_patch" if spec["support_mode"] == "ventral_patch" else spec["support_mode"]
        support_builder.LANDMARK_ANCHOR_FRACTIONS[bone.name] = 0.0
        support_builder.LANDMARK_SELECTION_MODES[bone.name] = builder_mode
        candidates, geometry = support_builder.candidates_for_landmark(
            mesh_obj,
            armature_obj,
            bone,
            all_weights,
            min_group_weight=0.02,
        )
        ordered = support_builder.angularly_balanced_order(
            candidates,
            angular_sectors=12,
            vertices_per_keypoint=vertices_per_keypoint,
        )
        if spec["support_mode"] == "ventral_patch":
            pool = ordered[: max(vertices_per_keypoint * 6, vertices_per_keypoint)]
            pool.sort(key=lambda item: (mesh_obj.matrix_world @ mesh_obj.data.vertices[item.vertex_id].co).z)
            selected = pool[:vertices_per_keypoint]
        else:
            selected = ordered[:vertices_per_keypoint]
        support_ids[spec["name"]] = [item.vertex_id for item in selected]
        audit["keypoints"].append(
            {
                **spec,
                "endpoint": "head",
                "support_vertex_ids": support_ids[spec["name"]],
                "geometry": geometry,
                "selection": support_builder.selection_audit(selected, angular_sectors=12),
            }
        )
    return support_ids, audit


def evaluated_core23(
    frame: int,
    armature_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    support_ids: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    count = len(CORE23_SPECS)
    support_count = max(len(values) for values in support_ids.values())
    joint_world = np.empty((count, 3), dtype=np.float64)
    support_vertex_ids = np.full((count, support_count), -1, dtype=np.int32)
    support_world = np.full((count, support_count, 3), np.nan, dtype=np.float64)
    for index, spec in enumerate(CORE23_SPECS):
        pose_bone = armature_obj.pose.bones.get(spec["bone"])
        if pose_bone is None:
            raise KeyError(spec["bone"])
        joint_world[index] = np.asarray(armature_obj.matrix_world @ pose_bone.head, dtype=np.float64)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    temp_mesh = evaluated.to_mesh()
    try:
        mesh_world = evaluated.matrix_world
        for index, spec in enumerate(CORE23_SPECS):
            ids = support_ids[spec["name"]]
            support_vertex_ids[index, : len(ids)] = ids
            for support_index, vertex_id in enumerate(ids):
                support_world[index, support_index] = np.asarray(
                    mesh_world @ temp_mesh.vertices[vertex_id].co,
                    dtype=np.float64,
                )
    finally:
        evaluated.to_mesh_clear()
    surface_world = np.nanmean(support_world, axis=1)
    return joint_world, surface_world, support_vertex_ids, support_world


def disable_hair() -> None:
    for obj in bpy.data.objects:
        if obj.type != "CURVES":
            continue
        obj.hide_viewport = True
        obj.hide_render = True
        try:
            obj.hide_set(True)
        except RuntimeError:
            pass
        for modifier in obj.modifiers:
            modifier.show_viewport = False
    bpy.context.view_layer.update()


def save_jpeg(source: Path, destination: Path, quality: int) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        size = rgb.size
        rgb.save(destination, quality=quality, subsampling=0)
    return size


def finite_list(array: np.ndarray) -> list[float] | None:
    return [float(value) for value in array] if np.isfinite(array).all() else None


def convert_view(
    source_dir: Path,
    scene_root: Path,
    frame: int,
    view_id: int,
    camera_obj: bpy.types.Object,
    joint_world: np.ndarray,
    surface_world: np.ndarray,
    support_world: np.ndarray,
    depth_available: bool,
    camera_world_z_override: float | None,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray], int]:
    stem = f"{view_id:08d}"
    image_path = scene_root / "blended_images" / f"{stem}.jpg"
    masked_path = scene_root / "blended_images" / f"{stem}_masked.jpg"
    mask_path = scene_root / "masks" / f"{stem}.png"
    depth_path = scene_root / "rendered_depth_maps" / f"{stem}.pfm"
    preview_path = scene_root / "rendered_depth_maps" / f"{stem}_preview.png"
    cam_path = scene_root / "cams" / f"{stem}_cam.txt"

    width, height = save_jpeg(source_dir / "normal_image.png", image_path, args.jpeg_quality)
    masked_size = save_jpeg(source_dir / "target.png", masked_path, args.jpeg_quality)
    if masked_size != (width, height):
        raise ValueError(f"Image size mismatch in {source_dir}")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "contour.png", mask_path)
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L")) > 0
    if mask.shape != (height, width):
        raise ValueError(f"Mask size mismatch in {source_dir}: {mask.shape} vs {(height, width)}")

    source_depth = read_exr(source_dir / "depth_image.exr")
    source_depth = np.where(np.isfinite(source_depth) & (source_depth > 0), source_depth, 0.0).astype(np.float32)
    if source_depth.shape != (height, width):
        raise ValueError(f"Depth size mismatch in {source_dir}: {source_depth.shape} vs {(height, width)}")
    depth = source_depth if depth_available else np.zeros_like(source_depth)
    write_pfm(depth_path, depth)
    shutil.copy2(source_dir / "depth_image.png", preview_path)

    intrinsic = camera_intrinsic(camera_obj, width, height)
    extrinsic = camera_extrinsic(camera_obj, camera_world_z_override)
    write_cam_file(cam_path, extrinsic, intrinsic, depth, args.depth_num)

    joint_camera, joint_uv = project_points(joint_world, extrinsic, intrinsic)
    surface_camera, _surface_uv = project_points(surface_world, extrinsic, intrinsic)
    support_camera, support_uv = project_points(support_world, extrinsic, intrinsic)
    if depth_available:
        support_visible, support_residual = support_visibility(
            support_camera,
            support_uv,
            depth,
            mask,
            args.depth_relative_tolerance,
            args.depth_absolute_tolerance,
            args.depth_window_radius,
        )
    else:
        support_visible = np.zeros(support_camera.shape[:-1], dtype=bool)
        support_residual = np.full(support_camera.shape[:-1], np.nan, dtype=np.float64)
    in_frame = (
        np.isfinite(joint_uv).all(axis=1)
        & (joint_camera[:, 2] > 0)
        & (joint_uv[:, 0] >= 0)
        & (joint_uv[:, 0] < width)
        & (joint_uv[:, 1] >= 0)
        & (joint_uv[:, 1] < height)
    )
    surface_visible = support_visible.any(axis=1)
    keypoints = []
    for index, spec in enumerate(CORE23_SPECS):
        valid_support = np.isfinite(support_world[index]).all(axis=1)
        residuals = support_residual[index][np.isfinite(support_residual[index])]
        keypoints.append(
            {
                "id": index,
                "name": spec["name"],
                "group": spec["group"],
                "bone": spec["bone"],
                "endpoint": "head",
                "defined_by_dataset": True,
                "joint_world_xyz": finite_list(joint_world[index]),
                "surface_world_xyz": finite_list(surface_world[index]),
                "camera_xyz": finite_list(joint_camera[index]),
                "surface_camera_xyz": finite_list(surface_camera[index]),
                "uv": finite_list(joint_uv[index]),
                "x": float(joint_uv[index, 0]),
                "y": float(joint_uv[index, 1]),
                "camera_depth": float(joint_camera[index, 2]),
                "surface_depth": float(surface_camera[index, 2]),
                "in_frame": bool(in_frame[index]),
                "surface_visible": bool(surface_visible[index]),
                "visible": bool(surface_visible[index]),
                "occluded_but_annotated": bool(in_frame[index] and not surface_visible[index]),
                "confidence": 1.0,
                "support_visible_count": int(support_visible[index].sum()),
                "support_vertex_count": int(valid_support.sum()),
                "support_min_depth_residual": float(residuals.min()) if residuals.size else None,
            }
        )

    relative = lambda path: str(path.relative_to(scene_root)).replace("/", "\\")
    view = {
        "frame": frame,
        "view_id": view_id,
        "camera": camera_obj.name,
        "image": relative(image_path),
        "masked_image": relative(masked_path),
        "mask": relative(mask_path),
        "depth_pfm": relative(depth_path),
        "depth_preview": relative(preview_path),
        "cam": relative(cam_path),
        "depth_available": depth_available,
        "depth_source_exr": str(source_dir / "depth_image.exr"),
        "depth_source_validation": "passed" if depth_available else "failed_systematic_camera_depth_scale_check",
        "camera_world_z_override": camera_world_z_override,
        "camera_extrinsic_source": "render_time_override" if camera_world_z_override is not None else "blend_scene",
        "intrinsic": intrinsic.tolist(),
        "extrinsic": extrinsic.tolist(),
        "keypoint_format": KEYPOINT_FORMAT,
        "keypoints": keypoints,
        "source_directory": str(source_dir),
        "source_global_id": view_id * args.frame_count + frame - 1,
    }
    annotation_path = scene_root / "annotations" / f"{stem}.json"
    write_json_atomic(annotation_path, view)
    arrays = {
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "joint_camera": joint_camera,
        "uv": joint_uv,
        "in_frame": in_frame,
        "surface_visible": surface_visible,
        "support_visible_count": support_visible.sum(axis=1).astype(np.int16),
        "depth_available": np.asarray(depth_available),
    }
    return view, arrays, int(surface_visible.sum())


def write_frame_keypoints(
    scene_root: Path,
    frame: int,
    joint_world: np.ndarray,
    surface_world: np.ndarray,
    support_vertex_ids: np.ndarray,
    support_world: np.ndarray,
    views: list[dict],
    arrays: list[dict[str, np.ndarray]],
    armature_name: str = "DeformationSystem",
    mesh_name: str = "MS_Mouse",
) -> None:
    names = [spec["name"] for spec in CORE23_SPECS]
    payload = {
        "format": KEYPOINT_FORMAT,
        "frame": frame,
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "armature": armature_name,
        "mesh": mesh_name,
        "coordinate_system": "Blender world; camera_xyz uses OpenCV +Z forward",
        "joint_coordinate_rule": "pose bone head only",
        "names": names,
        "groups": [spec["group"] for spec in CORE23_SPECS],
        "bones": [spec["bone"] for spec in CORE23_SPECS],
        "endpoints": ["head"] * len(names),
        "joint_world_xyz": joint_world.tolist(),
        "surface_world_xyz": surface_world.tolist(),
        "support_vertex_ids": support_vertex_ids.tolist(),
        "views": [{"view_id": view["view_id"], "camera": view["camera"], "keypoints": view["keypoints"]} for view in views],
    }
    write_json_atomic(scene_root / "keypoints.json", payload)
    temporary = scene_root / "keypoints.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            format=np.asarray(KEYPOINT_FORMAT),
            frame=np.asarray(frame, dtype=np.int32),
            names=np.asarray(names),
            groups=np.asarray([spec["group"] for spec in CORE23_SPECS]),
            bones=np.asarray([spec["bone"] for spec in CORE23_SPECS]),
            endpoints=np.asarray(["head"] * len(names)),
            view_ids=np.arange(len(views), dtype=np.int32),
            camera_names=np.asarray([view["camera"] for view in views]),
            intrinsics=np.stack([item["intrinsic"] for item in arrays]).astype(np.float32),
            extrinsics=np.stack([item["extrinsic"] for item in arrays]).astype(np.float32),
            joint_world_xyz=joint_world.astype(np.float32),
            surface_world_xyz=surface_world.astype(np.float32),
            support_vertex_ids=support_vertex_ids,
            support_world_xyz=support_world.astype(np.float32),
            joint_camera_xyz=np.stack([item["joint_camera"] for item in arrays]).astype(np.float32),
            uv=np.stack([item["uv"] for item in arrays]).astype(np.float32),
            in_frame=np.stack([item["in_frame"] for item in arrays]),
            surface_visible=np.stack([item["surface_visible"] for item in arrays]),
            support_visible_count=np.stack([item["support_visible_count"] for item in arrays]),
            depth_available=np.asarray([item["depth_available"] for item in arrays], dtype=bool),
        )
    temporary.replace(scene_root / "keypoints.npz")


def source_frame_dir(source_root: Path, prefix: str, view_id: int, frame: int, frame_count: int) -> Path:
    if prefix == "climb_the_wall_brown":
        # These two historical paired batches use different indexing schemes:
        # cam12 stores all C1 frames followed by all C2 frames, whereas cam45
        # alternates C4/F1, C5/F1, C4/F2, C5/F2, ... .
        if view_id < 2:
            camera_folder = "climb_the_wall_brown_cam12"
            global_id = view_id * frame_count + frame - 1
        elif view_id == 2:
            camera_folder = "climb_the_wall_brown_cam3"
            global_id = 2 * frame_count + frame - 1
        elif view_id < 5:
            camera_folder = "climb_the_wall_brown_cam45"
            global_id = 3 * frame_count + 2 * (frame - 1) + (view_id - 3)
        else:
            camera_folder = f"climb_the_wall_brown_cam{view_id + 1}"
            global_id = view_id * frame_count + frame - 1
    else:
        global_id = view_id * frame_count + frame - 1
        camera_folder = f"{prefix}_cam{view_id + 1}"
    return source_root / camera_folder / f"{global_id:012d}"


def validate_sources(source_root: Path, prefix: str, frame_count: int) -> None:
    missing = []
    expected_files = ("normal_image.png", "target.png", "contour.png", "depth_image.exr", "depth_image.png")
    for view_id in range(8):
        for frame in range(1, frame_count + 1):
            root = source_frame_dir(source_root, prefix, view_id, frame, frame_count)
            for name in expected_files:
                if not (root / name).is_file():
                    missing.append(str(root / name))
                    if len(missing) >= 20:
                        break
            if len(missing) >= 20:
                break
    if missing:
        raise FileNotFoundError("Missing Cylinder source files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    source_root = Path(bpy.path.abspath(args.source_root)).resolve()
    output_root = Path(bpy.path.abspath(args.output_root)).resolve()
    validate_sources(source_root, args.source_prefix, args.frame_count)
    armature_obj = bpy.data.objects.get(args.armature)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise KeyError(args.armature)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(args.mesh)
    cameras = [bpy.data.objects.get(name) for name in CAMERA_NAMES]
    if any(camera is None or camera.type != "CAMERA" for camera in cameras):
        raise KeyError(f"Expected cameras not found: {CAMERA_NAMES}")

    disable_hair()
    support_ids, support_audit = build_support_map(mesh_obj, armature_obj, args.vertices_per_keypoint)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "core23_vertex_support_map.pkl").open("wb") as handle:
        pickle.dump(
            {
                "format": support_audit["format"],
                "keypoint_format": KEYPOINT_FORMAT,
                "specs": CORE23_SPECS,
                "support_vertex_ids": support_ids,
            },
            handle,
            protocol=4,
        )
    write_json_atomic(output_root / "core23_vertex_support_map.audit.json", support_audit)

    frame_start = max(1, args.frame_start)
    frame_end = args.frame_end if args.frame_end > 0 else args.frame_count
    frame_end = min(args.frame_count, frame_end)
    if frame_start > frame_end:
        raise ValueError(f"Invalid frame selection: {frame_start}>{frame_end}")
    selected_frames = list(range(frame_start, frame_end + 1))
    invalid_depth_cameras = {
        name.strip()
        for name in args.invalid_depth_cameras.split(",")
        if name.strip()
    }
    camera_z_overrides = parse_camera_z_overrides(args.camera_world_z_overrides)
    unknown_override_cameras = sorted(set(camera_z_overrides) - set(CAMERA_NAMES))
    if unknown_override_cameras:
        raise ValueError(f"Unknown camera Z overrides: {unknown_override_cameras}")

    visible_total = 0
    for frame in selected_frames:
        scene_root = output_root / f"frame_{frame:06d}"
        joint_world, surface_world, support_vertex_ids, support_world = evaluated_core23(
            frame,
            armature_obj,
            mesh_obj,
            support_ids,
        )
        views = []
        arrays = []
        for view_id, camera_obj in enumerate(cameras):
            source_dir = source_frame_dir(source_root, args.source_prefix, view_id, frame, args.frame_count)
            view, view_arrays, visible_count = convert_view(
                source_dir,
                scene_root,
                frame,
                view_id,
                camera_obj,
                joint_world,
                surface_world,
                support_world,
                camera_obj.name not in invalid_depth_cameras,
                camera_z_overrides.get(camera_obj.name),
                args,
            )
            views.append(view)
            arrays.append(view_arrays)
            visible_total += visible_count
        write_pair_file(scene_root / "cams" / "pair.txt", len(cameras))
        scene_meta = {
            "frame": frame,
            "keypoint_format": KEYPOINT_FORMAT,
            "keypoint_names": [spec["name"] for spec in CORE23_SPECS],
            "views": views,
        }
        write_json_atomic(scene_root / "scene_meta.json", scene_meta)
        write_frame_keypoints(
            scene_root,
            frame,
            joint_world,
            surface_world,
            support_vertex_ids,
            support_world,
            views,
            arrays,
        )
        if frame == frame_start or frame == frame_end or frame % max(1, args.progress_every) == 0:
            print(
                "core23_conversion_progress",
                f"dataset={output_root.name}",
                f"frame={frame}/{frame_end}",
                flush=True,
            )

    metadata = {
        "format": FORMAT_NAME,
        "dataset_name": output_root.name,
        "dataset_root": str(output_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "source_cylinder_root": str(source_root),
        "source_prefix": args.source_prefix,
        "frames": [frame_start, frame_end],
        "frame_numbers": selected_frames,
        "scene_count": len(selected_frames),
        "incomplete_scene_count": 0,
        "scene_folders": [f"frame_{frame:06d}" for frame in selected_frames],
        "cameras": CAMERA_NAMES,
        "view_counts": {"8": len(selected_frames)},
        "render_resolution": [512, 512],
        "image_format": "JPEG",
        "mask_format": "PNG",
        "depth_format": "PFM converted from source EXR",
        "depth_units": "Blender world units",
        "source_depth_excludes_transparent_cylinder": True,
        "source_depth_exclusion_note": "The transparent cylinder was excluded when the source EXR depth pass was rendered.",
        "camera_world_z_overrides": camera_z_overrides,
        "camera_override_note": (
            "Explicit world-Z overrides listed above are applied before camera extrinsics "
            "and projected keypoints are exported."
        ),
        "depth_invalid_cameras": sorted(invalid_depth_cameras),
        "depth_invalid_camera_policy": "standard PFM is all-zero; RGB, mask, camera, and keypoints remain available",
        "layout": {
            "images": "frame_xxxxxx/blended_images/00000000.jpg",
            "masked_images": "frame_xxxxxx/blended_images/00000000_masked.jpg",
            "masks": "frame_xxxxxx/masks/00000000.png",
            "depths": "frame_xxxxxx/rendered_depth_maps/00000000.pfm",
            "depth_previews": "frame_xxxxxx/rendered_depth_maps/00000000_preview.png",
            "cameras": "frame_xxxxxx/cams/00000000_cam.txt",
            "pair_file": "frame_xxxxxx/cams/pair.txt",
            "annotations": "frame_xxxxxx/annotations/00000000.json",
        },
        "keypoint_annotations": {
            "format": KEYPOINT_FORMAT,
            "keypoint_count": len(CORE23_SPECS),
            "names": [spec["name"] for spec in CORE23_SPECS],
            "groups": [spec["group"] for spec in CORE23_SPECS],
            "bones": [spec["bone"] for spec in CORE23_SPECS],
            "endpoints": ["head"] * len(CORE23_SPECS),
            "joint_coordinate_rule": "pose bone head only; parent bone tails are never substituted",
            "tail_tip_note": "Tail11_M.head is used as the terminal head-only approximation",
            "tail_sampling_note": "Tail0/2/4/6/8/10/11 heads approximate base, five intermediate points, and tip",
            "vertex_support_map": str(output_root / "core23_vertex_support_map.pkl"),
            "vertex_support_audit": str(output_root / "core23_vertex_support_map.audit.json"),
            "visibility_rule": "at least one support vertex agrees with EXR depth and falls inside the mouse contour mask",
            "depth_relative_tolerance": args.depth_relative_tolerance,
            "depth_absolute_tolerance": args.depth_absolute_tolerance,
            "depth_window_radius": args.depth_window_radius,
            "visible_keypoint_instances": visible_total,
        },
    }
    write_json_atomic(output_root / "metadata.json", metadata)
    print(
        "core23_conversion_complete",
        f"dataset={output_root}",
        f"frames={len(selected_frames)}",
        f"views={len(selected_frames) * len(cameras)}",
        f"visible_instances={visible_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert the legacy black-fur Orbbec top-view sequence to BlenderPyMouse Core-23."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import convert_cylinder_renders_to_blenderpymouse_core23 as core23  # noqa: E402


LEGACY_TO_CORE23 = (0, 1, 2, 3, 4, 5, 6, 7, 10, 13)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--armature", default="DeformationSystem.001")
    parser.add_argument("--mesh", default="MS_Mouse.001")
    parser.add_argument(
        "--attached-meshes",
        nargs="*",
        default=["MouseFiberWindow", "立方体"],
        help="Rendered head-mounted meshes that belong to the mouse instance mask.",
    )
    parser.add_argument("--camera", default="CameraOrbbec")
    parser.add_argument(
        "--intrinsic-source",
        choices=("camera_json", "legacy_fit"),
        default="camera_json",
        help="Use exact fixed-camera metadata or retain the legacy per-frame fit for old renders.",
    )
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=273)
    parser.add_argument("--vertices-per-keypoint", type=int, default=16)
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--mask-boundary-luminance-threshold", type=int, default=140)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-window-radius", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--depth-num", type=int, default=192)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(args_after_dash())


def source_dir(root: Path, frame: int) -> Path:
    return root / f"{frame:012d}"


def validate_sources(root: Path, frames: list[int], intrinsic_source: str) -> None:
    required = ["normal_image.png", "depth_image.exr", "depth_image.png", "keypoint_coords.npy"]
    if intrinsic_source == "camera_json":
        required.append("camera.json")
    missing = [str(source_dir(root, frame) / name) for frame in frames for name in required if not (source_dir(root, frame) / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing[:30]))


def fit_intrinsic_from_legacy(
    joint_world: np.ndarray,
    extrinsic: np.ndarray,
    legacy_normalized: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    camera_xyz = (extrinsic[:3, :3] @ joint_world[np.asarray(LEGACY_TO_CORE23)].T).T + extrinsic[:3, 3]
    normalized = camera_xyz[:, :2] / camera_xyz[:, 2:3]
    legacy_pixels = legacy_normalized * np.asarray((width, height), dtype=np.float64)
    fit_x = np.column_stack((normalized[:, 0], np.ones(len(normalized))))
    fit_y = np.column_stack((normalized[:, 1], np.ones(len(normalized))))
    fx, cx = np.linalg.lstsq(fit_x, legacy_pixels[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(fit_y, legacy_pixels[:, 1], rcond=None)[0]
    if not (0.5 * width < fx < 2.0 * width and 0.5 * height < fy < 2.0 * height):
        raise ValueError(f"Implausible fitted intrinsic: fx={fx}, fy={fy}")
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64)


def rasterized_meshes_mask(
    mesh_objs: list[bpy.types.Object],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> Image.Image:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for mesh_obj in mesh_objs:
        evaluated = mesh_obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            world = np.asarray(
                [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices],
                dtype=np.float64,
            )
            camera_xyz, uv = core23.project_points(world, extrinsic, intrinsic)
            for triangle in mesh.loop_triangles:
                indices = list(triangle.vertices)
                if not np.all(camera_xyz[indices, 2] > 0) or not np.isfinite(uv[indices]).all():
                    continue
                draw.polygon([tuple(float(value) for value in uv[index]) for index in indices], fill=255)
        finally:
            evaluated.to_mesh_clear()
    return mask


def refine_blackfur_mask(
    mask: Image.Image,
    rgb: Image.Image,
    boundary_radius: int,
    luminance_threshold: int,
) -> Image.Image:
    if boundary_radius <= 0:
        return mask
    expanded = np.asarray(mask.filter(ImageFilter.MaxFilter(boundary_radius * 2 + 1))) > 0
    interior = np.asarray(mask.filter(ImageFilter.MinFilter(3))) > 0
    luminance = np.asarray(rgb.convert("L"))
    refined = interior | (expanded & (luminance < luminance_threshold))
    return Image.fromarray((refined.astype(np.uint8) * 255), mode="L")


def relative(scene_root: Path, path: Path) -> str:
    return str(path.relative_to(scene_root)).replace("/", "\\")


def convert_frame(
    frame: int,
    source_root: Path,
    output_root: Path,
    armature_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    instance_mesh_objs: list[bpy.types.Object],
    camera_obj: bpy.types.Object,
    support_ids: dict[str, list[int]],
    args: argparse.Namespace,
) -> tuple[int, float, float, np.ndarray]:
    source = source_dir(source_root, frame)
    scene_root = output_root / f"frame_{frame:06d}"
    image_dir = scene_root / "blended_images"
    mask_dir = scene_root / "masks"
    depth_dir = scene_root / "rendered_depth_maps"
    cam_dir = scene_root / "cams"
    annotation_dir = scene_root / "annotations"
    for directory in (image_dir, mask_dir, depth_dir, cam_dir, annotation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    joint_world, surface_world, support_vertex_ids, support_world = core23.evaluated_core23(
        frame, armature_obj, mesh_obj, support_ids
    )
    with Image.open(source / "normal_image.png") as source_image:
        rgb = source_image.convert("RGB")
        width, height = rgb.size
    extrinsic = core23.camera_extrinsic(camera_obj)
    legacy_normalized = np.load(source / "keypoint_coords.npy").astype(np.float64)
    if args.intrinsic_source == "camera_json":
        camera_record = json.loads((source / "camera.json").read_text(encoding="utf-8"))
        intrinsic = np.asarray(camera_record["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(camera_record["extrinsic"], dtype=np.float64)
        camera_intrinsic_source = "fixed_camera_json"
    else:
        intrinsic = fit_intrinsic_from_legacy(joint_world, extrinsic, legacy_normalized, width, height)
        camera_intrinsic_source = "per_frame_linear_fit_from_legacy_10-point_projection"
    geometry_mask = rasterized_meshes_mask(instance_mesh_objs, extrinsic, intrinsic, width, height)
    mask = refine_blackfur_mask(
        geometry_mask,
        rgb,
        args.mask_dilation,
        args.mask_boundary_luminance_threshold,
    )

    image_path = image_dir / "00000000.jpg"
    masked_path = image_dir / "00000000_masked.jpg"
    mask_path = mask_dir / "00000000.png"
    rgb.save(image_path, quality=args.jpeg_quality, subsampling=0)
    masked = Image.new("RGB", rgb.size)
    masked.paste(rgb, mask=mask)
    masked.save(masked_path, quality=args.jpeg_quality, subsampling=0)
    mask.save(mask_path)
    mask_array = np.asarray(mask) > 0

    depth = core23.read_exr(source / "depth_image.exr")
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
    if depth.shape != (height, width):
        raise ValueError(f"Depth shape mismatch at frame {frame}: {depth.shape} vs {(height, width)}")
    depth_path = depth_dir / "00000000.pfm"
    depth_preview_path = depth_dir / "00000000_preview.png"
    core23.write_pfm(depth_path, depth)
    shutil.copy2(source / "depth_image.png", depth_preview_path)
    cam_path = cam_dir / "00000000_cam.txt"
    core23.write_cam_file(cam_path, extrinsic, intrinsic, depth, args.depth_num)
    core23.write_pair_file(cam_dir / "pair.txt", 1)

    joint_camera, joint_uv = core23.project_points(joint_world, extrinsic, intrinsic)
    surface_camera, _ = core23.project_points(surface_world, extrinsic, intrinsic)
    support_camera, support_uv = core23.project_points(support_world, extrinsic, intrinsic)
    support_visible, support_residual = core23.support_visibility(
        support_camera,
        support_uv,
        depth,
        mask_array,
        args.depth_relative_tolerance,
        args.depth_absolute_tolerance,
        args.depth_window_radius,
    )
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
    for index, spec in enumerate(core23.CORE23_SPECS):
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
                "joint_world_xyz": core23.finite_list(joint_world[index]),
                "surface_world_xyz": core23.finite_list(surface_world[index]),
                "camera_xyz": core23.finite_list(joint_camera[index]),
                "surface_camera_xyz": core23.finite_list(surface_camera[index]),
                "uv": core23.finite_list(joint_uv[index]),
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

    legacy_pixels = legacy_normalized * np.asarray((width, height), dtype=np.float64)
    legacy_projection = joint_uv[np.asarray(LEGACY_TO_CORE23)]
    legacy_errors = np.linalg.norm(legacy_projection - legacy_pixels, axis=1)
    view = {
        "frame": frame,
        "view_id": 0,
        "camera": camera_obj.name,
        "image": relative(scene_root, image_path),
        "masked_image": relative(scene_root, masked_path),
        "mask": relative(scene_root, mask_path),
        "depth_pfm": relative(scene_root, depth_path),
        "depth_preview": relative(scene_root, depth_preview_path),
        "cam": relative(scene_root, cam_path),
        "depth_available": True,
        "depth_source_exr": str(source / "depth_image.exr"),
        "depth_source_validation": "passed",
        "intrinsic": intrinsic.tolist(),
        "extrinsic": extrinsic.tolist(),
        "camera_intrinsic_source": camera_intrinsic_source,
        "keypoint_format": core23.KEYPOINT_FORMAT,
        "keypoints": keypoints,
        "source_directory": str(source),
        "source_global_id": frame,
        "legacy_keypoints_normalized": legacy_normalized.tolist(),
        "legacy_keypoint_core23_indices": list(LEGACY_TO_CORE23),
        "legacy_projection_error_mean_px": float(legacy_errors.mean()),
        "legacy_projection_error_max_px": float(legacy_errors.max()),
    }
    core23.write_json_atomic(annotation_dir / "00000000.json", view)
    scene_meta = {
        "frame": frame,
        "keypoint_format": core23.KEYPOINT_FORMAT,
        "keypoint_names": [spec["name"] for spec in core23.CORE23_SPECS],
        "views": [view],
    }
    core23.write_json_atomic(scene_root / "scene_meta.json", scene_meta)
    arrays = [
        {
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "joint_camera": joint_camera,
            "uv": joint_uv,
            "in_frame": in_frame,
            "surface_visible": surface_visible,
            "support_visible_count": support_visible.sum(axis=1),
            "depth_available": True,
        }
    ]
    core23.write_frame_keypoints(
        scene_root,
        frame,
        joint_world,
        surface_world,
        support_vertex_ids,
        support_world,
        [view],
        arrays,
        armature_name=armature_obj.name,
        mesh_name=mesh_obj.name,
    )
    return int(surface_visible.sum()), float(legacy_errors.mean()), float(legacy_errors.max()), intrinsic


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    frames = list(range(args.frame_start, args.frame_end + 1))
    validate_sources(source_root, frames, args.intrinsic_source)
    armature_obj = bpy.data.objects.get(args.armature)
    mesh_obj = bpy.data.objects.get(args.mesh)
    camera_obj = bpy.data.objects.get(args.camera)
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise KeyError(args.armature)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(args.mesh)
    if camera_obj is None or camera_obj.type != "CAMERA":
        raise KeyError(args.camera)
    attached_mesh_objs = []
    for object_name in args.attached_meshes:
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "MESH":
            raise KeyError(object_name)
        if obj.hide_render:
            raise ValueError(f"Attached instance mesh is hidden from rendering: {object_name}")
        attached_mesh_objs.append(obj)
    instance_mesh_objs = [mesh_obj, *attached_mesh_objs]

    core23.disable_hair()
    support_ids, support_audit = core23.build_support_map(mesh_obj, armature_obj, args.vertices_per_keypoint)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "core23_vertex_support_map.pkl").open("wb") as handle:
        pickle.dump(
            {
                "format": support_audit["format"],
                "keypoint_format": core23.KEYPOINT_FORMAT,
                "specs": core23.CORE23_SPECS,
                "support_vertex_ids": support_ids,
            },
            handle,
            protocol=4,
        )
    core23.write_json_atomic(output_root / "core23_vertex_support_map.audit.json", support_audit)

    visible_total = 0
    legacy_mean_errors = []
    legacy_max_errors = []
    fitted_intrinsics = []
    for frame in frames:
        visible, legacy_mean, legacy_max, intrinsic = convert_frame(
            frame,
            source_root,
            output_root,
            armature_obj,
            mesh_obj,
            instance_mesh_objs,
            camera_obj,
            support_ids,
            args,
        )
        visible_total += visible
        legacy_mean_errors.append(legacy_mean)
        legacy_max_errors.append(legacy_max)
        fitted_intrinsics.append(intrinsic)
        if frame == frames[0] or frame == frames[-1] or frame % max(1, args.progress_every) == 0:
            print(f"openfield_core23_progress frame={frame}/{frames[-1]}", flush=True)

    with Image.open(source_dir(source_root, frames[0]) / "normal_image.png") as image:
        width, height = image.size
    fitted_intrinsics_array = np.stack(fitted_intrinsics)
    metadata = {
        "format": core23.FORMAT_NAME,
        "dataset_name": output_root.name,
        "dataset_root": str(output_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "source_render_root": str(source_root),
        "source_sequence": "orbbec_black_CW",
        "frames": [frames[0], frames[-1]],
        "frame_numbers": frames,
        "scene_count": len(frames),
        "incomplete_scene_count": 0,
        "scene_folders": [f"frame_{frame:06d}" for frame in frames],
        "cameras": [camera_obj.name],
        "view_counts": {"1": len(frames)},
        "single_view_sequence": True,
        "render_resolution": [width, height],
        "image_format": "JPEG",
        "mask_format": "PNG",
        "mask_source": "evaluated mouse-body and rendered head-mounted mesh union with black-fur RGB boundary refinement",
        "instance_meshes": [obj.name for obj in instance_mesh_objs],
        "mouse_attached_apparatus_meshes": [obj.name for obj in attached_mesh_objs],
        "mask_boundary_search_pixels": args.mask_dilation,
        "mask_boundary_luminance_threshold": args.mask_boundary_luminance_threshold,
        "depth_format": "PFM converted from source EXR",
        "depth_units": "Blender world units",
        "depth_invalid_cameras": [],
        "camera_calibration": {
            "camera": camera_obj.name,
            "extrinsic_source": "source camera.json" if args.intrinsic_source == "camera_json" else "current blend transform; verified against source EXR",
            "intrinsic_source": "source fixed-camera camera.json" if args.intrinsic_source == "camera_json" else "per-frame fx/fy/cx/cy linear fit from source keypoint_coords.npy and matching Blender bone heads",
            "intrinsic_is_per_frame": args.intrinsic_source != "camera_json",
            "intrinsic_median": np.median(fitted_intrinsics_array, axis=0).tolist(),
            "fx_range": [float(fitted_intrinsics_array[:, 0, 0].min()), float(fitted_intrinsics_array[:, 0, 0].max())],
            "fy_range": [float(fitted_intrinsics_array[:, 1, 1].min()), float(fitted_intrinsics_array[:, 1, 1].max())],
            "cx_range": [float(fitted_intrinsics_array[:, 0, 2].min()), float(fitted_intrinsics_array[:, 0, 2].max())],
            "cy_range": [float(fitted_intrinsics_array[:, 1, 2].min()), float(fitted_intrinsics_array[:, 1, 2].max())],
            "source_scene_animated_sensor_width_was_not_used": args.intrinsic_source == "legacy_fit",
            "fixed_camera_metadata_used": args.intrinsic_source == "camera_json",
            "legacy_projection_mean_error_px": float(np.mean(legacy_mean_errors)),
            "legacy_projection_max_error_px": float(np.max(legacy_max_errors)),
        },
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
            "format": core23.KEYPOINT_FORMAT,
            "keypoint_count": len(core23.CORE23_SPECS),
            "names": [spec["name"] for spec in core23.CORE23_SPECS],
            "groups": [spec["group"] for spec in core23.CORE23_SPECS],
            "bones": [spec["bone"] for spec in core23.CORE23_SPECS],
            "endpoints": ["head"] * len(core23.CORE23_SPECS),
            "joint_coordinate_rule": "pose bone head only",
            "legacy_keypoints": "source keypoint_coords.npy retained per view for calibration audit",
            "vertex_support_map": str(output_root / "core23_vertex_support_map.pkl"),
            "visibility_rule": "at least one support vertex agrees with EXR depth and lies inside the rasterized mouse mesh mask",
            "depth_relative_tolerance": args.depth_relative_tolerance,
            "depth_absolute_tolerance": args.depth_absolute_tolerance,
            "depth_window_radius": args.depth_window_radius,
            "visible_keypoint_instances": visible_total,
        },
    }
    core23.write_json_atomic(output_root / "metadata.json", metadata)
    print(
        f"openfield_core23_complete dataset={output_root} frames={len(frames)} views={len(frames)} visible_instances={visible_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add Core-23 masks, projections, and visibility to a rendered MoSeq dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import convert_cylinder_renders_to_blenderpymouse_core23 as core23  # noqa: E402


DEFAULT_CAMERA_NAMES = [f"Camera.{index:03d}" for index in range(1, 9)]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERA_NAMES),
        help="Comma-separated Blender camera object names in dataset view order.",
    )
    parser.add_argument("--armature", default="DeformationSystem.001")
    parser.add_argument("--mesh", default="MS_Mouse.001")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=200)
    parser.add_argument(
        "--source-frame-start",
        type=int,
        default=None,
        help="Override the source video frame index represented by Blender frame 1.",
    )
    parser.add_argument("--vertices-per-keypoint", type=int, default=16)
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.5)
    parser.add_argument("--depth-window-radius", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(args_after_dash())


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header {header!r}: {path}")
        width, height = (int(value) for value in handle.readline().decode("ascii").split())
        scale = float(handle.readline().decode("ascii"))
        endian = "<" if scale < 0 else ">"
        channels = 1 if header == "Pf" else 3
        data = np.fromfile(handle, dtype=endian + "f4")
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM size mismatch: {path}, {data.size}!={expected}")
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def rewrite_cam_file(
    path: Path,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> None:
    """Replace camera matrices while preserving the rendered depth range."""
    lines = path.read_text(encoding="utf-8").splitlines()
    intrinsic_start = lines.index("intrinsic")
    depth_line = next(
        line.strip() for line in lines[intrinsic_start + 4 :] if line.strip()
    )
    payload = ["extrinsic"]
    payload.extend(" ".join(f"{value:.10g}" for value in row) for row in extrinsic)
    payload.extend(["", "intrinsic"])
    payload.extend(" ".join(f"{value:.10g}" for value in row) for row in intrinsic)
    payload.extend(["", depth_line, ""])
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(payload), encoding="utf-8")
    temporary.replace(path)


def rasterized_mouse_mask(
    mesh_obj: bpy.types.Object,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    dilation: int,
) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        world = np.asarray(
            [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices],
            dtype=np.float64,
        )
        camera_xyz, uv = core23.project_points(world, extrinsic, intrinsic)
        polygons = []
        for triangle in mesh.loop_triangles:
            indices = np.asarray(triangle.vertices, dtype=np.int32)
            if not np.all(camera_xyz[indices, 2] > 0) or not np.isfinite(uv[indices]).all():
                continue
            points = np.rint(uv[indices]).astype(np.int32)
            if (
                points[:, 0].max() < 0
                or points[:, 1].max() < 0
                or points[:, 0].min() >= width
                or points[:, 1].min() >= height
            ):
                continue
            polygons.append(points)
    finally:
        evaluated.to_mesh_clear()
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    if dilation > 0:
        size = dilation * 2 + 1
        mask = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    return mask


def build_view(
    frame: int,
    view_id: int,
    camera_obj: bpy.types.Object,
    scene_root: Path,
    joint_world: np.ndarray,
    surface_world: np.ndarray,
    support_world: np.ndarray,
    mesh_obj: bpy.types.Object,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray], int]:
    stem = f"{view_id:08d}"
    image_path = scene_root / "blended_images" / f"{stem}.jpg"
    masked_path = scene_root / "blended_images" / f"{stem}_masked.jpg"
    mask_path = scene_root / "masks" / f"{stem}.png"
    depth_path = scene_root / "rendered_depth_maps" / f"{stem}.pfm"
    cam_path = scene_root / "cams" / f"{stem}_cam.txt"
    if not image_path.is_file() or not depth_path.is_file() or not cam_path.is_file():
        raise FileNotFoundError(f"Incomplete rendered view: {scene_root} view={view_id}")

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    depth = read_pfm(depth_path)
    if depth.shape != (height, width):
        raise ValueError(f"Depth/image shape mismatch: {depth.shape} vs {(height, width)}")
    intrinsic = core23.camera_intrinsic(camera_obj, width, height)
    extrinsic = core23.camera_extrinsic(camera_obj)
    rewrite_cam_file(cam_path, extrinsic, intrinsic)
    mask = rasterized_mouse_mask(
        mesh_obj, extrinsic, intrinsic, width, height, args.mask_dilation
    )
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_path)
    masked = Image.new("RGB", (width, height))
    masked.paste(rgb, mask=Image.fromarray(mask, mode="L"))
    masked.save(masked_path, quality=args.jpeg_quality, subsampling=0)

    joint_camera, joint_uv = core23.project_points(joint_world, extrinsic, intrinsic)
    surface_camera, _ = core23.project_points(surface_world, extrinsic, intrinsic)
    support_camera, support_uv = core23.project_points(support_world, extrinsic, intrinsic)
    support_visible, support_residual = core23.support_visibility(
        support_camera,
        support_uv,
        depth,
        mask > 0,
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
                "joint_world_xyz": core23.finite_list(joint_world[index]),
                "surface_world_xyz": core23.finite_list(surface_world[index]),
                "camera_xyz": core23.finite_list(joint_camera[index]),
                "surface_camera_xyz": core23.finite_list(surface_camera[index]),
                "uv": core23.finite_list(joint_uv[index]),
                "in_frame": bool(in_frame[index]),
                "visible": bool(surface_visible[index]),
                "surface_visible": bool(surface_visible[index]),
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
        "cam": relative(cam_path),
        "depth_available": True,
        "depth_excludes_transparent_glass": True,
        "depth_excludes_hair_curves": True,
        "intrinsic": intrinsic.tolist(),
        "extrinsic": extrinsic.tolist(),
        "keypoint_format": core23.KEYPOINT_FORMAT,
        "keypoints": keypoints,
    }
    core23.write_json_atomic(scene_root / "annotations" / f"{stem}.json", view)
    arrays = {
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "joint_camera": joint_camera,
        "uv": joint_uv,
        "in_frame": in_frame,
        "surface_visible": surface_visible,
        "support_visible_count": support_visible.sum(axis=1).astype(np.int16),
        "depth_available": np.asarray(True),
    }
    return view, arrays, int(surface_visible.sum())


def frame_complete(scene_root: Path, view_count: int) -> bool:
    return (
        (scene_root / "keypoints.json").is_file()
        and (scene_root / "keypoints.npz").is_file()
        and all(
            (scene_root / "masks" / f"{view_id:08d}.png").is_file()
            for view_id in range(view_count)
        )
    )


def existing_visible_count(scene_root: Path) -> int:
    payload = json.loads((scene_root / "keypoints.json").read_text(encoding="utf-8"))
    return sum(
        bool(keypoint.get("surface_visible", keypoint.get("visible", False)))
        for view in payload.get("views", [])
        for keypoint in view.get("keypoints", [])
    )


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    render_metadata_path = root / "metadata.json"
    render_metadata = (
        json.loads(render_metadata_path.read_text(encoding="utf-8"))
        if render_metadata_path.is_file()
        else {}
    )
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    if not camera_names:
        raise ValueError("--cameras must contain at least one camera name")
    armature = bpy.data.objects.get(args.armature)
    mesh = bpy.data.objects.get(args.mesh)
    if armature is None or armature.type != "ARMATURE":
        raise KeyError(args.armature)
    if mesh is None or mesh.type != "MESH":
        raise KeyError(args.mesh)
    cameras = [bpy.data.objects.get(name) for name in camera_names]
    if any(camera is None or camera.type != "CAMERA" for camera in cameras):
        raise KeyError(camera_names)

    support_ids, support_audit = core23.build_support_map(
        mesh, armature, args.vertices_per_keypoint
    )
    with (root / "core23_vertex_support_map.pkl").open("wb") as handle:
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
    core23.write_json_atomic(root / "core23_vertex_support_map.audit.json", support_audit)

    frames = list(range(args.frame_start, args.frame_end + 1))
    visible_total = 0
    processed = []
    for frame in frames:
        scene_root = root / f"frame_{frame:06d}"
        if args.resume and frame_complete(scene_root, len(cameras)):
            visible_total += existing_visible_count(scene_root)
            processed.append(frame)
            continue
        joint_world, surface_world, support_vertex_ids, support_world = core23.evaluated_core23(
            frame, armature, mesh, support_ids
        )
        views = []
        arrays = []
        for view_id, camera in enumerate(cameras):
            view, view_arrays, visible = build_view(
                frame,
                view_id,
                camera,
                scene_root,
                joint_world,
                surface_world,
                support_world,
                mesh,
                args,
            )
            views.append(view)
            arrays.append(view_arrays)
            visible_total += visible
        core23.write_frame_keypoints(
            scene_root,
            frame,
            joint_world,
            surface_world,
            support_vertex_ids,
            support_world,
            views,
            arrays,
            armature_name=armature.name,
            mesh_name=mesh.name,
        )
        core23.write_json_atomic(
            scene_root / "scene_meta.json",
            {"frame": frame, "keypoint_format": core23.KEYPOINT_FORMAT, "views": views},
        )
        processed.append(frame)
        if frame == frames[0] or frame == frames[-1] or frame % max(1, args.progress_every) == 0:
            print(f"core23_augment frame={frame}/{frames[-1]}", flush=True)

    source_stride = int(bpy.context.scene.get("MGF_source_stride", 1))
    source_start = (
        args.source_frame_start
        if args.source_frame_start is not None
        else int(bpy.context.scene.get("MGF_source_frame_start", 0))
    )
    width, height = Image.open(root / f"frame_{frames[0]:06d}" / "blended_images" / "00000000.jpg").size
    side_depth_wall_policy = bpy.context.scene.get("MGF_side_depth_wall_policy")
    depth_excluded_objects = render_metadata.get(
        "depth_ignore_objects",
        [
            "Part_GlassCylinder160mm",
            "Part_GlassCylinder160mm.001",
            "Part_AntiReflectiveGlass",
            "Part_AntiReflectiveGlass.001",
            "Part_AntiReflectiveGlass_Circle",
        ],
    )
    if side_depth_wall_policy == "ignore_only_wall_between_camera_and_mouse":
        depth_excluded_objects = [
            "MGF_MARS_HomecageShell camera-facing wall only (per side camera)",
            "CURVES hair geometry",
        ]

    metadata = {
        "format": core23.FORMAT_NAME,
        "dataset_name": root.name,
        "dataset_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "appearance_profile": {
            "homecage_material_profile": bpy.context.scene.get("MGF_homecage_material_profile"),
            "c57_ear_material_profile": bpy.context.scene.get("MGF_C57_ear_material_profile"),
            "ground_contact_correction_z_mm": bpy.context.scene.get(
                "MGF_ground_contact_correction_z_mm"
            ),
        },
        "frames": [frames[0], frames[-1]],
        "frame_numbers": processed,
        "scene_count": len(processed),
        "cameras": camera_names,
        "view_counts": {str(len(cameras)): len(processed)},
        "render_resolution": [width, height],
        "image_format": "JPEG",
        "mask_format": "PNG rasterized deformed mouse mesh with dilation",
        "depth_format": "PFM",
        "scene_unit_system": bpy.context.scene.unit_settings.system,
        "scene_length_unit": bpy.context.scene.unit_settings.length_unit,
        "blender_unit_scale_meters": float(bpy.context.scene.unit_settings.scale_length),
        "world_coordinate_units": "millimeter",
        "depth_units": "millimeter",
        "camera_extrinsics": "OpenCV world-to-camera rigid transform; inherited object scale removed",
        "masked_image_content": "rendered mouse foreground on black background",
        "depth_excludes_transparent_glass": True,
        "depth_excluded_objects": depth_excluded_objects,
        "depth_excludes_hair_curves": True,
        "side_depth_wall_policy": side_depth_wall_policy,
        "side_depth_proxy_version": bpy.context.scene.get(
            "MGF_side_depth_proxy_version"
        ),
        "side_depth_removed_source_faces": (
            {
                "HomecageCam_SideNegX": 0,
                "HomecageCam_SideNegY": 12,
                "HomecageCam_SidePosY": 1,
            }
            if side_depth_wall_policy == "ignore_only_wall_between_camera_and_mouse"
            else None
        ),
        "top_view_depth_wall_policy": (
            "full_shell_retained" if side_depth_wall_policy else None
        ),
        "source_video_sampling": {
            "source_fps": float(bpy.context.scene.get("MGF_source_fps", 30.0)),
            "source_stride": source_stride,
            "sampled_fps": float(bpy.context.scene.get("MGF_sampled_fps", 3.0)),
            "source_frame_indices": [
                source_start + (frame - 1) * source_stride for frame in processed
            ],
        },
        "keypoint_annotations": {
            "format": core23.KEYPOINT_FORMAT,
            "keypoint_count": len(core23.CORE23_SPECS),
            "names": [spec["name"] for spec in core23.CORE23_SPECS],
            "joint_coordinate_rule": "pose bone head only",
            "vertices_per_keypoint": args.vertices_per_keypoint,
            "visibility_rule": "at least one support vertex agrees with glass-free body depth",
            "visible_total": visible_total,
        },
        "layout": {
            "images": "frame_xxxxxx/blended_images/00000000.jpg",
            "masked_images": "frame_xxxxxx/blended_images/00000000_masked.jpg",
            "masks": "frame_xxxxxx/masks/00000000.png",
            "depths": "frame_xxxxxx/rendered_depth_maps/00000000.pfm",
            "cameras": "frame_xxxxxx/cams/00000000_cam.txt",
            "annotations": "frame_xxxxxx/annotations/00000000.json",
            "keypoints": "frame_xxxxxx/keypoints.npz",
        },
    }
    core23.write_json_atomic(root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

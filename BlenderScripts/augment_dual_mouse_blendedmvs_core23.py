#!/usr/bin/env python3
"""Add exact dual-instance masks and Core23 labels to a rendered dataset."""

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
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import augment_moseq_blendedmvs_core23 as single  # noqa: E402
import convert_cylinder_renders_to_blenderpymouse_core23 as core23  # noqa: E402


DEFAULT_CAMERAS = (
    "HomecageCam_Top",
    "HomecageCam_SideNegX",
    "HomecageCam_SideNegY",
    "HomecageCam_SidePosY",
)
INSTANCE_SPECS = (
    {"id": 0, "name": "black_mouse", "armature": "LP_Black_Armature", "mesh": "LP_Black_Mesh"},
    {"id": 1, "name": "brown_mouse", "armature": "LP_Brown_Armature", "mesh": "LP_Brown_Mesh"},
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=200)
    parser.add_argument("--vertices-per-keypoint", type=int, default=16)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.5)
    parser.add_argument("--depth-window-radius", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--source-plan", type=Path)
    return parser.parse_args(args_after_dash())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def emission_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


class InstanceMaskRenderer:
    """Render the two deformed body meshes as flat red/green IDs in one pass."""

    def __init__(self, scene: bpy.types.Scene, meshes: list[bpy.types.Object], root: Path):
        self.scene = scene
        self.meshes = meshes
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.renderable_types = {"MESH", "CURVE", "CURVES", "SURFACE", "META", "VOLUME", "FONT"}
        self.visibility = {
            obj.name: obj.hide_render for obj in bpy.data.objects if obj.type in self.renderable_types
        }
        self.materials = {obj.name: list(obj.data.materials) for obj in meshes}
        self.engine = scene.render.engine
        self.film_transparent = scene.render.film_transparent
        self.filepath = scene.render.filepath
        self.file_format = scene.render.image_settings.file_format
        self.color_mode = scene.render.image_settings.color_mode
        self.use_compositing = getattr(scene.render, "use_compositing", None)
        self.camera = scene.camera
        red = emission_material("MGF_InstanceID_Black_Red", (1.0, 0.0, 0.0, 1.0))
        green = emission_material("MGF_InstanceID_Brown_Green", (0.0, 1.0, 0.0, 1.0))
        for obj in bpy.data.objects:
            if obj.type in self.renderable_types:
                obj.hide_render = obj not in meshes
        for obj, material in zip(meshes, (red, green), strict=True):
            obj.hide_render = False
            obj.data.materials.clear()
            obj.data.materials.append(material)
        engine_items = {
            item.identifier
            for item in scene.render.bl_rna.properties["engine"].enum_items
        }
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT"
            if "BLENDER_EEVEE_NEXT" in engine_items
            else "BLENDER_EEVEE"
        )
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        if self.use_compositing is not None:
            scene.render.use_compositing = False

    def render(self, frame: int, view_id: int, camera: bpy.types.Object) -> list[np.ndarray]:
        path = self.root / f"frame_{frame:06d}_view_{view_id:02d}.png"
        self.scene.camera = camera
        self.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        alpha = rgba[..., 3] > 8
        red = alpha & (rgba[..., 0].astype(np.int16) > rgba[..., 1].astype(np.int16))
        green = alpha & (rgba[..., 1].astype(np.int16) > rgba[..., 0].astype(np.int16))
        return [red, green]

    def close(self) -> None:
        for obj in self.meshes:
            obj.data.materials.clear()
            for material in self.materials[obj.name]:
                obj.data.materials.append(material)
        for name, hidden in self.visibility.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.hide_render = hidden
        self.scene.render.engine = self.engine
        self.scene.render.film_transparent = self.film_transparent
        self.scene.render.filepath = self.filepath
        self.scene.render.image_settings.file_format = self.file_format
        self.scene.render.image_settings.color_mode = self.color_mode
        if self.use_compositing is not None:
            self.scene.render.use_compositing = self.use_compositing
        self.scene.camera = self.camera


def mask_box(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return {"valid": False, "xyxy_pixels": None, "cxcywh_normalized": None, "area_pixels": 0}
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    height, width = mask.shape
    return {
        "valid": True,
        "xyxy_pixels": [left, top, right, bottom],
        "cxcywh_normalized": [
            (left + right) / (2.0 * width),
            (top + bottom) / (2.0 * height),
            (right - left) / width,
            (bottom - top) / height,
        ],
        "area_pixels": int(mask.sum()),
    }


def keypoint_records(
    joint_world: np.ndarray,
    surface_world: np.ndarray,
    support_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, np.ndarray], int]:
    height, width = depth.shape
    joint_camera, joint_uv = core23.project_points(joint_world, extrinsic, intrinsic)
    surface_camera, _ = core23.project_points(surface_world, extrinsic, intrinsic)
    support_camera, support_uv = core23.project_points(support_world, extrinsic, intrinsic)
    support_visible, residual = core23.support_visibility(
        support_camera,
        support_uv,
        depth,
        mask,
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
    records = []
    for index, spec in enumerate(core23.CORE23_SPECS):
        residuals = residual[index][np.isfinite(residual[index])]
        records.append({
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
            "support_vertex_count": int(np.isfinite(support_world[index]).all(axis=1).sum()),
            "support_min_depth_residual": float(residuals.min()) if residuals.size else None,
        })
    arrays = {
        "joint_camera": joint_camera,
        "uv": joint_uv,
        "in_frame": in_frame,
        "surface_visible": surface_visible,
        "support_visible_count": support_visible.sum(axis=1).astype(np.int16),
    }
    return records, arrays, int(surface_visible.sum())


def frame_complete(root: Path, view_count: int) -> bool:
    return (
        (root / "keypoints.json").is_file()
        and (root / "keypoints.npz").is_file()
        and all(
            (root / "masks" / f"{view_id:08d}_instance{instance_id:02d}.png").is_file()
            for view_id in range(view_count)
            for instance_id in range(2)
        )
    )


def source_frame_indices(path: Path | None, frames: list[int]) -> list[int | None]:
    if path is None or not path.is_file():
        return [None] * len(frames)
    plan = json.loads(path.read_text(encoding="utf-8"))
    mapping = {int(row["scene_frame"]): int(row["source_image_frame"]) for row in plan["keyframes"]}
    return [mapping.get(frame) for frame in frames]


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    cameras = [bpy.data.objects.get(name) for name in camera_names]
    if any(camera is None or camera.type != "CAMERA" for camera in cameras):
        raise KeyError(camera_names)
    instances = []
    for spec in INSTANCE_SPECS:
        armature = bpy.data.objects.get(spec["armature"])
        mesh = bpy.data.objects.get(spec["mesh"])
        if armature is None or armature.type != "ARMATURE":
            raise KeyError(spec["armature"])
        if mesh is None or mesh.type != "MESH":
            raise KeyError(spec["mesh"])
        support_ids, audit = core23.build_support_map(mesh, armature, args.vertices_per_keypoint)
        instances.append({**spec, "armature_obj": armature, "mesh_obj": mesh, "support_ids": support_ids, "audit": audit})

    root.mkdir(parents=True, exist_ok=True)
    with (root / "core23_dual_instance_support_maps.pkl").open("wb") as handle:
        pickle.dump({
            "format": "blenderpymouse_dual_instance_core23_support_v1",
            "keypoint_format": core23.KEYPOINT_FORMAT,
            "specs": core23.CORE23_SPECS,
            "instances": [
                {"id": item["id"], "name": item["name"], "mesh": item["mesh"], "armature": item["armature"], "support_vertex_ids": item["support_ids"]}
                for item in instances
            ],
        }, handle, protocol=4)
    write_json(root / "core23_dual_instance_support_maps.audit.json", {
        "format": "blenderpymouse_dual_instance_core23_support_v1",
        "instances": [{"id": item["id"], "name": item["name"], "audit": item["audit"]} for item in instances],
    })

    frames = list(range(args.frame_start, args.frame_end + 1))
    mask_tmp = root / "_dual_instance_mask_tmp"
    if mask_tmp.exists():
        shutil.rmtree(mask_tmp)
    renderer = InstanceMaskRenderer(bpy.context.scene, [item["mesh_obj"] for item in instances], mask_tmp)
    visible_total = 0
    processed = []
    try:
        for frame in frames:
            scene_root = root / f"frame_{frame:06d}"
            if args.resume and frame_complete(scene_root, len(cameras)):
                with np.load(scene_root / "keypoints.npz", allow_pickle=False) as labels:
                    visible_total += int(np.asarray(labels["surface_visible"], dtype=bool).sum())
                processed.append(frame)
                continue
            bpy.context.scene.frame_set(frame)
            evaluated = []
            for item in instances:
                evaluated.append(core23.evaluated_core23(
                    frame, item["armature_obj"], item["mesh_obj"], item["support_ids"]
                ))
            views = []
            view_arrays = []
            for view_id, camera in enumerate(cameras):
                stem = f"{view_id:08d}"
                image_path = scene_root / "blended_images" / f"{stem}.jpg"
                depth_path = scene_root / "rendered_depth_maps" / f"{stem}.pfm"
                cam_path = scene_root / "cams" / f"{stem}_cam.txt"
                if not image_path.is_file() or not depth_path.is_file() or not cam_path.is_file():
                    raise FileNotFoundError(f"Incomplete render: frame={frame} view={view_id}")
                rgb = Image.open(image_path).convert("RGB")
                width, height = rgb.size
                depth = single.read_pfm(depth_path)
                intrinsic = core23.camera_intrinsic(camera, width, height)
                extrinsic = core23.camera_extrinsic(camera)
                single.rewrite_cam_file(cam_path, extrinsic, intrinsic)
                masks = renderer.render(frame, view_id, camera)
                mask_dir = scene_root / "masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                union = np.logical_or.reduce(masks)
                Image.fromarray((union * 255).astype(np.uint8), mode="L").save(mask_dir / f"{stem}.png")
                combined = Image.new("RGB", (width, height))
                combined.paste(rgb, mask=Image.fromarray((union * 255).astype(np.uint8), mode="L"))
                combined.save(scene_root / "blended_images" / f"{stem}_masked.jpg", quality=args.jpeg_quality, subsampling=0)
                instance_records = []
                instance_arrays = []
                for item, geometry, mask in zip(instances, evaluated, masks, strict=True):
                    joint_world, surface_world, support_ids, support_world = geometry
                    mask_path = mask_dir / f"{stem}_instance{item['id']:02d}.png"
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                    mask_image.save(mask_path)
                    masked_path = scene_root / "blended_images" / f"{stem}_instance{item['id']:02d}_masked.jpg"
                    masked = Image.new("RGB", (width, height))
                    masked.paste(rgb, mask=mask_image)
                    masked.save(masked_path, quality=args.jpeg_quality, subsampling=0)
                    keypoints, arrays, visible = keypoint_records(
                        joint_world, surface_world, support_world, intrinsic, extrinsic, depth, mask, args
                    )
                    visible_total += visible
                    instance_records.append({
                        "instance_id": item["id"],
                        "instance_name": item["name"],
                        "armature": item["armature"],
                        "mesh": item["mesh"],
                        "mask": str(mask_path.relative_to(scene_root)).replace("/", "\\"),
                        "masked_image": str(masked_path.relative_to(scene_root)).replace("/", "\\"),
                        "box": mask_box(mask),
                        "keypoints": keypoints,
                    })
                    instance_arrays.append(arrays)
                view = {
                    "frame": frame,
                    "view_id": view_id,
                    "camera": camera.name,
                    "image": str(image_path.relative_to(scene_root)).replace("/", "\\"),
                    "combined_masked_image": f"blended_images\\{stem}_masked.jpg",
                    "combined_mask": f"masks\\{stem}.png",
                    "depth_pfm": str(depth_path.relative_to(scene_root)).replace("/", "\\"),
                    "cam": str(cam_path.relative_to(scene_root)).replace("/", "\\"),
                    "intrinsic": intrinsic.tolist(),
                    "extrinsic": extrinsic.tolist(),
                    "depth_available": True,
                    "depth_units": "millimeter",
                    "depth_excludes_hair_curves": True,
                    "depth_wall_policy": "top full shell; side only nearest camera-facing wall ignored",
                    "instance_mask_rule": "visible deformed body mesh, exact mouse-mouse occlusion, hair excluded",
                    "instances": instance_records,
                }
                write_json(scene_root / "annotations" / f"{stem}.json", view)
                views.append(view)
                view_arrays.append({
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic,
                    "instances": instance_arrays,
                })

            names = [spec["name"] for spec in core23.CORE23_SPECS]
            keypoint_payload = {
                "format": "blenderpymouse_dual_instance_core23_v1",
                "keypoint_format": core23.KEYPOINT_FORMAT,
                "frame": frame,
                "source_blend": str(Path(bpy.data.filepath).resolve()),
                "coordinate_system": "Blender world; camera_xyz uses OpenCV +Z forward",
                "names": names,
                "groups": [spec["group"] for spec in core23.CORE23_SPECS],
                "bones": [spec["bone"] for spec in core23.CORE23_SPECS],
                "instances": [
                    {
                        "instance_id": item["id"],
                        "instance_name": item["name"],
                        "armature": item["armature"],
                        "mesh": item["mesh"],
                        "joint_world_xyz": evaluated[index][0].tolist(),
                        "surface_world_xyz": evaluated[index][1].tolist(),
                        "support_vertex_ids": evaluated[index][2].tolist(),
                    }
                    for index, item in enumerate(instances)
                ],
                "views": [
                    {"view_id": view["view_id"], "camera": view["camera"], "instances": view["instances"]}
                    for view in views
                ],
            }
            write_json(scene_root / "keypoints.json", keypoint_payload)
            temporary = scene_root / "keypoints.npz.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    format=np.asarray("blenderpymouse_dual_instance_core23_v1"),
                    keypoint_format=np.asarray(core23.KEYPOINT_FORMAT),
                    frame=np.asarray(frame, dtype=np.int32),
                    names=np.asarray(names),
                    groups=np.asarray([spec["group"] for spec in core23.CORE23_SPECS]),
                    bones=np.asarray([spec["bone"] for spec in core23.CORE23_SPECS]),
                    instance_ids=np.asarray([item["id"] for item in instances], dtype=np.int32),
                    instance_names=np.asarray([item["name"] for item in instances]),
                    armature_names=np.asarray([item["armature"] for item in instances]),
                    mesh_names=np.asarray([item["mesh"] for item in instances]),
                    view_ids=np.arange(len(views), dtype=np.int32),
                    camera_names=np.asarray([view["camera"] for view in views]),
                    intrinsics=np.stack([item["intrinsic"] for item in view_arrays]).astype(np.float32),
                    extrinsics=np.stack([item["extrinsic"] for item in view_arrays]).astype(np.float32),
                    joint_world_xyz=np.stack([item[0] for item in evaluated]).astype(np.float32),
                    surface_world_xyz=np.stack([item[1] for item in evaluated]).astype(np.float32),
                    support_vertex_ids=np.stack([item[2] for item in evaluated]),
                    support_world_xyz=np.stack([item[3] for item in evaluated]).astype(np.float32),
                    joint_camera_xyz=np.stack([[inst["joint_camera"] for inst in view["instances"]] for view in view_arrays]).astype(np.float32),
                    uv=np.stack([[inst["uv"] for inst in view["instances"]] for view in view_arrays]).astype(np.float32),
                    in_frame=np.stack([[inst["in_frame"] for inst in view["instances"]] for view in view_arrays]),
                    surface_visible=np.stack([[inst["surface_visible"] for inst in view["instances"]] for view in view_arrays]),
                    support_visible_count=np.stack([[inst["support_visible_count"] for inst in view["instances"]] for view in view_arrays]),
                    depth_available=np.ones(len(views), dtype=bool),
                )
            temporary.replace(scene_root / "keypoints.npz")
            write_json(scene_root / "scene_meta.json", {
                "frame": frame,
                "format": "blenderpymouse_dual_instance_core23_v1",
                "instance_count": 2,
                "views": views,
            })
            processed.append(frame)
            if frame == frames[0] or frame == frames[-1] or frame % max(1, args.progress_every) == 0:
                print(f"dual_core23_augment frame={frame}/{frames[-1]}", flush=True)
    finally:
        renderer.close()
        if mask_tmp.exists():
            shutil.rmtree(mask_tmp)

    source_indices = source_frame_indices(args.source_plan, processed)
    width, height = Image.open(root / f"frame_{processed[0]:06d}" / "blended_images" / "00000000.jpg").size
    metadata = {
        "format": "blenderpymouse_dual_instance_core23_v1",
        "dataset_name": root.name,
        "dataset_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "frames": [processed[0], processed[-1]],
        "frame_numbers": processed,
        "scene_count": len(processed),
        "instance_count": 2,
        "instances": [{key: item[key] for key in ("id", "name", "armature", "mesh")} for item in instances],
        "cameras": camera_names,
        "view_counts": {str(len(cameras)): len(processed)},
        "render_resolution": [width, height],
        "image_format": "JPEG",
        "mask_format": "PNG exact visible deformed body instance mask",
        "depth_format": "PFM",
        "world_coordinate_units": "millimeter",
        "depth_units": "millimeter",
        "camera_model": "perspective",
        "depth_excludes_hair_curves": True,
        "side_depth_wall_policy": "ignore_only_wall_between_camera_and_mouse",
        "side_depth_removed_source_faces": {
            "HomecageCam_SideNegX": 0,
            "HomecageCam_SideNegY": 12,
            "HomecageCam_SidePosY": 1,
        },
        "top_view_depth_wall_policy": "full_shell_retained",
        "source_pose_plan": str(args.source_plan.resolve()) if args.source_plan else None,
        "source_frame_indices": source_indices,
        "keypoint_annotations": {
            "format": core23.KEYPOINT_FORMAT,
            "keypoint_count_per_instance": len(core23.CORE23_SPECS),
            "names": [spec["name"] for spec in core23.CORE23_SPECS],
            "visible_keypoint_instances": visible_total,
            "array_layout": "view, instance, keypoint, coordinate",
        },
        "layout": {
            "images": "frame_xxxxxx/blended_images/00000000.jpg",
            "instance_masks": "frame_xxxxxx/masks/00000000_instance00.png",
            "depths": "frame_xxxxxx/rendered_depth_maps/00000000.pfm",
            "cameras": "frame_xxxxxx/cams/00000000_cam.txt",
            "annotations": "frame_xxxxxx/annotations/00000000.json",
            "keypoints": "frame_xxxxxx/keypoints.npz",
        },
    }
    write_json(root / "metadata.json", metadata)
    print(json.dumps({"status": "complete", "frames": len(processed), "visible": visible_total}, indent=2), flush=True)


if __name__ == "__main__":
    main()

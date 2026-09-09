#!/usr/bin/env python3
"""Backfill Blender rig keypoints into an existing BlenderPyMouse export.

The script must be run by Blender with the source animation project loaded:

    blender source.blend --background --python annotate_existing_blenderpymouse_keypoints.py -- \
      --dataset-root /path/to/dataset/BlenderPyMouse/source-name \
      --vertex-keypoint-map /path/to/mouse_hand_vertex_keypoint_map.pkl \
      --vertex-keypoint-audit /path/to/mouse_hand_vertex_keypoint_map.audit.json

Existing RGB, depth, and camera files are left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np


FORMAT_NAME = "mousegeoforge_blender_rig_keypoints_v1"
DEFAULT_ANCHOR_FRACTIONS = {"HeadEnd_M": 0.27}
SEMANTIC_NAMES = {
    "Fingers1_L": "left_palm",
    "Fingers1_R": "right_palm",
    "Wrist_L": "left_elbow",
    "Wrist_R": "right_elbow",
    "HeadEnd_M": "nose_tip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--vertex-keypoint-map", required=True)
    parser.add_argument("--vertex-keypoint-audit", default="")
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-window-radius", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--disable-hair-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide Blender CURVES hair objects while evaluating animation frames.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def resolve_path(text: str) -> Path:
    return Path(bpy.path.abspath(text)).resolve()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def dataset_relative_path(scene_root: Path, text: str) -> Path:
    return scene_root / Path(text.replace("\\", "/"))


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header {header!r} in {path}")
        dimensions = handle.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = handle.readline().decode("ascii").strip()
        width, height = (int(value) for value in dimensions.split())
        scale = float(handle.readline().decode("ascii").strip())
        channels = 3 if header == "PF" else 1
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.fromfile(handle, dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM payload has {data.size} values, expected {expected}: {path}")
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32, copy=False)


def load_landmark_configuration(
    map_path: Path,
    audit_path: Path | None,
) -> tuple[list[str], dict[str, list[int]], dict[str, float], dict | None]:
    with map_path.open("rb") as handle:
        vertex_to_keypoint = pickle.load(handle)
    if not isinstance(vertex_to_keypoint, dict):
        raise TypeError(f"Expected dict[int, str] in {map_path}")

    support_ids: dict[str, list[int]] = defaultdict(list)
    for vertex_id, name in vertex_to_keypoint.items():
        support_ids[str(name)].append(int(vertex_id))
    for values in support_ids.values():
        values.sort()

    audit = read_json(audit_path) if audit_path is not None and audit_path.exists() else None
    if audit is not None:
        names = [str(name) for name in audit.get("landmarks", []) if str(name) in support_ids]
        fractions = {
            name: float(audit.get("per_landmark", {}).get(name, {}).get("anchor_fraction", 0.0))
            for name in names
        }
    else:
        names = sorted(support_ids)
        fractions = {name: DEFAULT_ANCHOR_FRACTIONS.get(name, 0.0) for name in names}
    return names, dict(support_ids), fractions, audit


def semantic_name(name: str) -> str:
    if name in SEMANTIC_NAMES:
        return SEMANTIC_NAMES[name]
    side = "left" if name.endswith("_L") else "right" if name.endswith("_R") else "midline"
    return f"{side}_{name.rsplit('_', 1)[0].lower()}"


def disable_hair_evaluation() -> tuple[int, int]:
    object_count = 0
    modifier_count = 0
    for obj in bpy.data.objects:
        if obj.type != "CURVES":
            continue
        object_count += 1
        obj.hide_viewport = True
        obj.hide_render = True
        try:
            obj.hide_set(True)
        except RuntimeError:
            pass
        for modifier in obj.modifiers:
            if modifier.show_viewport:
                modifier.show_viewport = False
                modifier_count += 1
    bpy.context.view_layer.update()
    return object_count, modifier_count


def matrix_to_numpy(matrix) -> np.ndarray:
    return np.asarray([[float(value) for value in row] for row in matrix], dtype=np.float64)


def evaluated_landmarks(
    scene: bpy.types.Scene,
    frame: int,
    armature_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    names: list[str],
    support_ids: dict[str, list[int]],
    fractions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    joint_world = np.full((len(names), 3), np.nan, dtype=np.float64)
    max_support = max(len(support_ids[name]) for name in names)
    support_vertex_ids = np.full((len(names), max_support), -1, dtype=np.int32)
    support_world = np.full((len(names), max_support, 3), np.nan, dtype=np.float64)

    armature_world = armature_obj.matrix_world
    for index, name in enumerate(names):
        pose_bone = armature_obj.pose.bones.get(name)
        if pose_bone is None:
            raise KeyError(f"Pose bone {name!r} not found in {armature_obj.name!r}")
        fraction = fractions.get(name, 0.0)
        point = pose_bone.head.lerp(pose_bone.tail, fraction)
        joint_world[index] = np.asarray(armature_world @ point, dtype=np.float64)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    temp_mesh = evaluated.to_mesh()
    try:
        if max(max(values) for values in support_ids.values()) >= len(temp_mesh.vertices):
            raise IndexError(
                f"Vertex map exceeds evaluated mesh vertex count {len(temp_mesh.vertices)}"
            )
        mesh_world = evaluated.matrix_world
        for keypoint_index, name in enumerate(names):
            ids = support_ids[name]
            support_vertex_ids[keypoint_index, : len(ids)] = ids
            for support_index, vertex_id in enumerate(ids):
                support_world[keypoint_index, support_index] = np.asarray(
                    mesh_world @ temp_mesh.vertices[vertex_id].co,
                    dtype=np.float64,
                )
    finally:
        evaluated.to_mesh_clear()

    surface_world = np.nanmean(support_world, axis=1)
    return joint_world, surface_world, support_vertex_ids, support_world


def project_points(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = points_world.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    camera = np.full_like(flat, np.nan, dtype=np.float64)
    camera[valid] = (extrinsic[:3, :3] @ flat[valid].T).T + extrinsic[:3, 3]
    uv = np.full((len(flat), 2), np.nan, dtype=np.float64)
    in_front = valid & (camera[:, 2] > 1e-9)
    projected = (intrinsic @ camera[in_front].T).T
    uv[in_front] = projected[:, :2] / projected[:, 2:3]
    return camera.reshape(points_world.shape), uv.reshape(*points_world.shape[:-1], 2)


def support_visibility(
    support_camera: np.ndarray,
    support_uv: np.ndarray,
    depth: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
    window_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape[:2]
    visible = np.zeros(support_camera.shape[:-1], dtype=bool)
    residual = np.full(support_camera.shape[:-1], np.nan, dtype=np.float64)
    for index in np.ndindex(visible.shape):
        z = float(support_camera[index][2])
        u, v = (float(value) for value in support_uv[index])
        if not (math.isfinite(z) and z > 0 and math.isfinite(u) and math.isfinite(v)):
            continue
        x = int(round(u))
        y = int(round(v))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        x0, x1 = max(0, x - window_radius), min(width, x + window_radius + 1)
        y0, y1 = max(0, y - window_radius), min(height, y + window_radius + 1)
        values = depth[y0:y1, x0:x1]
        values = values[np.isfinite(values) & (values > 1e-8)]
        if not values.size:
            continue
        differences = np.abs(values.astype(np.float64) - z)
        minimum = float(differences.min())
        residual[index] = minimum
        tolerance = max(absolute_tolerance, relative_tolerance * abs(z))
        visible[index] = minimum <= tolerance
    return visible, residual


def finite_list(values: np.ndarray) -> list[float] | None:
    if not np.isfinite(values).all():
        return None
    return [float(value) for value in values]


def annotation_for_view(
    names: list[str],
    fractions: dict[str, float],
    joint_world: np.ndarray,
    surface_world: np.ndarray,
    support_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    joint_camera, joint_uv = project_points(joint_world, extrinsic, intrinsic)
    support_camera, support_uv = project_points(support_world, extrinsic, intrinsic)
    support_visible, support_residual = support_visibility(
        support_camera,
        support_uv,
        depth,
        args.depth_relative_tolerance,
        args.depth_absolute_tolerance,
        args.depth_window_radius,
    )
    height, width = depth.shape[:2]
    in_frame = (
        np.isfinite(joint_uv).all(axis=1)
        & (joint_camera[:, 2] > 0)
        & (joint_uv[:, 0] >= 0)
        & (joint_uv[:, 0] < width)
        & (joint_uv[:, 1] >= 0)
        & (joint_uv[:, 1] < height)
    )

    annotations = []
    for index, name in enumerate(names):
        valid_support = np.isfinite(support_world[index]).all(axis=1)
        visible_count = int((support_visible[index] & valid_support).sum())
        residual_values = support_residual[index][np.isfinite(support_residual[index])]
        annotations.append(
            {
                "name": name,
                "semantic_name": semantic_name(name),
                "bone": name,
                "anchor_fraction": float(fractions.get(name, 0.0)),
                "joint_world_xyz": finite_list(joint_world[index]),
                "surface_world_xyz": finite_list(surface_world[index]),
                "camera_xyz": finite_list(joint_camera[index]),
                "uv": finite_list(joint_uv[index]),
                "x": float(joint_uv[index, 0]) if np.isfinite(joint_uv[index, 0]) else None,
                "y": float(joint_uv[index, 1]) if np.isfinite(joint_uv[index, 1]) else None,
                "camera_depth": (
                    float(joint_camera[index, 2])
                    if np.isfinite(joint_camera[index, 2])
                    else None
                ),
                "in_frame": bool(in_frame[index]),
                "visible": bool(visible_count > 0),
                "support_visible_count": visible_count,
                "support_vertex_count": int(valid_support.sum()),
                "support_min_depth_residual": (
                    float(residual_values.min()) if residual_values.size else None
                ),
            }
        )
    arrays = {
        "joint_camera_xyz": joint_camera,
        "joint_uv": joint_uv,
        "in_frame": in_frame,
        "surface_visible": support_visible.any(axis=1),
        "support_visible_count": support_visible.sum(axis=1).astype(np.int16),
    }
    return annotations, arrays


def frame_number(path: Path) -> int:
    return int(path.name.removeprefix("frame_"))


def selected_scene_roots(dataset_root: Path, args: argparse.Namespace) -> list[Path]:
    roots = sorted(
        (path for path in dataset_root.glob("frame_*") if path.is_dir()),
        key=frame_number,
    )
    if args.frame_start > 0:
        roots = [path for path in roots if frame_number(path) >= args.frame_start]
    if args.frame_end > 0:
        roots = [path for path in roots if frame_number(path) <= args.frame_end]
    return roots


def annotate_scene(
    scene_root: Path,
    frame: int,
    names: list[str],
    support_ids: dict[str, list[int]],
    fractions: dict[str, float],
    armature_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    args: argparse.Namespace,
) -> dict:
    output_npz = scene_root / "keypoints.npz"
    output_json = scene_root / "keypoints.json"
    if output_npz.exists() and output_json.exists() and not args.overwrite:
        return {"status": "skipped", "frame": frame}

    scene_meta_path = scene_root / "scene_meta.json"
    scene_meta = read_json(scene_meta_path)
    views = scene_meta.get("views", [])
    if not views:
        raise ValueError(f"No views in {scene_meta_path}")

    joint_world, surface_world, support_vertex_ids, support_world = evaluated_landmarks(
        bpy.context.scene,
        frame,
        armature_obj,
        mesh_obj,
        names,
        support_ids,
        fractions,
    )

    view_ids = []
    camera_names = []
    intrinsics = []
    extrinsics = []
    joint_camera_by_view = []
    joint_uv_by_view = []
    in_frame_by_view = []
    visible_by_view = []
    visible_count_by_view = []
    view_annotations = []
    visible_total = 0

    for view in views:
        view_id = int(view["view_id"])
        annotation_path = scene_root / "annotations" / f"{view_id:08d}.json"
        annotation = read_json(annotation_path)
        intrinsic = np.asarray(view.get("intrinsic", annotation["intrinsic"]), dtype=np.float64)
        extrinsic = np.asarray(view.get("extrinsic", annotation["extrinsic"]), dtype=np.float64)
        depth_path = dataset_relative_path(scene_root, view["depth_pfm"])
        depth = read_pfm(depth_path)

        keypoints, arrays = annotation_for_view(
            names,
            fractions,
            joint_world,
            surface_world,
            support_world,
            extrinsic,
            intrinsic,
            depth,
            args,
        )
        annotation["keypoint_format"] = FORMAT_NAME
        annotation["keypoints"] = keypoints
        write_json_atomic(annotation_path, annotation)

        view["keypoint_format"] = FORMAT_NAME
        view["keypoints"] = keypoints
        view_ids.append(view_id)
        camera_names.append(str(view["camera"]))
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
        joint_camera_by_view.append(arrays["joint_camera_xyz"])
        joint_uv_by_view.append(arrays["joint_uv"])
        in_frame_by_view.append(arrays["in_frame"])
        visible_by_view.append(arrays["surface_visible"])
        visible_count_by_view.append(arrays["support_visible_count"])
        visible_total += int(arrays["surface_visible"].sum())
        view_annotations.append(
            {
                "view_id": view_id,
                "camera": str(view["camera"]),
                "keypoints": keypoints,
            }
        )

    scene_payload = {
        "format": FORMAT_NAME,
        "frame": frame,
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "armature": armature_obj.name,
        "mesh": mesh_obj.name,
        "coordinate_system": "Blender world coordinates; per-view camera_xyz uses OpenCV +Z forward",
        "names": names,
        "semantic_names": [semantic_name(name) for name in names],
        "anchor_fractions": [float(fractions.get(name, 0.0)) for name in names],
        "joint_world_xyz": joint_world.tolist(),
        "surface_world_xyz": surface_world.tolist(),
        "support_vertex_ids": support_vertex_ids.tolist(),
        "views": view_annotations,
    }
    write_json_atomic(output_json, scene_payload)

    temporary_npz = output_npz.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            format=np.asarray(FORMAT_NAME),
            frame=np.asarray(frame, dtype=np.int32),
            names=np.asarray(names),
            semantic_names=np.asarray([semantic_name(name) for name in names]),
            anchor_fractions=np.asarray([fractions.get(name, 0.0) for name in names], dtype=np.float32),
            view_ids=np.asarray(view_ids, dtype=np.int32),
            camera_names=np.asarray(camera_names),
            intrinsics=np.stack(intrinsics).astype(np.float32),
            extrinsics=np.stack(extrinsics).astype(np.float32),
            joint_world_xyz=joint_world.astype(np.float32),
            surface_world_xyz=surface_world.astype(np.float32),
            support_vertex_ids=support_vertex_ids,
            support_world_xyz=support_world.astype(np.float32),
            joint_camera_xyz=np.stack(joint_camera_by_view).astype(np.float32),
            uv=np.stack(joint_uv_by_view).astype(np.float32),
            in_frame=np.stack(in_frame_by_view),
            surface_visible=np.stack(visible_by_view),
            support_visible_count=np.stack(visible_count_by_view),
        )
    temporary_npz.replace(output_npz)

    scene_meta["keypoint_format"] = FORMAT_NAME
    scene_meta["keypoint_names"] = names
    scene_meta["views"] = views
    write_json_atomic(scene_meta_path, scene_meta)
    return {
        "status": "annotated",
        "frame": frame,
        "view_count": len(views),
        "keypoint_count": len(names),
        "visible_count": visible_total,
    }


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    map_path = resolve_path(args.vertex_keypoint_map)
    audit_path = resolve_path(args.vertex_keypoint_audit) if args.vertex_keypoint_audit else None
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    armature_obj = bpy.data.objects.get(args.armature)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise KeyError(f"Armature {args.armature!r} not found")
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(f"Mesh {args.mesh!r} not found")

    names, support_ids, fractions, audit = load_landmark_configuration(map_path, audit_path)
    if not names:
        raise ValueError("No keypoints found in vertex map")
    if audit is not None:
        expected_vertices = audit.get("topology", {}).get("vertex_count")
        if expected_vertices is not None and int(expected_vertices) != len(mesh_obj.data.vertices):
            raise ValueError(
                f"Vertex topology mismatch: map expects {expected_vertices}, "
                f"{mesh_obj.name} has {len(mesh_obj.data.vertices)}"
            )

    if args.disable_hair_evaluation:
        hair_objects, hair_modifiers = disable_hair_evaluation()
        print(
            "keypoint_annotation_hair_disabled",
            f"objects={hair_objects}",
            f"modifiers={hair_modifiers}",
            flush=True,
        )

    roots = selected_scene_roots(dataset_root, args)
    if not roots:
        raise ValueError(f"No selected frame_* directories under {dataset_root}")

    counts = defaultdict(int)
    visible_total = 0
    view_total = 0
    for index, scene_root in enumerate(roots, start=1):
        frame = frame_number(scene_root)
        result = annotate_scene(
            scene_root,
            frame,
            names,
            support_ids,
            fractions,
            armature_obj,
            mesh_obj,
            args,
        )
        counts[result["status"]] += 1
        visible_total += int(result.get("visible_count", 0))
        view_total += int(result.get("view_count", 0))
        if index == 1 or index == len(roots) or index % max(1, args.progress_every) == 0:
            print(
                "keypoint_progress",
                f"dataset={dataset_root.name}",
                f"processed={index}/{len(roots)}",
                f"frame={frame}",
                f"status={result['status']}",
                flush=True,
            )

    complete_scene_count = 0
    complete_view_count = 0
    complete_visible_count = 0
    for scene_root in roots:
        npz_path = scene_root / "keypoints.npz"
        json_path = scene_root / "keypoints.json"
        if not (npz_path.is_file() and json_path.is_file()):
            continue
        with np.load(npz_path, allow_pickle=False) as arrays:
            complete_scene_count += 1
            complete_view_count += int(arrays["view_ids"].shape[0])
            complete_visible_count += int(arrays["surface_visible"].sum())

    metadata_path = dataset_root / "metadata.json"
    metadata = read_json(metadata_path)
    metadata["keypoint_annotations"] = {
        "format": FORMAT_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "vertex_keypoint_map": str(map_path),
        "vertex_keypoint_audit": str(audit_path) if audit_path is not None else None,
        "keypoint_count": len(names),
        "names": names,
        "semantic_names": [semantic_name(name) for name in names],
        "selected_scene_count": len(roots),
        "complete_scene_count": complete_scene_count,
        "complete_view_count": complete_view_count,
        "complete_visible_keypoint_instances": complete_visible_count,
        "last_run_annotated_scene_count": counts["annotated"],
        "last_run_skipped_scene_count": counts["skipped"],
        "annotated_scene_count": complete_scene_count,
        "skipped_scene_count": len(roots) - complete_scene_count,
        "annotated_view_count": complete_view_count,
        "visible_keypoint_instances": complete_visible_count,
        "depth_relative_tolerance": args.depth_relative_tolerance,
        "depth_absolute_tolerance": args.depth_absolute_tolerance,
        "depth_window_radius": args.depth_window_radius,
    }
    write_json_atomic(metadata_path, metadata)
    print(
        "keypoint_annotation_complete",
        f"dataset={dataset_root}",
        f"annotated={counts['annotated']}",
        f"skipped={counts['skipped']}",
        f"views={view_total}",
        f"keypoints={len(names)}",
        f"visible_instances={visible_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()

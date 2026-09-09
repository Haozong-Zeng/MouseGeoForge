"""Render V19 with cameras reconstructed from the fixed real rig calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


CAMERAS = ("A", "B", "C")
CV_FROM_BLENDER_CAMERA = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
BUILTIN_CAMERA_MAPPING = {"A": "Camera.003", "B": "Camera.001", "C": "Camera.002"}
MODEL_BONE_LANDMARKS = {
    "snout_A": ("HeadEnd_M", "tail", "face"),
    "mouth_tip_A": ("JawEnd_M", "tail", "face"),
    "l_mid_A": ("Fingers1_L", "head", "palm"),
    "r_mid_A": ("Fingers1_R", "head", "palm"),
}
for _side, _suffix in (("l", "L"), ("r", "R")):
    for _digit, _finger in enumerate(("Index", "Middle", "Ring", "Pinky"), start=1):
        MODEL_BONE_LANDMARKS[f"{_side}{_digit}_joint_A"] = (
            f"{_finger}Finger1_{_suffix}",
            "head",
            "finger",
        )
        MODEL_BONE_LANDMARKS[f"{_side}{_digit}_mid_A"] = (
            f"{_finger}Finger2_{_suffix}",
            "head",
            "finger",
        )
        MODEL_BONE_LANDMARKS[f"{_side}{_digit}_tip_A"] = (
            f"{_finger}Finger3_{_suffix}",
            "tail",
            "finger",
        )
STATIC_RIG_LANDMARKS = {
    "spout_tip_A": ("Part_WaterPipe", 124, "nearest modeled spout-tip proxy"),
}


def blender_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--fixed-extrinsics", type=Path, required=True)
    parser.add_argument(
        "--rig-world-alignment",
        type=Path,
        help="Optional Anipose-to-Blender Sim(3) and calibration extrinsics JSON.",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated Blender frames")
    parser.add_argument("--cameras", default="A,B,C", help="Comma-separated subset of A,B,C")
    parser.add_argument("--frame-offset", type=int, default=1)
    parser.add_argument(
        "--camera-pose-mode",
        choices=("calibrated", "builtin"),
        default="calibrated",
        help="Use transformed real calibration poses or V19 built-in camera poses.",
    )
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution-percentage", type=int, default=100)
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args(blender_args())


def load_calibration(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    result = {}
    for value in raw.values():
        if not isinstance(value, dict) or str(value.get("name")) not in CAMERAS:
            continue
        name = str(value["name"])
        result[name] = {
            "size": [int(v) for v in value["size"]],
            "matrix": [[float(v) for v in row] for row in value["matrix"]],
        }
    if set(result) != set(CAMERAS):
        raise KeyError(f"Calibration is missing cameras: {sorted(set(CAMERAS) - set(result))}")
    return result


def load_fixed_extrinsics(path: Path) -> dict[str, list[list[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = data.get("fixed_relative_extrinsics")
    if fixed is None:
        fixed = data.get("camera_stabilization", {}).get("fixed_relative_extrinsics")
    if not fixed:
        raise KeyError(f"No fixed_relative_extrinsics in {path}")
    return fixed


def as_matrix3(values: list[list[float]]) -> Matrix:
    return Matrix(tuple(tuple(float(v) for v in row[:3]) for row in values[:3]))


def as_vector3(values: list[float]) -> Vector:
    return Vector(tuple(float(v) for v in values[:3]))


def configure_camera(
    scene: bpy.types.Scene,
    camera_name: str,
    intrinsic: list[list[float]],
    size: list[int],
    source_extrinsic: list[list[float]],
    scale: float,
    alignment_rotation: Matrix,
    alignment_translation: Vector,
) -> bpy.types.Object:
    width, height = size
    fx = float(intrinsic[0][0])
    fy = float(intrinsic[1][1])
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    source_rotation = as_matrix3(source_extrinsic)
    source_translation = as_vector3([row[3] for row in source_extrinsic[:3]])
    source_center = -(source_rotation.transposed() @ source_translation)
    blender_center = scale * (alignment_rotation @ source_center) + alignment_translation
    world_rotation = alignment_rotation @ source_rotation.transposed() @ CV_FROM_BLENDER_CAMERA

    data_name = f"MGF_PoseMatchedCam_{camera_name}_Data"
    object_name = f"MGF_PoseMatchedCam_{camera_name}"
    old_object = bpy.data.objects.get(object_name)
    if old_object is not None:
        bpy.data.objects.remove(old_object, do_unlink=True)
    old_data = bpy.data.cameras.get(data_name)
    if old_data is not None:
        bpy.data.cameras.remove(old_data)
    camera_data = bpy.data.cameras.new(data_name)
    camera = bpy.data.objects.new(object_name, camera_data)
    scene.collection.objects.link(camera)

    matrix_world = world_rotation.to_4x4()
    matrix_world.translation = blender_center
    camera.matrix_world = matrix_world
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = fx * camera_data.sensor_width / width
    pixel_ratio = fx / fy
    camera_data.shift_x = (width / 2.0 - cx) / width
    camera_data.shift_y = (cy - height / 2.0) * pixel_ratio / width
    camera_data.clip_start = 0.01
    camera_data.clip_end = 10000.0
    camera["mgf_source_camera"] = camera_name
    camera["mgf_expected_fx"] = fx
    camera["mgf_expected_fy"] = fy
    return camera


def collection_objects_recursive(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    objects = list(collection.objects)
    for child in collection.children:
        objects.extend(collection_objects_recursive(child))
    return objects


def hide_for_render() -> list[tuple[object, bool]]:
    states: list[tuple[object, bool]] = []
    visited: set[int] = set()
    for name in ("CamA", "CamB", "CamC", "MGF_VGGTDLC_Keypoints"):
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        pointer = collection.as_pointer()
        if pointer not in visited:
            states.append((collection, collection.hide_render))
            collection.hide_render = True
            visited.add(pointer)
        for obj in collection_objects_recursive(collection):
            pointer = obj.as_pointer()
            if pointer in visited:
                continue
            states.append((obj, obj.hide_render))
            obj.hide_render = True
            visited.add(pointer)
    physical_camera_prefixes = (
        "CameraItems.",
        "Part_CameraIO.",
        "Part_CameraLens.",
        "Part_CameraMount.",
        "Part_CameraPCB.",
        "Part_CamLen.",
    )
    for obj in bpy.data.objects:
        if not (
            obj.name.startswith("CTRL_")
            or obj.name.startswith("MGF_Hand")
            or obj.name.startswith(physical_camera_prefixes)
        ):
            continue
        pointer = obj.as_pointer()
        if pointer in visited:
            continue
        states.append((obj, obj.hide_render))
        obj.hide_render = True
        visited.add(pointer)
    return states


def restore_visibility(states: list[tuple[object, bool]]) -> None:
    for datablock, hidden in reversed(states):
        datablock.hide_render = hidden


def enable_cycles_gpu(scene: bpy.types.Scene) -> list[str]:
    used = []
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for compute_type in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                break
            except Exception:
                continue
        for device in prefs.devices:
            device.use = device.type != "CPU"
            if device.use:
                used.append(f"{device.type}:{device.name}")
        scene.cycles.device = "GPU" if used else "CPU"
    except Exception as exc:
        print(f"warning: could not configure Cycles GPU: {exc}")
    return used


def marker_projection_rows(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    camera_name: str,
    source_frame: int,
    blender_frame: int,
    width: int,
    height: int,
) -> list[dict]:
    rows = []
    for obj in bpy.data.objects:
        if not obj.name.startswith("DLC3D_"):
            continue
        ndc = world_to_camera_view(scene, camera, obj.matrix_world.translation)
        rows.append(
            {
                "camera": camera_name,
                "source_frame": source_frame,
                "blender_frame": blender_frame,
                "bodypart": obj.name.removeprefix("DLC3D_"),
                "x_synthetic": float(ndc.x * width),
                "y_synthetic": float((1.0 - ndc.y) * height),
                "camera_depth": float(ndc.z),
                "in_frame": int(ndc.z > 0.0 and 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0),
            }
        )
    return rows


def project_world_point(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    world: Vector,
    width: int,
    height: int,
) -> tuple[float, float, float, int]:
    ndc = world_to_camera_view(scene, camera, world)
    return (
        float(ndc.x * width),
        float((1.0 - ndc.y) * height),
        float(ndc.z),
        int(ndc.z > 0.0 and 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0),
    )


def model_bone_projection_rows(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    camera_name: str,
    source_frame: int,
    blender_frame: int,
    width: int,
    height: int,
) -> list[dict]:
    armature = bpy.data.objects.get("DeformationSystem")
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("Missing DeformationSystem armature")
    rows = []
    for bodypart, (bone_name, endpoint, category) in MODEL_BONE_LANDMARKS.items():
        bone = armature.pose.bones.get(bone_name)
        if bone is None:
            raise KeyError(f"Missing model landmark bone {bone_name}")
        local = bone.head if endpoint == "head" else bone.tail
        world = armature.matrix_world @ local
        x, y, depth, in_frame = project_world_point(
            scene, camera, world, width, height
        )
        rows.append(
            {
                "camera": camera_name,
                "source_frame": source_frame,
                "blender_frame": blender_frame,
                "bodypart": bodypart,
                "category": category,
                "bone": bone_name,
                "endpoint": endpoint,
                "world_x": float(world.x),
                "world_y": float(world.y),
                "world_z": float(world.z),
                "x_model": x,
                "y_model": y,
                "camera_depth": depth,
                "in_frame": in_frame,
            }
        )
    return rows


def static_rig_projection_rows(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    camera_name: str,
    source_frame: int,
    blender_frame: int,
    width: int,
    height: int,
) -> list[dict]:
    rows = []
    for bodypart, (object_name, vertex_index, note) in STATIC_RIG_LANDMARKS.items():
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "MESH":
            raise KeyError(f"Missing static rig mesh {object_name}")
        if vertex_index >= len(obj.data.vertices):
            raise IndexError(f"{object_name} lacks vertex {vertex_index}")
        world = obj.matrix_world @ obj.data.vertices[vertex_index].co
        x, y, depth, in_frame = project_world_point(
            scene, camera, world, width, height
        )
        rows.append(
            {
                "camera": camera_name,
                "source_frame": source_frame,
                "blender_frame": blender_frame,
                "bodypart": bodypart,
                "object": object_name,
                "vertex_index": vertex_index,
                "note": note,
                "world_x": float(world.x),
                "world_y": float(world.y),
                "world_z": float(world.z),
                "x_static": x,
                "y_static": y,
                "camera_depth": depth,
                "in_frame": in_frame,
            }
        )
    return rows


def camera_pose_comparison(cameras: dict[str, bpy.types.Object]) -> list[dict]:
    rows = []
    for name, temporary in cameras.items():
        builtin_name = BUILTIN_CAMERA_MAPPING[name]
        builtin = bpy.data.objects.get(builtin_name)
        if builtin is None or builtin.type != "CAMERA":
            continue
        temporary_rotation = temporary.matrix_world.to_3x3().normalized()
        builtin_rotation = builtin.matrix_world.to_3x3().normalized()
        delta = temporary_rotation @ builtin_rotation.transposed()
        trace = sum(delta[index][index] for index in range(3))
        cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        rows.append(
            {
                "camera": name,
                "builtin_camera": builtin_name,
                "temporary_center": [float(v) for v in temporary.matrix_world.translation],
                "builtin_center": [float(v) for v in builtin.matrix_world.translation],
                "center_delta": float(
                    (temporary.matrix_world.translation - builtin.matrix_world.translation).length
                ),
                "rotation_delta_deg": float(math.degrees(math.acos(cosine))),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted({int(value.strip()) for value in args.frames.split(",") if value.strip()})
    camera_names = tuple(
        value.strip().upper() for value in args.cameras.split(",") if value.strip()
    )
    if not camera_names or not set(camera_names).issubset(CAMERAS):
        raise ValueError(f"--cameras must be a subset of {CAMERAS}: {camera_names}")
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    fixed = load_fixed_extrinsics(args.fixed_extrinsics)
    calibration = load_calibration(args.calibration)
    scale = float(alignment["scale"])
    rotation = as_matrix3(alignment["rotation"])
    translation = as_vector3(alignment["translation"])
    camera_coordinate_system = "vggt_fixed_relative"
    if args.rig_world_alignment:
        rig_world = json.loads(args.rig_world_alignment.read_text(encoding="utf-8"))
        transform = rig_world.get("source_to_blender", rig_world.get("anipose_to_blender"))
        if transform is None:
            raise KeyError("rig-world-alignment lacks source_to_blender")
        scale = float(transform["scale"])
        rotation = as_matrix3(transform["rotation"])
        translation = as_vector3(transform["translation"])
        fixed = rig_world["calibration_extrinsics"]
        camera_coordinate_system = "anipose_calibration"

    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_percentage = args.resolution_percentage
    scene.render.pixel_aspect_x = 1.0
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        devices = enable_cycles_gpu(scene)
    else:
        devices = []

    cameras = {}
    for name in CAMERAS:
        width, height = calibration[name]["size"]
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        intrinsic = calibration[name]["matrix"]
        scene.render.pixel_aspect_y = float(intrinsic[0][0]) / float(intrinsic[1][1])
        cameras[name] = configure_camera(
            scene,
            name,
            intrinsic,
            calibration[name]["size"],
            fixed[name],
            scale,
            rotation,
            translation,
        )
        if args.camera_pose_mode == "builtin":
            builtin = bpy.data.objects.get(BUILTIN_CAMERA_MAPPING[name])
            if builtin is None or builtin.type != "CAMERA":
                raise KeyError(f"Missing built-in camera for {name}")
            cameras[name].matrix_world = builtin.matrix_world.copy()

    visibility_states = hide_for_render()
    projection_rows: list[dict] = []
    model_projection_rows: list[dict] = []
    static_projection_rows: list[dict] = []
    try:
        for blender_frame in frames:
            source_frame = blender_frame - args.frame_offset
            scene.frame_set(blender_frame)
            for name in camera_names:
                camera = cameras[name]
                width, height = calibration[name]["size"]
                intrinsic = calibration[name]["matrix"]
                scene.render.resolution_x = width
                scene.render.resolution_y = height
                scene.render.pixel_aspect_y = float(intrinsic[0][0]) / float(intrinsic[1][1])
                scene.camera = camera
                camera_dir = output_dir / "synthetic" / f"cam{name}"
                camera_dir.mkdir(parents=True, exist_ok=True)
                filename = f"frame_{source_frame:06d}.png"
                scene.render.filepath = str(camera_dir / filename)
                if not args.skip_render:
                    bpy.ops.render.render(write_still=True)
                projection_rows.extend(
                    marker_projection_rows(
                        scene,
                        camera,
                        name,
                        source_frame,
                        blender_frame,
                        width,
                        height,
                    )
                )
                model_projection_rows.extend(
                    model_bone_projection_rows(
                        scene,
                        camera,
                        name,
                        source_frame,
                        blender_frame,
                        width,
                        height,
                    )
                )
                static_projection_rows.extend(
                    static_rig_projection_rows(
                        scene,
                        camera,
                        name,
                        source_frame,
                        blender_frame,
                        width,
                        height,
                    )
                )
                print(f"rendered cam{name} source={source_frame} blender={blender_frame}")
    finally:
        restore_visibility(visibility_states)

    projection_path = output_dir / "blender_keypoint_projection.csv"
    with projection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projection_rows[0]))
        writer.writeheader()
        writer.writerows(projection_rows)
    model_projection_path = output_dir / "model_bone_projection.csv"
    with model_projection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_projection_rows[0]))
        writer.writeheader()
        writer.writerows(model_projection_rows)
    static_projection_path = output_dir / "static_rig_projection.csv"
    with static_projection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(static_projection_rows[0]))
        writer.writeheader()
        writer.writerows(static_projection_rows)
    manifest = {
        "source_blend": bpy.data.filepath,
        "alignment": str(args.alignment.resolve()),
        "fixed_extrinsics": str(args.fixed_extrinsics.resolve()),
        "rig_world_alignment": str(args.rig_world_alignment.resolve()) if args.rig_world_alignment else None,
        "calibration": str(args.calibration.resolve()),
        "blender_frames": frames,
        "source_frames": [frame - args.frame_offset for frame in frames],
        "camera_names": list(camera_names),
        "engine": args.engine,
        "samples": args.samples,
        "skip_render": args.skip_render,
        "cycles_devices": devices,
        "temporary_camera_formula": "C_b=s*R_align*C_src+t; R_world=R_align*R_src.T*diag(1,-1,-1)",
        "camera_model_collections_hidden": ["CamA", "CamB", "CamC"],
        "physical_camera_prefixes_hidden": [
            "CameraItems.",
            "Part_CameraIO.",
            "Part_CameraLens.",
            "Part_CameraMount.",
            "Part_CameraPCB.",
            "Part_CamLen.",
        ],
        "camera_coordinate_system": camera_coordinate_system,
        "camera_pose_mode": args.camera_pose_mode,
        "camera_pose_comparison": camera_pose_comparison(cameras),
        "model_bone_landmarks": {
            name: {"bone": value[0], "endpoint": value[1], "category": value[2]}
            for name, value in MODEL_BONE_LANDMARKS.items()
        },
        "static_rig_landmarks": {
            name: {"object": value[0], "vertex_index": value[1], "note": value[2]}
            for name, value in STATIC_RIG_LANDMARKS.items()
        },
        "saved_blend": False,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

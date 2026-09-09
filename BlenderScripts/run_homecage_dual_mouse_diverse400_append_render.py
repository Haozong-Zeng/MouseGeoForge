#!/usr/bin/env python3
"""Append-render frames 201-400 for the 400-frame CRIM13 Homecage dataset."""

from __future__ import annotations

import json
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import bpy


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
PROJECT_NAME = "Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1"
SOURCE_BLEND = ROOT / "BlenderPyMouseProjects" / "Homecage" / f"{PROJECT_NAME}.blend"
DATASET = ROOT / "dataset" / "BlenderPyMouse" / PROJECT_NAME
RENDERER = ROOT / "BlenderScripts" / "blender_water_reaching_rendering.py"
STATUS = DATASET / "pipeline_status.json"
CAMERAS = [
    "HomecageCam_Top",
    "HomecageCam_SideNegX",
    "HomecageCam_SideNegY",
    "HomecageCam_SidePosY",
]
SIDE_COLLECTIONS = {
    "HomecageCam_SideNegX": "MGF_DepthProxy_SideNegX",
    "HomecageCam_SideNegY": "MGF_DepthProxy_SideNegY",
    "HomecageCam_SidePosY": "MGF_DepthProxy_SidePosY",
}


def write_status(stage: str, **extra: object) -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": PROJECT_NAME,
        "source_blend": str(SOURCE_BLEND),
        "dataset_root": str(DATASET),
        "frame_range": [1, 400],
        "append_render_range": [201, 400],
        "view_count": 4,
        "cameras": CAMERAS,
        "resolution": [1280, 720],
        "rgb_engine": "CYCLES",
        "cycles_samples": 64,
        "depth_wall_policy": "side: ignore only nearest camera-facing wall; top: retain full shell",
        "pose_or_animation_edits": False,
        **extra,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def preflight() -> None:
    opened = Path(bpy.data.filepath).resolve()
    if opened != SOURCE_BLEND.resolve():
        raise ValueError(f"Expected {SOURCE_BLEND}, opened {opened}")
    scene = bpy.context.scene
    if (scene.frame_start, scene.frame_end) != (1, 400):
        raise ValueError(f"Expected frame range 1..400, found {scene.frame_start}..{scene.frame_end}")
    for name in CAMERAS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "CAMERA":
            raise KeyError(f"Missing camera {name!r}")
        if obj.data.type != "PERSP":
            raise ValueError(f"Dataset camera {name!r} must remain perspective")
    shell = bpy.data.objects.get("MGF_MARS_HomecageShell")
    if shell is None or "MGF_Homecage_Shell" not in {c.name for c in shell.users_collection}:
        raise KeyError("Missing source Homecage shell collection")
    expected_faces = {
        "HomecageCam_SideNegX": 0,
        "HomecageCam_SideNegY": 12,
        "HomecageCam_SidePosY": 1,
    }
    for camera, collection_name in SIDE_COLLECTIONS.items():
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            raise KeyError(collection_name)
        proxies = [obj for obj in collection.objects if obj.get("mgf_depth_proxy")]
        if len(proxies) != 1:
            raise RuntimeError(f"Expected one proxy in {collection_name}, found {len(proxies)}")
        proxy = proxies[0]
        if proxy.get("mgf_camera") != camera:
            raise ValueError(f"Proxy/camera mismatch in {collection_name}")
        if int(proxy.get("mgf_removed_source_face_index", -1)) != expected_faces[camera]:
            raise ValueError(f"Wrong removed source face in {collection_name}")
    for name, object_type in (
        ("LP_Black_Armature", "ARMATURE"),
        ("LP_Black_Mesh", "MESH"),
        ("LP_Brown_Armature", "ARMATURE"),
        ("LP_Brown_Mesh", "MESH"),
    ):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != object_type:
            raise KeyError(name)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100


def main() -> None:
    preflight()
    camera_names = ",".join(CAMERAS)
    side_ignore = ";".join(f"{name}=MGF_Homecage_Shell" for name in SIDE_COLLECTIONS)
    side_show = ";".join(f"{name}={collection}" for name, collection in SIDE_COLLECTIONS.items())
    write_status("rendering_rgb_depth", completed_prefix=[1, 200])
    sys.argv = [
        "blender",
        "--",
        "--output-root", str(DATASET),
        "--cameras", camera_names,
        "--camera-model-collections=",
        "--no-hide-physical-camera-prefixes",
        "--per-camera-hide-collections=",
        "--depth-ignore-collections=",
        "--per-camera-depth-ignore-collections", side_ignore,
        "--per-camera-depth-show-collections", side_show,
        "--depth-ignore-mode", "hide-only",
        "--depth-exclude-hair-curves",
        "--start-frame", "201",
        "--end-frame", "400",
        "--frame-step", "1",
        "--resolution-x", "1280",
        "--resolution-y", "720",
        "--image-format", "JPEG",
        "--jpeg-quality", "95",
        "--engine", "CYCLES",
        "--cycles-samples", "64",
        "--save-depth-preview",
        "--no-keep-depth-exr",
        "--mouse-model", "LP_Black_Mesh",
        "--armature", "LP_Black_Armature",
        "--vertex-keypoint-map=",
        "--resume",
    ]
    runpy.run_path(str(RENDERER), run_name="__main__")
    write_status("rgb_depth_complete", next_stage="dual_instance_core23_augmentation")
    print("DUAL_HOMECAGE_RGB_DEPTH_APPEND_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status("failed", traceback=traceback.format_exc())
        raise

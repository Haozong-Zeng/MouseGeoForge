#!/usr/bin/env python3
"""Render the Homecage black-fur animation and augment it with Core-23."""

from __future__ import annotations

import json
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import bpy


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
PROJECT_NAME = "Homecage-BlackFur-11_01_22_12wk_m0-Core23-Sample10-Frames200-399"
SOURCE_BLEND = ROOT / "dataset" / "Homecage" / f"{PROJECT_NAME}.blend"
DATASET = ROOT / "dataset" / "BlenderPyMouse" / PROJECT_NAME
RENDERER = ROOT / "BlenderScripts" / "blender_water_reaching_rendering.py"
AUGMENTER = ROOT / "BlenderScripts" / "augment_moseq_blendedmvs_core23.py"
STATUS = DATASET / "pipeline_status.json"
CAMERAS = [
    "HomecageCam_Top",
    "HomecageCam_SideNegX",
    "HomecageCam_SideNegY",
    "HomecageCam_SidePosY",
]


def write_status(stage: str, **extra: object) -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def run_script(path: Path, arguments: list[str]) -> None:
    sys.argv = ["blender", "--", *arguments]
    runpy.run_path(str(path), run_name="__main__")


def prepare_scene() -> None:
    if Path(bpy.data.filepath).resolve() != SOURCE_BLEND.resolve():
        raise ValueError(f"Expected {SOURCE_BLEND}, opened {bpy.data.filepath}")
    scene = bpy.context.scene
    missing = [name for name in CAMERAS if bpy.data.objects.get(name) is None]
    if missing:
        raise KeyError(f"Missing Homecage cameras: {missing}")
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene["mgf_dataset_render_preset"] = "homecage_blackfur_core23_4view_1280x720_v1"
    scene["mgf_world_coordinate_unit"] = "millimeter"
    scene["MGF_source_frame_start"] = 2000
    bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE_BLEND))


def main() -> None:
    prepare_scene()
    camera_names = ",".join(CAMERAS)
    write_status(
        "rendering",
        frame_range=[1, 200],
        view_count=len(CAMERAS),
        cameras=CAMERAS,
        resolution=[1280, 720],
        source_blend=str(SOURCE_BLEND),
    )
    run_script(
        RENDERER,
        [
            "--output-root", str(DATASET),
            "--cameras", camera_names,
            "--camera-model-collections=",
            "--no-hide-physical-camera-prefixes",
            "--per-camera-hide-collections=",
            "--depth-ignore-collections=",
            "--per-camera-depth-ignore-collections",
            (
                "HomecageCam_SideNegX=MGF_Homecage_Shell;"
                "HomecageCam_SideNegY=MGF_Homecage_Shell;"
                "HomecageCam_SidePosY=MGF_Homecage_Shell"
            ),
            "--per-camera-depth-show-collections",
            (
                "HomecageCam_SideNegX=MGF_DepthProxy_SideNegX;"
                "HomecageCam_SideNegY=MGF_DepthProxy_SideNegY;"
                "HomecageCam_SidePosY=MGF_DepthProxy_SidePosY"
            ),
            "--depth-ignore-mode", "hide-only",
            "--depth-exclude-hair-curves",
            "--start-frame", "1",
            "--end-frame", "200",
            "--frame-step", "1",
            "--resolution-x", "1280",
            "--resolution-y", "720",
            "--image-format", "JPEG",
            "--jpeg-quality", "95",
            "--engine", "CYCLES",
            "--cycles-samples", "64",
            "--save-depth-preview",
            "--no-keep-depth-exr",
            "--mouse-model", "MS_Mouse.001",
            "--armature", "DeformationSystem.001",
            "--vertex-keypoint-map=",
            "--resume",
        ],
    )
    write_status("augmenting_core23", frame_range=[1, 200], view_count=len(CAMERAS))
    run_script(
        AUGMENTER,
        [
            "--dataset-root", str(DATASET),
            "--cameras", camera_names,
            "--armature", "DeformationSystem.001",
            "--mesh", "MS_Mouse.001",
            "--frame-start", "1",
            "--frame-end", "200",
            "--source-frame-start", "2000",
            "--no-resume",
        ],
    )
    write_status("complete", frame_range=[1, 200], view_count=len(CAMERAS))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status("failed", traceback=traceback.format_exc())
        raise

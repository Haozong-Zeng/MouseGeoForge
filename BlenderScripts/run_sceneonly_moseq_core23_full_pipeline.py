#!/usr/bin/env python3
"""Render the current SceneOnly MoSeq animation and augment it with Core-23."""

from __future__ import annotations

import json
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
SOURCE_BLEND = ROOT / "dataset" / "Cylinder" / (
    "PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend"
)
DATASET = ROOT / "dataset" / "BlenderPyMouse" / (
    "PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200"
)
PREPARE = ROOT / "BlenderScripts" / "prepare_sceneonly_moseq_core23_render.py"
RENDERER = ROOT / "BlenderScripts" / "blender_water_reaching_rendering.py"
AUGMENTER = ROOT / "BlenderScripts" / "augment_moseq_blendedmvs_core23.py"
STATUS = DATASET / "pipeline_status.json"
CAMERAS = [f"Camera.{index:03d}" for index in range(12, 20)]


def write_status(stage: str, **extra) -> None:
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


def main() -> None:
    camera_names = ",".join(CAMERAS)
    run_script(PREPARE, [])
    write_status(
        "rendering",
        frame_range=[1, 200],
        view_count=len(CAMERAS),
        cameras=CAMERAS,
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
            "--depth-ignore-collections", "MGF_DepthIgnoreTransparentGlass",
            "--depth-ignore-mode", "hide-only",
            "--depth-exclude-hair-curves",
            "--start-frame", "1",
            "--end-frame", "200",
            "--frame-step", "1",
            "--resolution-x", "1024",
            "--resolution-y", "1024",
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

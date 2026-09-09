#!/usr/bin/env python3
"""Run the complete BrownFur MoSeq render and Core-23 augmentation pipeline."""

from __future__ import annotations

import json
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
DATASET = ROOT / "dataset" / "BlenderPyMouse" / (
    "PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200-ClimbWallAppearance"
)
RENDERER = ROOT / "BlenderScripts" / "blender_water_reaching_rendering.py"
AUGMENTER = ROOT / "BlenderScripts" / "augment_moseq_blendedmvs_core23.py"
STATUS = DATASET / "pipeline_status.json"


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
    camera_names = ",".join(f"Camera.{index:03d}" for index in range(1, 9))
    rig_mapping = ";".join(
        (
            "Camera.001=MGF_Rig_Camera.001_Horizontal_Front",
            "Camera.002=MGF_Rig_Camera.002_Horizontal_BackLeft",
            "Camera.003=MGF_Rig_Camera.003_Horizontal_BackRight",
            "Camera.004=MGF_Rig_Camera.004_Horizontal_Back",
            "Camera.005=MGF_Rig_Camera.005_Horizontal_FrontRight",
            "Camera.006=MGF_Rig_Camera.006_Horizontal_FrontLeft",
            "Camera.007=MGF_Rig_Camera.007_Bottom",
        )
    )
    write_status("rendering", frame_range=[1, 200], view_count=8)
    run_script(
        RENDERER,
        [
            "--output-root", str(DATASET),
            "--cameras", camera_names,
            "--camera-model-collections=",
            "--no-hide-physical-camera-prefixes",
            "--per-camera-hide-collections", rig_mapping,
            "--depth-ignore-collections", "MGF_DepthIgnoreTransparentGlass",
            "--depth-ignore-mode", "hide-only",
            "--depth-exclude-hair-curves",
            "--start-frame", "1",
            "--end-frame", "200",
            "--resolution-x", "1024",
            "--resolution-y", "1024",
            "--image-format", "JPEG",
            "--jpeg-quality", "95",
            "--engine", "CYCLES",
            "--save-depth-preview",
            "--no-keep-depth-exr",
            "--vertex-keypoint-map=",
            "--resume",
        ],
    )
    write_status("augmenting_core23", frame_range=[1, 200], view_count=8)
    run_script(
        AUGMENTER,
        [
            "--dataset-root", str(DATASET),
            "--frame-start", "1",
            "--frame-end", "200",
            "--no-resume",
        ],
    )
    write_status("complete", frame_range=[1, 200], view_count=8)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status("failed", traceback=traceback.format_exc())
        raise

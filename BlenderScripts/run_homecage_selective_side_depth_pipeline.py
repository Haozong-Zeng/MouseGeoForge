#!/usr/bin/env python3
"""Refresh side depth only, then rebuild Core-23 annotations."""

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
REFRESH = ROOT / "BlenderScripts" / "refresh_homecage_selective_side_depth.py"
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
        "operation": "selective_side_depth_refresh",
        **extra,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def run_script(path: Path, arguments: list[str]) -> None:
    sys.argv = ["blender", "--", *arguments]
    runpy.run_path(str(path), run_name="__main__")


def main() -> None:
    if Path(bpy.data.filepath).resolve() != SOURCE_BLEND.resolve():
        raise ValueError(f"Expected {SOURCE_BLEND}, opened {bpy.data.filepath}")
    write_status(
        "refreshing_side_depth",
        frame_range=[1, 200],
        side_view_ids=[1, 2, 3],
        top_view_depth="unchanged",
        depth_wall_policy="ignore_only_wall_between_camera_and_mouse",
    )
    run_script(
        REFRESH,
        [
            "--dataset-root", str(DATASET),
            "--frame-start", "1",
            "--frame-end", "200",
            "--engine", "BLENDER_EEVEE_NEXT",
            "--cycles-samples", "1",
            "--commit",
        ],
    )
    write_status(
        "augmenting_core23",
        frame_range=[1, 200],
        side_view_ids=[1, 2, 3],
        top_view_depth="unchanged",
    )
    run_script(
        AUGMENTER,
        [
            "--dataset-root", str(DATASET),
            "--cameras", ",".join(CAMERAS),
            "--armature", "DeformationSystem.001",
            "--mesh", "MS_Mouse.001",
            "--frame-start", "1",
            "--frame-end", "200",
            "--source-frame-start", "2000",
            "--no-resume",
        ],
    )
    write_status(
        "complete",
        frame_range=[1, 200],
        side_view_ids=[1, 2, 3],
        top_view_depth="unchanged",
        depth_wall_policy="ignore_only_wall_between_camera_and_mouse",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status("failed", traceback=traceback.format_exc())
        raise

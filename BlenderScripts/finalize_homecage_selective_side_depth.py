#!/usr/bin/env python3
"""Rebuild Core-23 after selective side-depth refresh and finalize metadata."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import bpy


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
PROJECT_NAME = "Homecage-BlackFur-11_01_22_12wk_m0-Core23-Sample10-Frames200-399"
SOURCE_BLEND = ROOT / "dataset" / "Homecage" / f"{PROJECT_NAME}.blend"
DATASET = ROOT / "dataset" / "BlenderPyMouse" / PROJECT_NAME
AUGMENTER = ROOT / "BlenderScripts" / "augment_moseq_blendedmvs_core23.py"
STATUS = DATASET / "pipeline_status.json"
DEPTH_STATUS = DATASET / "selective_side_depth_status.json"
TOP_DEPTH_STATUS = DATASET / "top_full_shell_depth_status.json"
CAMERAS = [
    "HomecageCam_Top",
    "HomecageCam_SideNegX",
    "HomecageCam_SideNegY",
    "HomecageCam_SidePosY",
]


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_status(stage: str, **extra: object) -> None:
    write_json_atomic(
        STATUS,
        {
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "selective_side_and_full_shell_top_depth_refresh",
            **extra,
        },
    )


def cleanup_interrupted_temp() -> int:
    temporary = (DATASET / "_selective_depth_tmp").resolve()
    dataset = DATASET.resolve()
    if not temporary.exists():
        return 0
    if dataset not in temporary.parents or temporary.name != "_selective_depth_tmp":
        raise ValueError(f"Refuse to remove unexpected temporary path: {temporary}")
    count = sum(1 for path in temporary.rglob("*") if path.is_file())
    shutil.rmtree(temporary)
    return count


def run_augmenter() -> None:
    sys.argv = [
        "blender",
        "--",
        "--dataset-root", str(DATASET),
        "--cameras", ",".join(CAMERAS),
        "--armature", "DeformationSystem.001",
        "--mesh", "MS_Mouse.001",
        "--frame-start", "1",
        "--frame-end", "200",
        "--source-frame-start", "2000",
        "--no-resume",
    ]
    runpy.run_path(str(AUGMENTER), run_name="__main__")


def validate_depth_outputs() -> None:
    missing = []
    for frame in range(1, 201):
        frame_root = DATASET / f"frame_{frame:06d}"
        for view_id in range(0, 4):
            stem = f"{view_id:08d}"
            for relative in (
                Path("rendered_depth_maps") / f"{stem}.pfm",
                Path("rendered_depth_maps") / f"{stem}_preview.png",
                Path("cams") / f"{stem}_cam.txt",
            ):
                path = frame_root / relative
                if not path.is_file() or path.stat().st_size <= 0:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing/empty selective depth outputs: {missing[:10]}")


def main() -> None:
    if Path(bpy.data.filepath).resolve() != SOURCE_BLEND.resolve():
        raise ValueError(f"Expected {SOURCE_BLEND}, opened {bpy.data.filepath}")
    validate_depth_outputs()
    removed_temp_files = cleanup_interrupted_temp()
    write_status(
        "augmenting_core23",
        frame_range=[1, 200],
        side_view_ids=[1, 2, 3],
        top_view_depth="full_shell_retained",
        removed_interrupted_temp_files=removed_temp_files,
    )
    run_augmenter()
    validate_depth_outputs()
    completed_at = datetime.now(timezone.utc).isoformat()
    write_json_atomic(
        DEPTH_STATUS,
        {
            "status": "complete",
            "completed_at_utc": completed_at,
            "frame_range": [1, 200],
            "side_view_ids": [1, 2, 3],
            "side_depth_file_triplets": 600,
            "resolution": [1280, 720],
            "depth_render_engine": "BLENDER_EEVEE",
            "rgb_render_engine": "CYCLES (unchanged)",
            "depth_wall_policy": "ignore_only_wall_between_camera_and_mouse",
            "top_view_depth": "full_shell_retained",
            "side_camera_removed_source_faces": {
                "HomecageCam_SideNegX": 0,
                "HomecageCam_SideNegY": 12,
                "HomecageCam_SidePosY": 1,
            },
            "depth_excludes_hair_curves": True,
            "pose_or_animation_edits": False,
            "removed_interrupted_temp_files": removed_temp_files,
        },
    )
    write_json_atomic(
        TOP_DEPTH_STATUS,
        {
            "status": "complete",
            "completed_at_utc": completed_at,
            "frame_range": [1, 200],
            "view_id": 0,
            "camera": "HomecageCam_Top",
            "top_depth_file_triplets": 200,
            "resolution": [1280, 720],
            "depth_render_engine": "BLENDER_EEVEE",
            "rgb_render_engine": "CYCLES (unchanged)",
            "top_depth_wall_policy": "full_shell_retained",
            "shell_depth_material": "MGF_DepthProxy_Opaque (temporary override)",
            "depth_excludes_hair_curves": True,
            "side_view_depth": "unchanged from selective near-wall refresh",
            "pose_or_animation_edits": False,
        },
    )
    write_status(
        "complete",
        frame_range=[1, 200],
        side_view_ids=[1, 2, 3],
        side_depth_file_triplets=600,
        top_view_depth="full_shell_retained",
        top_depth_file_triplets=200,
        depth_wall_policy="ignore_only_wall_between_camera_and_mouse",
        pose_or_animation_edits=False,
    )
    print("SELECTIVE_SIDE_DEPTH_DATASET_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status("failed", traceback=traceback.format_exc())
        raise

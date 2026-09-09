#!/usr/bin/env python3
"""Build one distributable Blender project for every Anipose 3D trial."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_VIEWS = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "water-reaching-dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "BlenderPyMouseProjects" / "PseudoSynthetic",
    )
    parser.add_argument(
        "--base-blend",
        type=Path,
        default=(
            project_root
            / "backup"
            / "blender"
            / "PyMouseStudio-BrowmFur-WaterReaching-IK-BilateralHand-Control-"
            "HairGN-Experiment.blend"
        ),
    )
    parser.add_argument(
        "--rig-inspection",
        type=Path,
        default=project_root / "output" / "bilateral_hand_control" / "rig_inspection.json",
    )
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(
            "/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
        ),
    )
    parser.add_argument("--trial", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-copy-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def windows_path(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        return drive + ":\\" + text[7:].replace("/", "\\")
    if text.startswith("/"):
        return "\\\\wsl.localhost\\Ubuntu" + text.replace("/", "\\")
    return text


def discover_trials(dataset_root: Path) -> list[dict[str, object]]:
    trials = []
    for trial_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        pose = trial_dir / "pose-3d" / "pawfilt.csv"
        calibration = trial_dir / "calibration" / "calibration.toml"
        videos = {
            view: trial_dir / "videos-raw" / f"pawfilt_cam{view}.mp4"
            for view in REQUIRED_VIEWS
        }
        if pose.is_file() and calibration.is_file() and all(
            path.is_file() for path in videos.values()
        ):
            trials.append(
                {
                    "name": trial_dir.name,
                    "directory": trial_dir,
                    "pose": pose,
                    "calibration": calibration,
                    "videos": videos,
                }
            )
    return trials


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + "\n".join(command), encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND\n" + "\n".join(command) + "\n\nOUTPUT\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def require_outputs(paths: tuple[Path, ...], log_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Blender did not create expected outputs "
            f"{missing}; inspect {log_path} for a Python exception"
        )


def copy_sources(trial: dict[str, object], destination: Path) -> None:
    source_dir = destination / "source"
    video_dir = source_dir / "videos-raw"
    video_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trial["pose"], source_dir / "pawfilt.csv")
    shutil.copy2(trial["calibration"], source_dir / "calibration.toml")
    for view, path in trial["videos"].items():
        shutil.copy2(path, video_dir / f"pawfilt_cam{view}.mp4")


def stage_command(
    blender: Path,
    blend: Path,
    script: Path,
    arguments: list[str],
) -> list[str]:
    return [
        str(blender),
        "--background",
        windows_path(blend),
        "--python",
        windows_path(script),
        "--",
        *arguments,
    ]


def build_trial(
    args: argparse.Namespace,
    project_root: Path,
    trial: dict[str, object],
) -> dict[str, object]:
    name = str(trial["name"])
    destination = args.output_root / name
    alignment_dir = destination / "alignment"
    logs_dir = destination / "logs"
    stage1 = destination / "01_hand_drive.blend"
    stage2 = destination / "02_reach_clamp.blend"
    final = destination / f"{name}_AniposeSyntheticReady.blend"
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = destination / "project_manifest.json"
    manifest = {
        "trial": name,
        "status": "running",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "base_blend": str(args.base_blend.resolve()),
        "source_trial": str(trial["directory"].resolve()),
        "final_blend": str(final.resolve()),
        "finger_pose_mode": "normalized_anipose",
    }
    write_json(manifest_path, manifest)

    try:
        if not args.no_copy_sources and (args.overwrite or not (destination / "source").exists()):
            if not args.dry_run:
                copy_sources(trial, destination)

        aligned = alignment_dir / "aligned_keypoints_blender.csv"
        descriptors = alignment_dir / "hand_descriptors.csv"
        alignment_report = alignment_dir / "alignment.json"
        if args.overwrite or not all(
            path.is_file() for path in (aligned, descriptors, alignment_report)
        ):
            run_logged(
                [
                    sys.executable,
                    str(project_root / "tools" / "prepare_anipose_blender_hand_alignment.py"),
                    "--rig-inspection",
                    str(args.rig_inspection),
                    "--anipose-pose",
                    str(trial["pose"]),
                    "--calibration",
                    str(trial["calibration"]),
                    "--output-dir",
                    str(alignment_dir),
                ],
                logs_dir / "00_alignment.log",
                args.dry_run,
            )

        stage1_report = destination / "01_hand_drive.json"
        if args.overwrite or not stage1.is_file() or not stage1_report.is_file():
            stage1_log = logs_dir / "01_hand_drive.log"
            run_logged(
                stage_command(
                    args.blender,
                    args.base_blend,
                    project_root / "tools" / "drive_bilateral_hand_controllers_from_vggt_dlc.py",
                    [
                        "--descriptors", windows_path(descriptors),
                        "--keypoints", windows_path(aligned),
                        "--output", windows_path(stage1),
                        "--report", windows_path(stage1_report),
                        "--palm-orientation-mode", "forearm-roll",
                        "--elbow-pole-mode", "rest-prior",
                        "--compact-fingers",
                        "--drive-normalized-fingers",
                        "--principal-prebind-palm-branch",
                        "--palm-roll-median-radius", "4",
                        "--finger-length-compensation", "1.2",
                        "--cup-front-clearance", "5.5",
                        "--cup-front-max-push", "12.5",
                        "--cup-rim-clearance", "2.8",
                        "--cup-rim-max-push", "10.0",
                        "--allow-cup-correction-in-active-window",
                    ],
                ),
                stage1_log,
                args.dry_run,
            )
            require_outputs((stage1, stage1_report), stage1_log, args.dry_run)

        stage2_report = destination / "02_reach_clamp.json"
        if args.overwrite or not stage2.is_file() or not stage2_report.is_file():
            stage2_log = logs_dir / "02_reach_clamp.log"
            run_logged(
                stage_command(
                    args.blender,
                    stage1,
                    project_root / "BlenderScripts" / "build_v19_elbow_reach_clamp.py",
                    [
                        "--output", windows_path(stage2),
                        "--report", windows_path(stage2_report),
                    ],
                ),
                stage2_log,
                args.dry_run,
            )
            require_outputs((stage2, stage2_report), stage2_log, args.dry_run)

        stage3_report = destination / "03_shoulder_follow.json"
        if args.overwrite or not final.is_file() or not stage3_report.is_file():
            stage3_log = logs_dir / "03_shoulder_follow.log"
            run_logged(
                stage_command(
                    args.blender,
                    stage2,
                    project_root / "BlenderScripts" / "build_v19_elbow_shoulder_follow.py",
                    [
                        "--output", windows_path(final),
                        "--report", windows_path(stage3_report),
                        "--pack-resources",
                        "--trial-name", name,
                        "--source-anipose", windows_path(trial["pose"]),
                        "--calibration", windows_path(trial["calibration"]),
                    ],
                ),
                stage3_log,
                args.dry_run,
            )
            require_outputs((final, stage3_report), stage3_log, args.dry_run)

        manifest.update(
            {
                "status": "dry_run" if args.dry_run else "complete",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "outputs": {
                    "alignment": str(alignment_report.resolve()),
                    "hand_drive": str(stage1.resolve()),
                    "reach_clamp": str(stage2.resolve()),
                    "final": str(final.resolve()),
                },
            }
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise

    write_json(manifest_path, manifest)
    return manifest


def write_summary(output_root: Path, manifests: list[dict[str, object]]) -> None:
    write_json(output_root / "batch_summary.json", manifests)
    with (output_root / "batch_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("trial", "status", "elapsed_seconds", "final_blend", "error"),
        )
        writer.writeheader()
        for item in manifests:
            writer.writerow(
                {
                    "trial": item.get("trial"),
                    "status": item.get("status"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "final_blend": item.get("final_blend"),
                    "error": item.get("error", ""),
                }
            )


def write_audit(output_root: Path, trial_names: list[str]) -> dict[str, object]:
    rows = []
    for name in trial_names:
        directory = output_root / name
        alignment = json.loads(
            (directory / "alignment" / "alignment.json").read_text(encoding="utf-8")
        )
        hand = json.loads(
            (directory / "01_hand_drive.json").read_text(encoding="utf-8")
        )
        shoulder = json.loads(
            (directory / "03_shoulder_follow.json").read_text(encoding="utf-8")
        )
        normalized_fingers = hand.get("normalized_finger_pose", {}).get(
            "sides", {}
        )
        normalized_finger_count = sum(
            len(side_values) for side_values in normalized_fingers.values()
        )
        final = directory / f"{name}_AniposeSyntheticReady.blend"
        audits = shoulder["audit"]
        refinements = shoulder.get("elbow_refinement", {})
        below20 = sum(len(value["frames_below_20_deg"]) for value in audits.values())
        min_bend = min(value["minimum_elbow_bend_deg"] for value in audits.values())
        adjusted = sum(
            len(value.get("frames_adjusted", [])) for value in refinements.values()
        )
        rows.append(
            {
                "trial": name,
                "frame_count": alignment["frame_count"],
                "scene_frame_start": hand["scene_frame_range"][0],
                "scene_frame_end": hand["scene_frame_range"][1],
                "valid_descriptor_rows": alignment["valid_descriptor_rows"],
                "bind_frame_blender": hand["bind_frame_blender"],
                "camera_center_rmse": alignment["alignment_diagnostics"]["center_rmse"],
                "camera_rotation_mean_deg": alignment["alignment_diagnostics"]["rotation_mean_deg"],
                "minimum_elbow_bend_deg": min_bend,
                "frames_below_20_deg": below20,
                "refined_frame_sides": adjusted,
                "normalized_finger_count": normalized_finger_count,
                "final_blend_bytes": final.stat().st_size,
                "source_video_count": len(list((directory / "source" / "videos-raw").glob("*.mp4"))),
            }
        )

    bad_range = sum(
        row["scene_frame_start"] != 1
        or row["scene_frame_end"] != row["frame_count"]
        for row in rows
    )
    summary = {
        "project_count": len(rows),
        "bad_scene_range_count": bad_range,
        "frames_below_20_deg_total": sum(row["frames_below_20_deg"] for row in rows),
        "minimum_elbow_bend_deg": min(row["minimum_elbow_bend_deg"] for row in rows),
        "projects_with_refinement": sum(row["refined_frame_sides"] > 0 for row in rows),
        "refined_frame_sides_total": sum(row["refined_frame_sides"] for row in rows),
        "missing_source_video_sets": sum(row["source_video_count"] != 3 for row in rows),
        "projects_missing_normalized_fingers": sum(
            row["normalized_finger_count"] != 8 for row in rows
        ),
        "final_blend_bytes_total": sum(row["final_blend_bytes"] for row in rows),
    }
    write_json(output_root / "audit_summary.json", {"summary": summary, "projects": rows})
    with (output_root / "audit_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    for required in (args.dataset_root, args.base_blend, args.rig_inspection, args.blender):
        if not required.exists():
            raise FileNotFoundError(required)

    trials = discover_trials(args.dataset_root)
    if args.trial:
        selected = set(args.trial)
        trials = [trial for trial in trials if trial["name"] in selected]
        missing = selected - {str(trial["name"]) for trial in trials}
        if missing:
            raise ValueError(f"Unknown or incomplete trials: {sorted(missing)}")
    if args.limit is not None:
        trials = trials[: args.limit]
    if not trials:
        raise RuntimeError("No complete Anipose trials found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for index, trial in enumerate(trials, start=1):
        print(f"[{index}/{len(trials)}] {trial['name']}", flush=True)
        try:
            manifest = build_trial(args, project_root, trial)
        except Exception as error:
            manifest_path = args.output_root / str(trial["name"]) / "project_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests.append(manifest)
            write_summary(args.output_root, manifests)
            raise RuntimeError(f"Failed trial {trial['name']}: {error}") from error
        manifests.append(manifest)
        write_summary(args.output_root, manifests)

    audit = write_audit(
        args.output_root,
        [str(trial["name"]) for trial in trials],
    )
    if (
        audit["bad_scene_range_count"]
        or audit["frames_below_20_deg_total"]
        or audit["missing_source_video_sets"]
        or audit["projects_missing_normalized_fingers"]
    ):
        raise RuntimeError(f"Batch audit failed: {audit}")

    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "trials": len(manifests),
                "complete": sum(item["status"] == "complete" for item in manifests),
                "audit": audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

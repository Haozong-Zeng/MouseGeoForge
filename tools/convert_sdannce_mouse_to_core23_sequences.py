#!/usr/bin/env python3
"""Convert the s-DANNCE MOUSE cohort to compact Core23 pose sequences.

The downloaded cohort contains 3D trajectories and behavior annotations but
does not contain RGB videos or camera calibration. Social recordings appear
twice (once from each animal's perspective); exact swapped duplicates are
deduplicated while retaining both animals' labels from the canonical file.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from core23_temporal_format import (
    CORE23,
    FORMAT,
    map_to_core23,
    save_npz_atomic,
    sha256,
    stable_split,
    write_json_atomic,
    write_jsonl_atomic,
)


SOURCE_RAT23 = (
    "Snout",
    "EarL",
    "EarR",
    "SpineF",
    "SpineM",
    "SpineL",
    "TailBase",
    "ShoulderL",
    "ElbowL",
    "WristL",
    "HandL",
    "ShoulderR",
    "ElbowR",
    "WristR",
    "HandR",
    "HipL",
    "KneeL",
    "AnkleL",
    "FootL",
    "HipR",
    "KneeR",
    "AnkleR",
    "FootR",
)

SOURCE_TO_CORE23 = {
    "Snout": "nose",
    "EarL": "left_ear",
    "EarR": "right_ear",
    "SpineF": "neck",
    "SpineM": "spine_center",
    "SpineL": "lumbar_spine",
    "TailBase": "tail_base",
    "ElbowL": "left_elbow",
    "HandL": "left_forepaw_center",
    "ElbowR": "right_elbow",
    "HandR": "right_forepaw_center",
    "KneeL": "left_knee",
    "FootL": "left_hindpaw_center",
    "KneeR": "right_knee",
    "FootR": "right_hindpaw_center",
}

FILENAME = re.compile(
    r"^MOUSE_(?P<subject>[^_]+)_(?P<date>\d{8})_(?P<recording>\d+)_(?P<condition>[LS])\.mat$"
)
FPS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/sDANNCE-Mouse/raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/sDANNCE-Mouse-Core23-PoseSequences"),
    )
    return parser.parse_args()


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.shape == () else value)


def load_sdannce(path: Path) -> Any:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    return payload["sdannce"]


def group_source_files(paths: list[Path]) -> list[tuple[str, list[Path]]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        match = FILENAME.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected s-DANNCE filename: {path.name}")
        fields = match.groupdict()
        if fields["condition"] == "S":
            key = f"mouse_{fields['date']}_{fields['recording']}_social"
        else:
            key = f"mouse_{fields['date']}_{fields['recording']}_{fields['subject']}_lone"
        groups.setdefault(key, []).append(path)
    return [(key, sorted(value)) for key, value in sorted(groups.items())]


def as_track(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, len(SOURCE_RAT23), 3), dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (3, len(SOURCE_RAT23)):
        raise ValueError(f"Unexpected s-DANNCE track shape: {array.shape}")
    return array.transpose(0, 2, 1).astype(np.float32)


def stack_behavior(subject: Any, own: str, partner: str, instances: int) -> np.ndarray:
    first = np.asarray(getattr(subject, own))
    if instances == 1:
        return first[:, None]
    second = np.asarray(getattr(subject, partner))
    return np.stack((first, second), axis=1)


def validate_duplicate(canonical: Any, duplicate: Any) -> None:
    first = as_track(canonical.m1)
    second = as_track(canonical.m2)
    other_first = as_track(duplicate.m1)
    other_second = as_track(duplicate.m2)
    if first.shape != other_second.shape or second.shape != other_first.shape:
        raise ValueError("Social duplicate shapes do not swap exactly")
    if not np.allclose(first, other_second, equal_nan=True):
        raise ValueError("Social duplicate m1/m2 tracks do not swap exactly")
    if not np.allclose(second, other_first, equal_nan=True):
        raise ValueError("Social duplicate m2/m1 tracks do not swap exactly")


def convert_group(sequence_id: str, source_paths: list[Path], output_root: Path) -> dict:
    canonical_path = source_paths[0]
    source = load_sdannce(canonical_path)
    first = as_track(source.m1)
    second = as_track(source.m2)
    social = bool(int(source.issoc))
    if social != (second.shape[0] > 0):
        raise ValueError(f"issoc/track mismatch in {canonical_path}")
    if social and len(source_paths) == 2:
        validate_duplicate(source, load_sdannce(source_paths[1]))
    elif social and len(source_paths) != 2:
        raise ValueError(f"Expected two perspective files for {sequence_id}, got {len(source_paths)}")

    source_xyz = first[:, None] if not social else np.stack((first, second), axis=1)
    core_xyz, core_valid = map_to_core23(source_xyz, SOURCE_RAT23, SOURCE_TO_CORE23)
    frame_count, instance_count = source_xyz.shape[:2]
    focal_name = scalar_text(source.mouseid)
    partner_name = scalar_text(source.mouseid_p) if social else ""
    instance_names = [focal_name] + ([partner_name] if social else [])
    if social and (not partner_name or partner_name == focal_name):
        # Some reciprocal files contain a metadata typo; the other filename is reliable.
        other = FILENAME.match(source_paths[1].name)
        assert other is not None
        partner_name = other.group("subject")
        instance_names[1] = partner_name

    sequence_root = output_root / "sequences" / sequence_id
    sequence_path = sequence_root / "sequence.npz"
    arrays: dict[str, np.ndarray] = {
        "format": np.asarray(FORMAT),
        "names": np.asarray(CORE23),
        "source_names": np.asarray(SOURCE_RAT23),
        "instance_ids": np.arange(instance_count, dtype=np.int32),
        "instance_names": np.asarray(instance_names),
        "joint_world_xyz": core_xyz,
        "world_valid": core_valid,
        "source_joint_world_xyz": np.nan_to_num(source_xyz, nan=0.0).astype(np.float32),
        "source_world_valid": np.isfinite(source_xyz).all(axis=-1),
        "frame_indices": np.arange(frame_count, dtype=np.int64),
        "timestamps_seconds": np.arange(frame_count, dtype=np.float64) / FPS,
        "fps": np.asarray(FPS, dtype=np.float32),
        "fixed_camera": np.asarray(True),
        "camera_available": np.asarray(False),
        "rgb_available": np.asarray(False),
        "behavior_llac": stack_behavior(source, "llac", "part_llac", instance_count),
        "behavior_hlac": stack_behavior(source, "hlac", "part_hlac", instance_count),
        "behavior_lljc": stack_behavior(source, "lljc", "part_lljc", instance_count),
        "behavior_hljc": stack_behavior(source, "hljc", "part_hljc", instance_count),
        "embedding_action": stack_behavior(
            source, "cz_action", "part_cz_action", instance_count
        ).astype(np.float32),
        "embedding_joint": stack_behavior(
            source, "sz_joint", "part_sz_joint", instance_count
        ).astype(np.float32),
    }
    save_npz_atomic(sequence_path, **arrays)
    split = stable_split(sequence_id)
    record = {
        "format": FORMAT,
        "sequence_id": sequence_id,
        "source": "sDANNCE-Mouse",
        "split": split,
        "sequence_npz": sequence_path.as_posix(),
        "frame_count": frame_count,
        "fps": FPS,
        "instance_count": instance_count,
        "instance_names": instance_names,
        "fixed_camera": True,
        "camera_available": False,
        "rgb_available": False,
        "recommended_max_temporal_speed_mm_s": 5000.0,
        "pose_to_camera_alignment_valid": False,
        "source_files": [path.as_posix() for path in source_paths],
        "source_file_sha256": [sha256(path) for path in source_paths],
        "condition": "social" if social else "lone",
        "group": scalar_text(source.group),
        "mouse_strain": scalar_text(source.mousestrain),
    }
    write_json_atomic(sequence_root / "sequence_meta.json", record)
    return record


def main() -> None:
    args = parse_args()
    paths = sorted(args.raw_root.glob("MOUSE_*.mat"))
    if not paths:
        raise FileNotFoundError(f"No MOUSE_*.mat files in {args.raw_root}")
    groups = group_source_files(paths)
    records = []
    for index, (sequence_id, source_paths) in enumerate(groups, start=1):
        record = convert_group(sequence_id, source_paths, args.output_root)
        records.append(record)
        print(
            f"[{index:03d}/{len(groups):03d}] {sequence_id} "
            f"T={record['frame_count']} M={record['instance_count']}"
        )
    write_jsonl_atomic(args.output_root / "sequences.jsonl", records)
    metadata = {
        "format": FORMAT,
        "dataset_name": args.output_root.name,
        "data_origin": "real_pose_sequence",
        "source_project": "s-DANNCE",
        "source_repository": "https://github.com/tqxli/sdannce",
        "source_dataset": "MOUSE cohort, Harvard Dataverse doi:10.7910/DVN/VKJHTD",
        "source_file_count": len(paths),
        "sequence_count": len(records),
        "deduplicated_social_perspective_files": len(paths) - len(records),
        "frame_count": sum(record["frame_count"] for record in records),
        "social_sequence_count": sum(record["condition"] == "social" for record in records),
        "lone_sequence_count": sum(record["condition"] == "lone" for record in records),
        "keypoint_schema": "Core23 partial",
        "keypoint_names": list(CORE23),
        "source_keypoint_schema": "s-DANNCE rat23",
        "source_keypoint_names": list(SOURCE_RAT23),
        "source_to_core23": SOURCE_TO_CORE23,
        "mapped_core23_names": list(SOURCE_TO_CORE23.values()),
        "unmapped_core23_names": [
            name for name in CORE23 if name not in SOURCE_TO_CORE23.values()
        ],
        "world_coordinate_unit": "millimeter",
        "fps": FPS,
        "fixed_camera": True,
        "camera_available": False,
        "rgb_available": False,
        "note": (
            "The MOUSE cohort release contains 3D pose/behavior arrays only. "
            "No RGB, 2D projections, masks, intrinsics, or extrinsics are invented."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(args.output_root / "metadata.json", metadata)
    (args.output_root / "README.md").write_text(
        "# s-DANNCE Mouse Core23 pose sequences\n\n"
        "Compact, lossless temporal conversion of the released MOUSE cohort. "
        "Each `sequences/*/sequence.npz` stores `[T,M,23,3]` Core23 coordinates, "
        "validity, 50 Hz timestamps, source rat23 coordinates, and behavioral labels.\n\n"
        "Social recordings released twice from reciprocal animal perspectives are "
        "deduplicated after exact swapped-track validation. This is a pose-only real "
        "dataset: image, mask, camera, and reprojection supervision are unavailable.\n\n"
        "The source 3D trajectory is preserved without smoothing. The temporal "
        "loader emits `temporal_transition_valid` and by default masks point "
        "transitions faster than 5000 mm/s as likely tracking jumps.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

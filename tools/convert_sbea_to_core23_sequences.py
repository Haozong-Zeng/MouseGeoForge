#!/usr/bin/env python3
"""Convert released SBeA mouse sequences to compact BlenderPyMouse-style data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
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


SOURCE_SBEA16 = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_front_limb",
    "right_front_limb",
    "left_hind_limb",
    "right_hind_limb",
    "left_front_claw",
    "right_front_claw",
    "left_hind_claw",
    "right_hind_claw",
    "back",
    "root_tail",
    "mid_tail",
    "tip_tail",
)

SOURCE_TO_CORE23 = {
    "nose": "nose",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "neck": "neck",
    "left_front_limb": "left_elbow",
    "right_front_limb": "right_elbow",
    "left_hind_limb": "left_knee",
    "right_hind_limb": "right_knee",
    "left_front_claw": "left_forepaw_center",
    "right_front_claw": "right_forepaw_center",
    "left_hind_claw": "left_hindpaw_center",
    "right_hind_claw": "right_hindpaw_center",
    "back": "spine_center",
    "root_tail": "tail_base",
    "mid_tail": "tail_q3",
    "tip_tail": "tail_tip",
}

CAMERA_COUNT = 4
SOURCE_CAMERA_MATRIX_ORDER = (1, 2, 3, 0)
MOUSE_SEQUENCE_GLOBS = (
    "fig2_data/pose tracking/*-id3d.mat",
    "fig3_data/identification/oft_20cm_social/*-id3d.mat",
    "fig5_data/social/*-id3d.mat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unpacked-root", type=Path, default=Path("dataset/SBeA/unpacked")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/SBeA-Mouse16-Core23-RealMultiviewTemporal"),
    )
    return parser.parse_args()


def decode_compressed_counts(encoded: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        more = True
        while more:
            code = ord(encoded[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                value |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def rle_box(rle: dict[str, Any]) -> tuple[np.ndarray, bool]:
    height, width = map(int, rle["size"])
    cursor = 0
    foreground = False
    x_min, y_min = width, height
    x_max = y_max = -1
    for count in decode_compressed_counts(str(rle["counts"])):
        if count < 0 or cursor + count > height * width:
            return np.zeros(4, dtype=np.float32), False
        if foreground and count:
            start = cursor
            end = cursor + count - 1
            first_x, first_y = divmod(start, height)
            last_x, last_y = divmod(end, height)
            x_min = min(x_min, first_x)
            x_max = max(x_max, last_x)
            if first_x == last_x:
                y_min = min(y_min, first_y)
                y_max = max(y_max, last_y)
            else:
                y_min = 0
                y_max = height - 1
        cursor += count
        foreground = not foreground
    if cursor != height * width or x_max < x_min or y_max < y_min:
        return np.zeros(4, dtype=np.float32), False
    left, top = x_min / width, y_min / height
    right, bottom = (x_max + 1) / width, (y_max + 1) / height
    return np.asarray(
        ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
        dtype=np.float32,
    ), True


def parse_identity_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if rows and all(cell.strip().isdigit() for cell in rows[0]):
        rows = rows[1:]
    return [[cell.strip() for cell in row] for row in rows]


def video_properties(paths: list[Path]) -> tuple[int, float, int, int]:
    properties = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {path}")
            properties.append(
                (
                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                    float(capture.get(cv2.CAP_PROP_FPS)),
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            )
        finally:
            capture.release()
    if len(set(properties)) != 1:
        raise ValueError(f"Camera video properties differ: {properties}")
    return properties[0]


def decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsic, rotation, center_h, *_ = cv2.decomposeProjectionMatrix(projection)
    intrinsic = intrinsic / intrinsic[2, 2]
    center = (center_h[:3] / center_h[3]).reshape(3)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ center
    return intrinsic, extrinsic


def write_mask_bundle(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)


def convert_sequence(id_path: Path, unpacked_root: Path, output_root: Path) -> dict:
    prefix = id_path.with_name(id_path.name[: -len("-id3d.mat")])
    sequence_id = prefix.name
    cali_path = Path(f"{prefix}-caliParas.mat")
    identity_path = Path(f"{prefix}-corrpredid.csv")
    video_paths = [Path(f"{prefix}-camera-{camera}.avi") for camera in range(CAMERA_COUNT)]
    json_paths = [
        Path(f"{prefix}-camera-{camera}-correctedresult.json")
        for camera in range(CAMERA_COUNT)
    ]
    required = [cali_path, *video_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing companions for {id_path}: {missing}")
    mask_available = identity_path.is_file() and all(path.is_file() for path in json_paths)
    partially_missing_masks = identity_path.is_file() or any(path.is_file() for path in json_paths)
    if partially_missing_masks and not mask_available:
        raise FileNotFoundError(
            f"Incomplete SBeA identity/mask companions for {id_path}"
        )

    source = loadmat(id_path, squeeze_me=True, struct_as_record=False)
    coords = np.asarray(source["coords3d"], dtype=np.float32)
    # A few released files contain trailing spaces (for example ``"A9 "``).
    # Identity CSV rows do not, so normalize before ordering masks and boxes.
    instance_names = [
        str(value).strip() for value in np.asarray(source["name3d"]).reshape(-1)
    ]
    instance_count = len(instance_names)
    expected = instance_count * len(SOURCE_SBEA16) * 3
    if coords.ndim != 2 or coords.shape[1] != expected:
        raise ValueError(f"Unexpected SBeA coords3d shape in {id_path}: {coords.shape}")
    source_xyz = coords.reshape(coords.shape[0], instance_count, len(SOURCE_SBEA16), 3)
    core_xyz, core_valid = map_to_core23(source_xyz, SOURCE_SBEA16, SOURCE_TO_CORE23)
    frame_count = coords.shape[0]

    rgb_count, fps, width, height = video_properties(video_paths)
    identity_rows = parse_identity_rows(identity_path) if mask_available else []
    annotations = []
    if mask_available:
        for path in json_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            annotations.append(payload["annotations"][0])
        aligned_count = min(
            rgb_count,
            frame_count,
            len(identity_rows),
            *(len(annotation["segmentations"]) for annotation in annotations),
        )
    else:
        aligned_count = min(rgb_count, frame_count)
    if aligned_count <= 0:
        raise ValueError(f"No aligned RGB/mask/pose frames in {id_path}")

    boxes = np.zeros((aligned_count, CAMERA_COUNT, instance_count, 4), dtype=np.float32)
    box_valid = np.zeros((aligned_count, CAMERA_COUNT, instance_count), dtype=bool)
    mask_scores = np.zeros_like(box_valid, dtype=np.float32)
    ordered_masks: list[list[list[dict | None]]] = [
        [[None for _ in range(instance_count)] for _ in range(aligned_count)]
        for _ in range(CAMERA_COUNT)
    ]
    stable_index = {name: index for index, name in enumerate(instance_names)}
    if mask_available:
        for camera, annotation in enumerate(annotations):
            segmentations = annotation["segmentations"]
            scores = annotation.get("score", [])
            for frame in range(aligned_count):
                slot_names = identity_rows[frame]
                for slot, rle in enumerate(segmentations[frame]):
                    if slot >= len(slot_names):
                        continue
                    instance = stable_index.get(slot_names[slot])
                    if instance is None or not isinstance(rle, dict):
                        continue
                    box, valid = rle_box(rle)
                    boxes[frame, camera, instance] = box
                    box_valid[frame, camera, instance] = valid
                    ordered_masks[camera][frame][instance] = rle
                    if frame < len(scores) and slot < len(scores[frame]):
                        mask_scores[frame, camera, instance] = float(scores[frame][slot])

    calibration = loadmat(cali_path, squeeze_me=True, struct_as_record=False)["caliParams"]
    source_projection = np.asarray(calibration.cam_mat_all, dtype=np.float64)
    if source_projection.shape != (3, 4, CAMERA_COUNT):
        raise ValueError(f"Unexpected projection matrix shape: {source_projection.shape}")
    by_source_camera = {
        camera: source_projection[:, :, index]
        for index, camera in enumerate(SOURCE_CAMERA_MATRIX_ORDER)
    }
    projections = np.stack([by_source_camera[camera] for camera in range(CAMERA_COUNT)])
    decomposed = [decompose_projection(projection) for projection in projections]
    intrinsics = np.stack([item[0] for item in decomposed])
    extrinsics = np.stack([item[1] for item in decomposed])

    sequence_root = output_root / "sequences" / sequence_id
    sequence_path = sequence_root / "sequence.npz"
    mask_path = sequence_root / "aligned_masks.json.gz"
    rgb_pose_indices = np.arange(aligned_count, dtype=np.int64)
    save_npz_atomic(
        sequence_path,
        format=np.asarray(FORMAT),
        names=np.asarray(CORE23),
        source_names=np.asarray(SOURCE_SBEA16),
        instance_ids=np.arange(instance_count, dtype=np.int32),
        instance_names=np.asarray(instance_names),
        joint_world_xyz=core_xyz,
        world_valid=core_valid,
        source_joint_world_xyz=np.nan_to_num(source_xyz, nan=0.0).astype(np.float32),
        source_world_valid=np.isfinite(source_xyz).all(axis=-1),
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_seconds=np.arange(frame_count, dtype=np.float64) / fps,
        fps=np.asarray(fps, dtype=np.float32),
        fixed_camera=np.asarray(True),
        camera_available=np.asarray(True),
        rgb_available=np.asarray(True),
        rgb_pose_indices=rgb_pose_indices,
        video_frame_indices=np.arange(aligned_count, dtype=np.int64),
        boxes_cxcywh=boxes,
        box_valid=box_valid,
        mask_scores=mask_scores,
        projection_matrices=projections.astype(np.float64),
        intrinsics=intrinsics.astype(np.float64),
        extrinsics=extrinsics.astype(np.float64),
        pose_to_camera_alignment_valid=np.asarray(False),
        calibration_rotation_hint=np.asarray(calibration.rotation, dtype=np.float64),
        calibration_translation_hint=np.asarray(calibration.translation, dtype=np.float64),
    )
    if mask_available:
        write_mask_bundle(
            mask_path,
            {
                "format": FORMAT,
                "sequence_id": sequence_id,
                "image_size": [width, height],
                "camera_ids": list(range(CAMERA_COUNT)),
                "pose_frame_indices": rgb_pose_indices.tolist(),
                "instance_names": instance_names,
                "segmentations_by_camera": ordered_masks,
            },
        )
    relative_parent = prefix.parent.resolve().relative_to(unpacked_root.resolve())
    split = stable_split(sequence_id)
    record = {
        "format": FORMAT,
        "sequence_id": sequence_id,
        "source": "SBeA",
        "source_subset": relative_parent.as_posix(),
        "split": split,
        "sequence_npz": sequence_path.as_posix(),
        "mask_rle": mask_path.as_posix() if mask_available else None,
        "frame_count": frame_count,
        "rgb_frame_count": aligned_count,
        "fps": fps,
        "image_size": [width, height],
        "instance_count": instance_count,
        "instance_names": instance_names,
        "camera_count": CAMERA_COUNT,
        "fixed_camera": True,
        "camera_available": True,
        "rgb_available": True,
        "mask_available": mask_available,
        "pose_to_camera_alignment_valid": False,
        "recommended_max_temporal_speed_mm_s": 5000.0,
        "video_paths": [path.as_posix() for path in video_paths],
        "source_annotation_paths": (
            [path.as_posix() for path in json_paths] if mask_available else []
        ),
        "source_pose_path": id_path.as_posix(),
        "source_pose_sha256": sha256(id_path),
        "source_calibration_path": cali_path.as_posix(),
        "source_identity_path": identity_path.as_posix() if mask_available else None,
    }
    write_json_atomic(sequence_root / "sequence_meta.json", record)
    return record


def main() -> None:
    args = parse_args()
    id_paths = sorted(
        path
        for pattern in MOUSE_SEQUENCE_GLOBS
        for path in args.unpacked_root.glob(pattern)
    )
    if not id_paths:
        raise FileNotFoundError(f"No SBeA mouse id3d files below {args.unpacked_root}")
    records = []
    for index, path in enumerate(id_paths, start=1):
        record = convert_sequence(path, args.unpacked_root, args.output_root)
        records.append(record)
        print(
            f"[{index:02d}/{len(id_paths):02d}] {record['sequence_id']} "
            f"pose={record['frame_count']} rgb={record['rgb_frame_count']}"
        )
    write_jsonl_atomic(args.output_root / "sequences.jsonl", records)
    metadata = {
        "format": FORMAT,
        "dataset_name": args.output_root.name,
        "data_origin": "real_multiview_temporal",
        "source_project": "SBeA",
        "source_repository": "https://github.com/YNCris/SBeA_release",
        "sequence_count": len(records),
        "pose_frame_count": sum(record["frame_count"] for record in records),
        "rgb_aligned_frame_count": sum(record["rgb_frame_count"] for record in records),
        "mask_aligned_frame_count": sum(
            record["rgb_frame_count"] for record in records if record["mask_available"]
        ),
        "mask_sequence_count": sum(record["mask_available"] for record in records),
        "views_per_sequence": CAMERA_COUNT,
        "instance_count": 2,
        "keypoint_schema": "Core23 partial",
        "keypoint_names": list(CORE23),
        "source_keypoint_schema": "SBeA mouse16",
        "source_keypoint_names": list(SOURCE_SBEA16),
        "source_to_core23": SOURCE_TO_CORE23,
        "mapped_core23_names": list(SOURCE_TO_CORE23.values()),
        "unmapped_core23_names": [
            name for name in CORE23 if name not in SOURCE_TO_CORE23.values()
        ],
        "world_coordinate_unit": "millimeter",
        "fixed_camera": True,
        "camera_parameters_are_constant_per_sequence": True,
        "pose_to_camera_alignment_valid": False,
        "note": (
            "The released 900-frame AVIs align with annotation/pose indices 0..899; "
            "the remaining 3D trajectory is retained as pose-only supervision. "
            "Projection matrices describe the fixed rig, but the release omits the "
            "rigid transform from ground-aligned id3d coordinates back to the "
            "calibration reconstruction frame, so 3D-to-2D reprojection is disabled."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(args.output_root / "metadata.json", metadata)
    (args.output_root / "README.md").write_text(
        "# SBeA mouse Core23 real multiview temporal dataset\n\n"
        "Compact BlenderPyMouse-style sequence conversion of 40 released two-mouse "
        "recordings. Each sequence keeps the full 27,000-frame 3D trajectory and "
        "references four fixed-camera AVI files. The released AVIs contain 900 "
        "aligned frames; boxes and identity-ordered COCO RLE masks are provided for "
        "that interval.\n\n"
        "The fixed camera projection matrices are retained and decomposed, but "
        "3D-to-2D reprojection supervision is deliberately disabled because the "
        "ground-alignment transform applied to `id3d` is absent from the release.\n\n"
        "The source 3D trajectory is preserved without smoothing. The temporal "
        "loader emits `temporal_transition_valid` and by default masks point "
        "transitions faster than 5000 mm/s as likely tracking jumps.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

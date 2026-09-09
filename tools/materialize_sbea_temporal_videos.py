#!/usr/bin/env python3
"""Copy SBeA AVI sources into the converted temporal dataset atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from core23_temporal_format import write_json_atomic, write_jsonl_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "dataset/SBeA-Mouse16-Core23-RealMultiviewTemporal/sequences.jsonl"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def resolve(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def copy_atomic(source: Path, destination: Path) -> tuple[str, int]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        while block := input_stream.read(16 * 1024 * 1024):
            digest.update(block)
            output_stream.write(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    shutil.copystat(source, temporary)
    temporary.replace(destination)
    if destination.stat().st_size != source.stat().st_size:
        raise IOError(f"Copied size mismatch: {source} -> {destination}")
    return digest.hexdigest(), source.stat().st_size


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest = resolve(project_root, args.manifest)
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_bytes = 0
    total_files = 0
    for index, record in enumerate(records, start=1):
        source_paths = record.get("source_video_paths", record["video_paths"])
        if len(source_paths) != int(record["camera_count"]):
            raise ValueError(f"Camera/video mismatch for {record['sequence_id']}")
        sequence_root = resolve(project_root, record["sequence_npz"]).parent
        copied_paths: list[str] = []
        hashes: list[str] = []
        sizes: list[int] = []
        existing_hashes = record.get("video_sha256", [])
        existing_sizes = record.get("video_sizes_bytes", [])
        for camera, source_path in enumerate(source_paths):
            source = resolve(project_root, source_path)
            destination = sequence_root / "videos" / f"camera-{camera}.avi"
            can_reuse = (
                destination.is_file()
                and camera < len(existing_hashes)
                and camera < len(existing_sizes)
                and destination.stat().st_size == int(existing_sizes[camera])
                and (
                    not source.exists()
                    or source.stat().st_size == int(existing_sizes[camera])
                )
            )
            if can_reuse:
                digest, size = existing_hashes[camera], int(existing_sizes[camera])
            else:
                digest, size = copy_atomic(source, destination)
            copied_paths.append(destination.relative_to(project_root).as_posix())
            hashes.append(digest)
            sizes.append(size)
        record["source_video_paths"] = source_paths
        record["video_paths"] = copied_paths
        record["video_sha256"] = hashes
        record["video_sizes_bytes"] = sizes
        record["video_storage"] = "copied_into_sequence_directory"
        write_json_atomic(sequence_root / "sequence_meta.json", record)
        total_bytes += sum(sizes)
        total_files += len(sizes)
        print(
            f"[{index:02d}/{len(records):02d}] {record['sequence_id']} "
            f"{sum(sizes) / (1024**2):.1f} MiB"
        )

    write_jsonl_atomic(manifest, records)
    metadata_path = manifest.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "video_storage": "copied_into_sequence_directory",
            "video_file_count": total_files,
            "video_total_bytes": total_bytes,
            "dataset_is_self_contained_for_rgb": True,
        }
    )
    write_json_atomic(metadata_path, metadata)
    print(f"Copied {total_files} videos, {total_bytes / (1024**3):.3f} GiB")


if __name__ == "__main__":
    main()

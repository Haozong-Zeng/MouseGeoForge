#!/usr/bin/env python3
"""Build a recording-level manifest for the extracted CRIM13 dataset."""

from __future__ import annotations

import csv
import re
import struct
from pathlib import Path


DATASET = Path("/home/haozong/projects/MouseGeoForge/dataset/CRIM13")
OUTPUT = DATASET / "manifest.csv"
ANNOTATION_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\S+)\s*$")
SOURCES = (
    ("train", "train1", DATASET / "train1"),
    ("train", "train2", DATASET / "train2"),
    ("test", "test1", DATASET / "test1"),
    ("test", "test2", DATASET / "test2"),
    ("validation_train", "validation", DATASET / "validation" / "train"),
    ("validation_test", "validation", DATASET / "validation" / "test"),
)


def seq_header(path: Path) -> dict[str, int | float]:
    with path.open("rb") as stream:
        header = stream.read(1024)
    if len(header) != 1024 or struct.unpack_from("<I", header, 0)[0] != 0xFEED:
        raise ValueError(f"Invalid Norpix SEQ header: {path}")
    fields = struct.unpack_from("<9I", header, 548)
    return {
        "width": fields[0],
        "height": fields[1],
        "image_format": fields[5],
        "num_frames": fields[6],
        "fps": struct.unpack_from("<d", header, 584)[0],
    }


def annotation_frames(path: Path) -> int:
    final_frame = 0
    previous_end = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ANNOTATION_RE.match(line)
        if not match:
            continue
        start, end, _ = match.groups()
        start_frame = int(start)
        end_frame = int(end)
        if start_frame != previous_end + 1 or end_frame < start_frame:
            raise ValueError(f"Non-contiguous annotation in {path}: {line!r}")
        previous_end = end_frame
        final_frame = end_frame
    return final_frame


def relative(path: Path | None) -> str:
    return path.relative_to(DATASET).as_posix() if path is not None else ""


def corrupt_entries(directory: Path) -> set[str]:
    marker = directory / "CORRUPT_ENTRIES.txt"
    if not marker.is_file():
        return set()
    entries: set[str] = set()
    for line in marker.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("- ") and ": " in line:
            entries.add(line[2:].split(": ", 1)[0])
    return entries


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, package, directory in SOURCES:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        corrupt = corrupt_entries(directory if package != "validation" else DATASET / "validation")
        recording_ids: set[str] = set()
        suffixes = ("_t.seq", "_s.seq", "_t.txt", "_t-track.mat", "_t-seek.mat")
        for path in directory.iterdir():
            if not path.is_file() or path.name.endswith("-backup.txt"):
                continue
            for suffix in suffixes:
                if path.name.endswith(suffix):
                    recording_ids.add(path.name[: -len(suffix)])
                    break
        for recording_id in sorted(recording_ids):
            top_path = directory / f"{recording_id}_t.seq"
            side_path = directory / f"{recording_id}_s.seq"
            annotation_path = directory / f"{recording_id}_t.txt"
            track_path = directory / f"{recording_id}_t-track.mat"
            seek_path = directory / f"{recording_id}_t-seek.mat"
            issues: list[str] = []

            top: dict[str, int | float] | None = None
            if top_path.is_file():
                top = seq_header(top_path)
            else:
                corrupt_name = top_path.name in corrupt
                issues.append("corrupt_top" if corrupt_name else "top_unavailable")

            side: dict[str, int | float] | None = None
            if side_path.is_file():
                side = seq_header(side_path)
                if top is not None and side["num_frames"] != top["num_frames"]:
                    issues.append("top_side_frame_mismatch")
            else:
                corrupt_name = side_path.name in corrupt
                issues.append("corrupt_side" if corrupt_name else "side_unavailable")

            primary = top or side
            if primary is None:
                issues.append("video_unavailable")
                continue

            if not annotation_path.is_file():
                issues.append("annotation_missing")
                label_frames = 0
            else:
                label_frames = annotation_frames(annotation_path)
                if label_frames != primary["num_frames"]:
                    issues.append("annotation_frame_mismatch")
            if not track_path.is_file():
                issues.append("track_missing")
            if not seek_path.is_file():
                issues.append("seek_missing")

            rows.append(
                {
                    "split": split,
                    "package": package,
                    "recording_id": recording_id,
                    "top_seq": relative(top_path) if top_path.is_file() else "",
                    "side_seq": relative(side_path) if side_path.is_file() else "",
                    "annotation_txt": relative(annotation_path) if annotation_path.is_file() else "",
                    "track_mat": relative(track_path) if track_path.is_file() else "",
                    "seek_mat": relative(seek_path) if seek_path.is_file() else "",
                    "num_frames": primary["num_frames"],
                    "annotation_frames": label_frames,
                    "width": primary["width"],
                    "height": primary["height"],
                    "fps_top": f"{top['fps']:.9f}" if top else "",
                    "fps_side": f"{side['fps']:.9f}" if side else "",
                    "image_format": primary["image_format"],
                    "issues": ";".join(issues),
                }
            )
    rows.sort(key=lambda row: (str(row["split"]), str(row["recording_id"])))
    return rows


def main() -> None:
    rows = build_rows()
    fieldnames = list(rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

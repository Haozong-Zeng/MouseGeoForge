#!/usr/bin/env python3
"""Extract synchronized frame sets from camera videos with OpenCV."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


@dataclass(frozen=True)
class VideoInfo:
    camera: str
    path: Path
    width: int
    height: int
    frame_count: int
    fps: float


def infer_camera_name(path: Path) -> str:
    """Infer a compact camera name from filenames such as pawfilt_camA.mp4."""
    stem = path.stem
    patterns = (
        r".*(?:^|[_-])cam(?:era)?[_-]?([A-Za-z0-9]+)$",
        r"^(?:cam|camera)[_-]?([A-Za-z0-9]+)$",
        r"^p([A-Za-z0-9]+)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return stem


def read_video_info(path: Path, camera: str | None = None) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Video has invalid dimensions: {path}")
    if frame_count <= 0:
        raise ValueError(f"Video has no readable frame count: {path}")
    return VideoInfo(
        camera=camera or infer_camera_name(path),
        path=path,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
    )


def discover_videos(
    video_dir: Path,
    camera_names: Sequence[str] | None = None,
    extensions: Sequence[str] = VIDEO_EXTENSIONS,
) -> list[VideoInfo]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    extensions = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions)
    candidates = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in extensions)
    if not candidates:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    by_camera: dict[str, Path] = {}
    for path in candidates:
        name = infer_camera_name(path)
        if name in by_camera:
            raise ValueError(f"Multiple videos map to camera {name!r}: {by_camera[name]} and {path}")
        by_camera[name] = path

    if camera_names:
        missing = [name for name in camera_names if name not in by_camera]
        if missing:
            available = ", ".join(sorted(by_camera))
            raise ValueError(f"Missing video(s) for camera(s): {', '.join(missing)}. Available: {available}")
        ordered = [(name, by_camera[name]) for name in camera_names]
    else:
        ordered = sorted(by_camera.items())

    return [read_video_info(path, camera=name) for name, path in ordered]


def parse_frame_spec(spec: str, frame_count: int) -> list[int]:
    """Parse comma lists, Python-like ranges, or 'all' into frame indices."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    spec = spec.strip()
    if not spec:
        raise ValueError("Frame spec is empty")
    if spec.lower() == "all":
        return list(range(frame_count))

    frames: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) > 3:
                raise ValueError(f"Bad frame range {part!r}; expected start:stop[:step]")
            start = int(bits[0]) if bits[0] else 0
            stop = int(bits[1]) if len(bits) > 1 and bits[1] else frame_count
            step = int(bits[2]) if len(bits) > 2 and bits[2] else 1
            if step <= 0:
                raise ValueError(f"Frame range step must be positive: {part!r}")
            frames.extend(range(start, min(stop, frame_count), step))
        else:
            frames.append(int(part))

    deduped: list[int] = []
    seen: set[int] = set()
    for frame in frames:
        if frame < 0 or frame >= frame_count:
            raise ValueError(f"Frame index {frame} is outside valid range 0..{frame_count - 1}")
        if frame not in seen:
            seen.add(frame)
            deduped.append(frame)
    if not deduped:
        raise ValueError("No frames selected")
    return deduped


def common_frame_count(videos: Sequence[VideoInfo]) -> int:
    if not videos:
        raise ValueError("No videos were provided")
    return min(video.frame_count for video in videos)


def extract_frame_sets(
    videos: Sequence[VideoInfo],
    frame_indices: Sequence[int],
    output_dir: Path,
    image_extension: str = ".png",
) -> list[dict[str, object]]:
    """Extract each requested frame into output_dir/frame_000000/pA.png style sets."""
    records: list[dict[str, object]] = []
    for _, frame_records in iter_extract_frame_sets(videos, frame_indices, output_dir, image_extension):
        records.extend(frame_records)
    return records


def iter_extract_frame_sets(
    videos: Sequence[VideoInfo],
    frame_indices: Sequence[int],
    output_dir: Path,
    image_extension: str = ".png",
) -> Iterable[tuple[int, list[dict[str, object]]]]:
    """Yield extracted frame-set records while keeping video captures open."""
    if not image_extension.startswith("."):
        image_extension = "." + image_extension

    output_dir.mkdir(parents=True, exist_ok=True)

    captures: list[tuple[VideoInfo, cv2.VideoCapture]] = []
    try:
        for video in videos:
            cap = cv2.VideoCapture(str(video.path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video.path}")
            captures.append((video, cap))

        for frame_index in frame_indices:
            frame_dir = output_dir / f"frame_{frame_index:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, object]] = []

            for video, cap in captures:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read frame {frame_index} from {video.path}")

                image_path = frame_dir / f"p{video.camera}{image_extension}"
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Failed to write image: {image_path}")

                records.append(
                    {
                        "frame_index": frame_index,
                        "camera": video.camera,
                        "image_path": str(image_path),
                        "video_path": str(video.path),
                        "width": video.width,
                        "height": video.height,
                    }
                )
            yield frame_index, records
    finally:
        for _, cap in captures:
            cap.release()


def write_manifest(records: Iterable[dict[str, object]], output_path: Path) -> None:
    rows = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_index", "camera", "image_path", "video_path", "width", "height"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("temp_data/videos"))
    parser.add_argument("--video-dir", type=Path, help="Defaults to DATASET/videos-raw.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-names", nargs="+", default=["A", "B", "C"])
    parser.add_argument(
        "--frames",
        default="0",
        help="Frame list/range, e.g. 0,10,20 or 0:405:5 or all. Default: 0.",
    )
    parser.add_argument("--image-extension", default=".png")
    parser.add_argument("--manifest", type=Path, help="Defaults to OUTPUT_DIR/manifest.csv.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    video_dir = args.video_dir or args.dataset / "videos-raw"
    videos = discover_videos(video_dir, args.camera_names)
    frame_count = common_frame_count(videos)
    frame_indices = parse_frame_spec(args.frames, frame_count)
    records = extract_frame_sets(videos, frame_indices, args.output_dir, args.image_extension)
    manifest = args.manifest or args.output_dir / "manifest.csv"
    write_manifest(records, manifest)

    cameras = ", ".join(video.camera for video in videos)
    print(f"Extracted {len(frame_indices)} frame set(s) for cameras: {cameras}")
    print(f"Wrote frames to {args.output_dir}")
    print(f"Wrote manifest to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

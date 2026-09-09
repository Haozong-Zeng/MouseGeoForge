#!/usr/bin/env python3
"""Score candidate SBeA mask/pose indices against the released short AVI clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode_coco_rle(rle: dict) -> np.ndarray:
    """Decode COCO's compressed column-major run-length representation."""
    encoded = rle["counts"]
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

    height, width = map(int, rle["size"])
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    foreground = False
    for count in counts:
        if count < 0 or cursor + count > flat.size:
            raise ValueError("Invalid compressed COCO RLE")
        if foreground:
            flat[cursor : cursor + count] = 1
        cursor += count
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError(f"COCO RLE covers {cursor} pixels, expected {flat.size}")
    return flat.reshape((height, width), order="F").astype(bool)


def mask_darkness_score(frame: np.ndarray, masks: list[dict]) -> tuple[float, float]:
    union = np.zeros(frame.shape[:2], dtype=bool)
    for rle in masks:
        union |= decode_coco_rle(rle)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if not union.any() or union.all():
        return float("nan"), float("nan")
    inside = float(gray[union].mean())
    outside = float(gray[~union].mean())
    return outside - inside, float(union.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_prefix", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--video-frames", type=int, nargs="+", default=(0, 100, 300, 600, 899)
    )
    args = parser.parse_args()

    video_path = Path(f"{args.sequence_prefix}-camera-{args.camera}.avi")
    json_path = Path(
        f"{args.sequence_prefix}-camera-{args.camera}-correctedresult.json"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segmentations = payload["annotations"][0]["segmentations"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = len(segmentations) / video_count
    print(f"video_frames={video_count} annotation_frames={len(segmentations)} ratio={ratio:g}")
    try:
        for video_index in args.video_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode video frame {video_index}")
            candidates = {
                "same": video_index,
                "stride": min(int(round(video_index * ratio)), len(segmentations) - 1),
            }
            parts = []
            for name, annotation_index in candidates.items():
                contrast, area = mask_darkness_score(
                    frame, segmentations[annotation_index]
                )
                parts.append(
                    f"{name}[{annotation_index}]:contrast={contrast:.3f},area={area:.5f}"
                )
            print(f"video[{video_index}] " + " ".join(parts))
    finally:
        capture.release()


if __name__ == "__main__":
    main()

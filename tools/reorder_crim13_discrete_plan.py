#!/usr/bin/env python3
"""Move one selected source pose to a fixed scene slot without changing the set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--height-report", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, required=True)
    parser.add_argument("--scene-frame", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    keyframes = plan["keyframes"]
    matches = [index for index, row in enumerate(keyframes) if int(row["source_image_frame"]) == args.source_frame]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source frame {args.source_frame}, got {matches}")
    row = keyframes.pop(matches[0])
    keyframes.insert(args.scene_frame - 1, row)
    for index, keyframe in enumerate(keyframes, start=1):
        keyframe["scene_frame"] = index
    plan["source_label_frames"] = [int(keyframe["source_image_frame"]) for keyframe in keyframes]
    plan["frame_order_policy"] = (
        "chronological except geometry-QA replacements occupy the rejected frame's original scene slot"
    )
    plan.setdefault("incremental_replacement", {})["replacement_scene_slots"] = {
        str(args.source_frame): args.scene_frame
    }
    args.plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    height = json.loads(args.height_report.read_text(encoding="utf-8"))
    by_source = {int(row["source_image_frame"]): row for row in height["keyframes"]}
    reordered = []
    for keyframe in keyframes:
        source = int(keyframe["source_image_frame"])
        height_row = by_source[source]
        height_row["scene_frame"] = int(keyframe["scene_frame"])
        reordered.append(height_row)
    height["keyframes"] = reordered
    args.height_report.write_text(json.dumps(height, indent=2), encoding="utf-8")
    print(json.dumps({
        "plan": str(args.plan),
        "height_report": str(args.height_report),
        "moved_source_frame": args.source_frame,
        "scene_frame": args.scene_frame,
        "frame_count": len(keyframes),
    }, indent=2))


if __name__ == "__main__":
    main()

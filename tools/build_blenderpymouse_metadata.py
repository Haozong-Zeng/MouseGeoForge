#!/usr/bin/env python3
"""Build a root metadata.json for BlenderPyMouse BlendedMVS-like exports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root containing frame_*/{blended_images,rendered_depth_maps,cams}.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format-name", default="water_reaching_blendedmvs_like")
    return parser.parse_args()


def scene_is_complete(scene_root: Path) -> tuple[bool, list[int], list[str]]:
    cams_dir = scene_root / "cams"
    image_dir = scene_root / "blended_images"
    depth_dir = scene_root / "rendered_depth_maps"
    missing: list[str] = []
    view_ids: list[int] = []
    for cam_path in sorted(cams_dir.glob("*_cam.txt")):
        view_id = int(cam_path.stem.split("_")[0])
        view_ids.append(view_id)
        for path in [
            image_dir / f"{view_id:08d}.jpg",
            depth_dir / f"{view_id:08d}.pfm",
        ]:
            if not path.exists():
                missing.append(str(path.relative_to(scene_root)))
    if not (cams_dir / "pair.txt").exists():
        missing.append("cams/pair.txt")
    if not (scene_root / "scene_meta.json").exists():
        missing.append("scene_meta.json")
    return len(missing) == 0 and bool(view_ids), view_ids, missing


def load_camera_names(scene_root: Path, view_ids: list[int]) -> list[str]:
    names: list[str] = []
    for view_id in view_ids:
        ann_path = scene_root / "annotations" / f"{view_id:08d}.json"
        if ann_path.exists():
            try:
                item = json.loads(ann_path.read_text(encoding="utf-8"))
                names.append(str(item.get("camera", f"view_{view_id:08d}")))
                continue
            except json.JSONDecodeError:
                pass
        names.append(f"view_{view_id:08d}")
    return names


def load_existing_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not preserve existing metadata from {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    scene_roots = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("frame_"))
    complete_scenes: list[str] = []
    incomplete: list[dict] = []
    view_counts: dict[int, int] = {}
    first_complete: Path | None = None
    first_view_ids: list[int] = []

    for scene_root in scene_roots:
        complete, view_ids, missing = scene_is_complete(scene_root)
        if complete:
            complete_scenes.append(scene_root.name)
            first_complete = first_complete or scene_root
            first_view_ids = first_view_ids or view_ids
            view_counts[len(view_ids)] = view_counts.get(len(view_ids), 0) + 1
        else:
            incomplete.append({"scene": scene_root.name, "view_ids": view_ids, "missing": missing[:20]})

    frame_numbers = []
    for name in complete_scenes:
        try:
            frame_numbers.append(int(name.split("_")[-1]))
        except ValueError:
            pass

    cameras = load_camera_names(first_complete, first_view_ids) if first_complete is not None else []
    output = args.output.resolve() if args.output else root / "metadata.json"
    existing = load_existing_metadata(output)
    available_frame_range = [min(frame_numbers), max(frame_numbers)] if frame_numbers else []
    requested_frames = existing.get("requested_frames")
    old_frames = existing.get("frames")
    if requested_frames is None and isinstance(old_frames, list) and old_frames != available_frame_range:
        requested_frames = old_frames
    if (
        isinstance(requested_frames, list)
        and len(requested_frames) == 2
        and available_frame_range
    ):
        requested_frames = [
            min(requested_frames[0], available_frame_range[0]),
            max(requested_frames[1], available_frame_range[1]),
        ]

    metadata = {
        **existing,
        "format": args.format_name,
        "dataset_name": root.name,
        "dataset_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(complete_scenes),
        "incomplete_scene_count": len(incomplete),
        "scene_folders": complete_scenes,
        "incomplete_scenes": incomplete,
        "frames": available_frame_range,
        "frame_numbers": frame_numbers,
        "cameras": cameras,
        "view_counts": {str(key): value for key, value in sorted(view_counts.items())},
        "image_format": "JPEG",
        "depth_format": "PFM",
        "layout": {
            "images": "frame_xxxxxx/blended_images/00000000.jpg",
            "masked_images": "frame_xxxxxx/blended_images/00000000_masked.jpg",
            "depths": "frame_xxxxxx/rendered_depth_maps/00000000.pfm",
            "cameras": "frame_xxxxxx/cams/00000000_cam.txt",
            "pair_file": "frame_xxxxxx/cams/pair.txt",
            "annotations": "frame_xxxxxx/annotations/00000000.json",
        },
    }
    if requested_frames is not None:
        metadata["requested_frames"] = requested_frames
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "scene_count": len(complete_scenes), "incomplete": len(incomplete)}, indent=2))


if __name__ == "__main__":
    main()

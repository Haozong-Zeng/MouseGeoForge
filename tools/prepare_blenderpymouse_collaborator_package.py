#!/usr/bin/env python3
"""Build a self-contained 10-frame BlenderPyMouse collaborator example package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_FRAMES = (1, 11, 22, 31, 45, 68, 77, 82, 85, 88)
DEFAULT_REASONS = {
    1: "neutral quadruped baseline",
    11: "early rise with forelimb support",
    22: "partial upright transition",
    31: "fully upright posture",
    45: "high reach with asymmetric forelimbs",
    68: "late high-reach posture",
    77: "descent transition",
    82: "landing and low crouch",
    85: "quadruped recovery",
    88: "low grounded posture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--frames",
        default=",".join(str(frame) for frame in DEFAULT_FRAMES),
        help="Comma-separated source frame numbers.",
    )
    return parser.parse_args()


def parse_frames(text: str) -> list[int]:
    frames = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(frames) != 10 or len(set(frames)) != 10:
        raise ValueError("Exactly 10 unique frame numbers are required")
    return frames


def view_ids(frame_root: Path) -> list[int]:
    result = []
    for path in sorted((frame_root / "blended_images").iterdir()):
        if not path.is_file() or path.stem.endswith("_masked"):
            continue
        try:
            result.append(int(path.stem))
        except ValueError:
            continue
    if not result:
        raise ValueError(f"No numbered RGB views in {frame_root}")
    return result


def make_contact_sheet(source: Path, frames: list[int], output: Path) -> None:
    first_root = source / f"frame_{frames[0]:06d}"
    available_views = view_ids(first_root)
    if len(available_views) <= 4:
        selected_views = available_views
    else:
        selected_views = [available_views[index] for index in (0, len(available_views) // 3, 2 * len(available_views) // 3, len(available_views) - 1)]

    tile_width, tile_height, label_height = 180, 180, 24
    sheet = Image.new(
        "RGB",
        (tile_width * len(selected_views), (tile_height + label_height) * len(frames)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, frame in enumerate(frames):
        y0 = row * (tile_height + label_height)
        draw.text((4, y0 + 4), f"frame {frame:06d}", fill="black")
        for column, view_id in enumerate(selected_views):
            image_dir = source / f"frame_{frame:06d}" / "blended_images"
            candidates = [image_dir / f"{view_id:08d}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(f"Missing view {view_id} for frame {frame}")
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((tile_width, tile_height))
                x = column * tile_width + (tile_width - image.width) // 2
                y = y0 + label_height + (tile_height - image.height) // 2
                sheet.paste(image, (x, y))
            draw.text((column * tile_width + 100, y0 + 4), f"view {view_id}", fill="black")
    sheet.save(output, quality=92)


def sanitize_metadata(source: Path, output: Path, frames: list[int]) -> dict:
    source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    first_frame = source / f"frame_{frames[0]:06d}"
    count = len(view_ids(first_frame))
    metadata = dict(source_metadata)
    metadata.update(
        {
            "dataset_name": output.name,
            "dataset_root": ".",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset_name": source.name,
            "source_dataset_format": source_metadata.get("format"),
            "frames": [min(frames), max(frames)],
            "frame_numbers": frames,
            "scene_count": len(frames),
            "incomplete_scene_count": 0,
            "scene_folders": [f"frame_{frame:06d}" for frame in frames],
            "incomplete_scenes": [],
            "view_counts": {str(count): len(frames)},
            "availability": {
                "rgb": True,
                "masked_rgb": True,
                "camera_intrinsics": True,
                "camera_extrinsics": True,
                "depth": True,
                "mouse_mask": True,
                "per_view_annotations": True,
                "keypoints_json": True,
                "keypoints_npz": True,
            },
            "camera_convention": {
                "extrinsic_direction": "camera_from_world",
                "axes": "OpenCV: +x right, +y down, +z forward",
                "intrinsic_units": "pixels",
                "fixed_rig": True,
            },
            "depth_convention": {
                "format": "single-channel float32 PFM",
                "row_order": "PFM bottom-to-top; current loader applies vertical flip",
                "unit": source_metadata.get("depth_units", "Blender world units"),
                "value_type": "source Blender depth pass; declare camera_z or ray_distance for new submissions",
                "invalid_value": 0.0,
                "per_frame_normalized": False,
            },
            "representative_selection": {
                "method": "3D Core23 pose diversity followed by four-view visual review",
                "selected_frames": frames,
                "selection_reasons": {str(frame): DEFAULT_REASONS.get(frame, "representative pose") for frame in frames},
                "contact_sheet": "representative_frames_contact_sheet.jpg",
            },
        }
    )
    for key in ("source_cylinder_root", "source_render_root"):
        metadata.pop(key, None)
    if "source_blend" in metadata:
        metadata["source_blend"] = Path(str(metadata["source_blend"]).replace("\\", "/")).name
    annotations = metadata.get("keypoint_annotations")
    if isinstance(annotations, dict):
        annotations["vertex_support_map"] = "core23_vertex_support_map.pkl"
        annotations["vertex_support_audit"] = "core23_vertex_support_map.audit.json"
    return metadata


def write_manifest(output: Path, frames: list[int]) -> None:
    with (output / "frame_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_id", "source_frame", "selection_reason", "quality_notes"),
        )
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "frame_id": f"{frame:06d}",
                    "source_frame": frame,
                    "selection_reason": DEFAULT_REASONS.get(frame, "representative pose"),
                    "quality_notes": "visually checked in four camera views",
                }
            )


def write_checksums(output: Path) -> None:
    checksum_path = output / "CONTENTS.sha256"
    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {path.relative_to(output).as_posix()}")
    checksum_path.write_text("\n".join(records) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    frames = parse_frames(args.frames)
    if not (source / "metadata.json").is_file():
        raise FileNotFoundError(source / "metadata.json")
    output.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        source_frame = source / f"frame_{frame:06d}"
        if not source_frame.is_dir():
            raise FileNotFoundError(source_frame)
        shutil.copytree(source_frame, output / source_frame.name, dirs_exist_ok=True)

    for name in ("core23_vertex_support_map.pkl", "core23_vertex_support_map.audit.json"):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / name)

    metadata = sanitize_metadata(source, output, frames)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_manifest(output, frames)
    make_contact_sheet(source, frames, output / "representative_frames_contact_sheet.jpg")
    write_checksums(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "source": str(source),
                "frames": frames,
                "views_per_frame": len(view_ids(source / f"frame_{frames[0]:06d}")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

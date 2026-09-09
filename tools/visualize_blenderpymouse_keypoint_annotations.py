#!/usr/bin/env python3
"""Create stratified projection overlays for BlenderPyMouse keypoint annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MAJOR_LABELS = {"Wrist_L", "Wrist_R", "Fingers1_L", "Fingers1_R", "HeadEnd_M"}
COLORS = {
    "L": (0, 235, 255),
    "R": (255, 60, 220),
    "M": (255, 225, 40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--views-per-frame", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--thumbnail-width", type=int, default=480)
    parser.add_argument("--contact-columns", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative_path(root: Path, text: str) -> Path:
    return root / Path(text.replace("\\", "/"))


def side_for_name(name: str) -> str:
    if name.endswith("_L"):
        return "L"
    if name.endswith("_R"):
        return "R"
    return "M"


def point_map(keypoints: list[dict]) -> dict[str, dict]:
    return {str(item["name"]): item for item in keypoints}


def hand_edges(side: str) -> list[tuple[str, str]]:
    palm = f"Fingers1_{side}"
    edges = [(f"Wrist_{side}", palm)]
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        chain = [palm] + [f"{finger}Finger{segment}_{side}" for segment in (1, 2, 3)]
        edges.extend(zip(chain[:-1], chain[1:]))
    return edges


def drawable_xy(item: dict) -> tuple[float, float] | None:
    uv = item.get("uv")
    if not uv or len(uv) != 2:
        return None
    x, y = float(uv[0]), float(uv[1])
    if not item.get("in_frame", False):
        return None
    return x, y


def draw_overlay(image_path: Path, annotation: dict, title: str, output_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    points = point_map(annotation.get("keypoints", []))

    for side in ("L", "R"):
        color = COLORS[side]
        for first_name, second_name in hand_edges(side):
            first = drawable_xy(points.get(first_name, {}))
            second = drawable_xy(points.get(second_name, {}))
            if first is None or second is None:
                continue
            both_visible = points[first_name].get("visible") and points[second_name].get("visible")
            line_color = color if both_visible else tuple(channel // 2 for channel in color)
            draw.line((first, second), fill=line_color, width=2)

    visible_count = 0
    in_frame_count = 0
    for name, item in points.items():
        xy = drawable_xy(item)
        if xy is None:
            continue
        in_frame_count += 1
        visible = bool(item.get("visible"))
        visible_count += int(visible)
        color = COLORS[side_for_name(name)]
        x, y = xy
        radius = 5 if name in MAJOR_LABELS else 3
        if visible:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0))
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=(255, 140, 40), width=2)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=(255, 140, 40), width=2)
        if name in MAJOR_LABELS:
            draw.text((x + 7, y - 8), name, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    header_height = 28
    draw.rectangle((0, 0, image.width, header_height), fill=(0, 0, 0))
    draw.text(
        (8, 8),
        f"{title}  visible={visible_count}/{in_frame_count}",
        fill=(255, 255, 255),
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "image": str(image_path.resolve()),
        "overlay": str(output_path.resolve()),
        "visible": visible_count,
        "in_frame": in_frame_count,
        "keypoints": len(points),
    }


def select_samples(
    datasets_root: Path,
    samples_per_dataset: int,
    views_per_frame: int,
    rng: random.Random,
) -> list[tuple[Path, Path, Path]]:
    selected = []
    for dataset_root in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        frame_roots = sorted(
            (
                path
                for path in dataset_root.glob("frame_*")
                if (path / "keypoints.npz").is_file() and (path / "scene_meta.json").is_file()
            ),
            key=lambda path: path.name,
        )
        if not frame_roots:
            continue
        chosen_frames = rng.sample(frame_roots, min(samples_per_dataset, len(frame_roots)))
        for frame_root in chosen_frames:
            annotations = sorted((frame_root / "annotations").glob("*.json"))
            annotated = [path for path in annotations if load_json(path).get("keypoints")]
            chosen_views = rng.sample(annotated, min(views_per_frame, len(annotated)))
            for annotation_path in chosen_views:
                annotation = load_json(annotation_path)
                image_path = relative_path(frame_root, annotation["image"])
                selected.append((dataset_root, frame_root, annotation_path))
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
    return selected


def make_contact_sheet(
    overlays: list[Path],
    output_path: Path,
    thumbnail_width: int,
    columns: int,
) -> None:
    thumbnails = []
    for path in overlays:
        image = Image.open(path).convert("RGB")
        height = round(image.height * thumbnail_width / image.width)
        thumbnails.append(image.resize((thumbnail_width, height), Image.Resampling.LANCZOS))
    if not thumbnails:
        raise ValueError("No overlays available for contact sheet")
    rows = (len(thumbnails) + columns - 1) // columns
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new("RGB", (columns * thumbnail_width, rows * cell_height), (24, 24, 24))
    for index, image in enumerate(thumbnails):
        x = (index % columns) * thumbnail_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    datasets_root = args.datasets_root.resolve()
    output_dir = args.output_dir.resolve()
    rng = random.Random(args.seed)
    samples = select_samples(
        datasets_root,
        args.samples_per_dataset,
        args.views_per_frame,
        rng,
    )

    records = []
    overlay_paths = []
    for dataset_root, frame_root, annotation_path in samples:
        annotation = load_json(annotation_path)
        view_id = int(annotation["view_id"])
        image_path = relative_path(frame_root, annotation["image"])
        relative_output = Path(dataset_root.name) / f"{frame_root.name}_view_{view_id:02d}.png"
        output_path = output_dir / relative_output
        record = draw_overlay(
            image_path,
            annotation,
            f"{dataset_root.name} / {frame_root.name} / {annotation['camera']}",
            output_path,
        )
        record.update(
            {
                "dataset": dataset_root.name,
                "frame": int(annotation["frame"]),
                "view_id": view_id,
                "camera": annotation["camera"],
            }
        )
        records.append(record)
        overlay_paths.append(output_path)

    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(
        overlay_paths,
        contact_sheet,
        args.thumbnail_width,
        args.contact_columns,
    )
    manifest = {
        "seed": args.seed,
        "sample_count": len(records),
        "contact_sheet": str(contact_sheet),
        "samples": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"keypoint QA overlays: {len(records)}")
    print(f"contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()

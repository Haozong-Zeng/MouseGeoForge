#!/usr/bin/env python3
"""Create fixed-frame, all-camera Core-23 projection QA sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "axial": (255, 220, 35),
    "tail": (255, 145, 35),
    "paw": (20, 220, 210),
    "limb_joint": (245, 55, 120),
    "bottom": (135, 100, 255),
}
EDGES = [
    ("nose", "head_center"),
    ("head_center", "neck"),
    ("neck", "spine_center"),
    ("spine_center", "lumbar_spine"),
    ("lumbar_spine", "tail_base"),
    ("tail_base", "tail_q1"),
    ("tail_q1", "tail_q2"),
    ("tail_q2", "tail_q3"),
    ("tail_q3", "tail_q4"),
    ("tail_q4", "tail_q5"),
    ("tail_q5", "tail_tip"),
    ("left_elbow", "left_forepaw_center"),
    ("right_elbow", "right_forepaw_center"),
    ("left_knee", "left_hindpaw_center"),
    ("right_knee", "right_hindpaw_center"),
]
LABELS = {
    "nose": "N",
    "head_center": "Hd",
    "left_ear": "LEar",
    "right_ear": "REar",
    "neck": "Ne",
    "spine_center": "Sp",
    "lumbar_spine": "Lu",
    "tail_base": "T0",
    "tail_tip": "T6",
    "left_forepaw_center": "LFP",
    "right_forepaw_center": "RFP",
    "left_hindpaw_center": "LHP",
    "right_hindpaw_center": "RHP",
    "left_elbow": "LE",
    "right_elbow": "RE",
    "left_knee": "LK",
    "right_knee": "RK",
    "abdomen_center": "Ab",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="1,31,61,91")
    parser.add_argument("--thumbnail", type=int, default=384)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_path(scene_root: Path, text: str) -> Path:
    return scene_root / Path(text.replace("\\", "/"))


def xy(item: dict) -> tuple[float, float] | None:
    uv = item.get("uv")
    if not item.get("in_frame") or not uv:
        return None
    return float(uv[0]), float(uv[1])


def annotate(scene_root: Path, annotation: dict, output: Path) -> None:
    image = Image.open(image_path(scene_root, annotation["image"])).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    points = {item["name"]: item for item in annotation["keypoints"]}
    for first_name, second_name in EDGES:
        first = xy(points[first_name])
        second = xy(points[second_name])
        if first is None or second is None:
            continue
        draw.line((first, second), fill=(85, 255, 100), width=2)
    visible = 0
    in_frame = 0
    for name, item in points.items():
        point = xy(item)
        if point is None:
            continue
        in_frame += 1
        is_visible = bool(item.get("surface_visible", item.get("visible", False)))
        visible += int(is_visible)
        x, y = point
        color = COLORS.get(item.get("group"), (255, 255, 255))
        radius = 4
        if is_visible:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=1)
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)
        label = LABELS.get(name)
        if label:
            draw.text((x + 5, y - 6), label, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    draw.rectangle((0, 0, image.width, 26), fill=(0, 0, 0))
    draw.text(
        (7, 7),
        f"frame {annotation['frame']:03d} | {annotation['camera']} | visible {visible}/{in_frame}",
        fill=(255, 255, 255),
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def contact_sheet(paths: list[Path], output: Path, thumbnail: int) -> None:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        images.append(image.resize((thumbnail, thumbnail), Image.Resampling.LANCZOS))
    columns = min(8, max(1, len(images)))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumbnail, rows * thumbnail), (20, 20, 20))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * thumbnail, (index // columns) * thumbnail))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def main() -> None:
    args = parse_args()
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    for dataset in args.dataset:
        dataset = dataset.resolve()
        overlays = []
        for frame in frames:
            scene_root = dataset / f"frame_{frame:06d}"
            for annotation_path in sorted((scene_root / "annotations").glob("*.json")):
                annotation = load_json(annotation_path)
                output = args.output_dir.resolve() / dataset.name / f"frame_{frame:06d}_{annotation_path.stem}.png"
                annotate(scene_root, annotation, output)
                overlays.append(output)
        sheet = args.output_dir.resolve() / f"{dataset.name}_core23_all_cameras_contact_sheet.jpg"
        contact_sheet(overlays, sheet, args.thumbnail)
        print(f"contact_sheet={sheet}")


if __name__ == "__main__":
    main()

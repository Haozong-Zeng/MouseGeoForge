#!/usr/bin/env python3
"""Overlay limb joints on WalkAround renders and compose review artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LEFT = (35, 166, 213)
RIGHT = (238, 112, 43)
CORE = (225, 37, 93)
CANDIDATE = (25, 170, 103)
TEXT = (250, 250, 250)
PANEL = (18, 18, 20, 218)

LABELS = {
    "left_shoulder": "L-S",
    "left_elbow": "L-E",
    "left_wrist": "L-W",
    "right_shoulder": "R-S",
    "right_elbow": "R-E",
    "right_wrist": "R-W",
    "left_hip": "L-H",
    "left_knee": "L-K",
    "left_ankle": "L-A",
    "right_hip": "R-H",
    "right_knee": "R-K",
    "right_ankle": "R-A",
}

CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)

CORE_NAMES = {"left_elbow", "right_elbow", "left_knee", "right_knee"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def point(record: dict, name: str) -> tuple[float, float]:
    value = record["joints"][name]["pixel"]
    return float(value[0]), float(value[1])


def marker_color(name: str, extended: bool) -> tuple[int, int, int]:
    if not extended:
        return LEFT if name.startswith("left_") else RIGHT
    return CORE if name in CORE_NAMES else CANDIDATE


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, color: tuple[int, int, int], font) -> None:
    x, y = xy
    tx, ty = x + 10, y - 24
    box = draw.textbbox((tx, ty), label, font=font, stroke_width=1)
    draw.rounded_rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), radius=3, fill=PANEL)
    draw.text((tx, ty), label, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))


def annotate(record: dict, extended: bool, output_path: Path) -> None:
    image = Image.open(record["image"]).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(18, bold=True)
    title_font = load_font(22, bold=True)
    names = set(LABELS) if extended else CORE_NAMES

    if extended:
        for chain in CHAINS:
            color = LEFT if chain[0].startswith("left_") else RIGHT
            coords = [point(record, name) for name in chain]
            draw.line(coords, fill=(*color, 210), width=5, joint="curve")

    for name in LABELS:
        if name not in names or not record["joints"][name]["in_frame"]:
            continue
        x, y = point(record, name)
        color = marker_color(name, extended)
        radius = 10 if name in CORE_NAMES else 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 235), outline=(255, 255, 255, 255), width=3)
        draw_label(draw, (x, y), LABELS[name], color, font)

    title = "Core-4: elbows + knees" if not extended else "Extended candidates: shoulders / hips / wrists / ankles"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.rounded_rectangle((14, 12, title_box[2] + 32, title_box[3] + 27), radius=5, fill=PANEL)
    draw.text((23, 18), title, fill=TEXT, font=title_font)
    frame_text = f"frame {record['frame']:03d} | {record['view']}"
    draw.text((23, 48), frame_text, fill=(225, 225, 225), font=font, stroke_width=1, stroke_fill=(0, 0, 0))

    if extended:
        legend = "pink: Core joint   green: candidate   blue/orange: L/R chain"
        draw.rounded_rectangle((14, image.height - 42, 535, image.height - 10), radius=5, fill=PANEL)
        draw.text((23, image.height - 37), legend, fill=TEXT, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def side_by_side(left_path: Path, right_path: Path, output_path: Path) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (245, 245, 245))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def contact_sheet(paths: list[Path], output_path: Path, columns: int = 2) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_w * columns, cell_h * rows), (30, 30, 32))
    for index, image in enumerate(images):
        canvas.paste(image, ((index % columns) * cell_w, (index // columns) * cell_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = payload["records"]
    by_view: dict[str, list[dict]] = {}
    for record in records:
        by_view.setdefault(record["view"], []).append(record)

    comparison_paths = []
    for record in sorted(by_view.get("top", []), key=lambda item: item["frame"]):
        frame = int(record["frame"])
        core_path = output_dir / "annotated" / "top_core" / f"frame_{frame:04d}.png"
        extended_path = output_dir / "annotated" / "top_extended" / f"frame_{frame:04d}.png"
        compare_path = output_dir / "comparison_frames" / f"frame_{frame:04d}.png"
        annotate(record, False, core_path)
        annotate(record, True, extended_path)
        side_by_side(core_path, extended_path, compare_path)
        comparison_paths.append(compare_path)

    for view_name in ("left_oblique", "right_oblique"):
        annotated_paths = []
        for record in sorted(by_view.get(view_name, []), key=lambda item: item["frame"]):
            path = output_dir / "annotated" / view_name / f"frame_{int(record['frame']):04d}.png"
            annotate(record, True, path)
            annotated_paths.append(path)
        contact_sheet(annotated_paths, output_dir / f"{view_name}_extended_contact_sheet.jpg", columns=2)

    if comparison_paths:
        selected = [comparison_paths[index] for index in [0, len(comparison_paths) // 3, 2 * len(comparison_paths) // 3, len(comparison_paths) - 1]]
        contact_sheet(selected, output_dir / "top_core_vs_extended_contact_sheet.jpg", columns=1)
        video_path = output_dir / "walkaround_core_vs_extended_top.mp4"
        ffmpeg = shutil.which(args.ffmpeg)
        if ffmpeg is None:
            print(f"video_skipped=ffmpeg executable not found: {args.ffmpeg}")
        else:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(args.fps),
                    "-start_number",
                    str(payload["frame_range"][0]),
                    "-i",
                    str(output_dir / "comparison_frames" / "frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    str(video_path),
                ],
                check=True,
            )
            print(f"video_written={video_path}")


if __name__ == "__main__":
    main()

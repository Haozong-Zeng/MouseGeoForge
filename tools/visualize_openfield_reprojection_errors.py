#!/usr/bin/env python3
"""Visualize legacy-to-Blender keypoint reprojection errors for Openfield data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LEGACY_NAMES = (
    "nose",
    "head_center",
    "left_ear",
    "right_ear",
    "neck",
    "spine_center",
    "lumbar_spine",
    "tail_base",
    "tail_q3",
    "tail_tip",
)
LEGACY_COLOR = (0, 235, 255)
PROJECTED_COLOR = (255, 45, 185)
VECTOR_COLOR = (255, 225, 30)
TEXT_COLOR = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-points", type=int, default=24)
    parser.add_argument("--top-frames", type=int, default=12)
    parser.add_argument("--zoom-radius", type=int, default=28)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_markers(
    image: Image.Image,
    legacy_xy: tuple[float, float],
    projected_xy: tuple[float, float],
    radius: int,
    width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    lx, ly = legacy_xy
    px, py = projected_xy
    draw.line((lx, ly, px, py), fill=VECTOR_COLOR, width=width)
    draw.ellipse((lx - radius, ly - radius, lx + radius, ly + radius), outline=LEGACY_COLOR, width=width)
    draw.line((px - radius, py, px + radius, py), fill=PROJECTED_COLOR, width=width)
    draw.line((px, py - radius, px, py + radius), fill=PROJECTED_COLOR, width=width)


def collect_records(root: Path) -> tuple[list[dict], list[dict], dict]:
    metadata = load_json(root / "metadata.json")
    point_records = []
    frame_records = []
    for frame in metadata["frame_numbers"]:
        scene_root = root / f"frame_{frame:06d}"
        annotation = load_json(scene_root / "annotations" / "00000000.json")
        width, height = Image.open(scene_root / "blended_images" / "00000000.jpg").size
        legacy = np.asarray(annotation["legacy_keypoints_normalized"], dtype=np.float64)
        legacy *= np.asarray((width, height), dtype=np.float64)
        indices = annotation["legacy_keypoint_core23_indices"]
        projected = np.asarray([annotation["keypoints"][index]["uv"] for index in indices], dtype=np.float64)
        vectors = projected - legacy
        errors = np.linalg.norm(vectors, axis=1)
        for legacy_index, (legacy_xy, projected_xy, vector, error) in enumerate(
            zip(legacy, projected, vectors, errors)
        ):
            point_records.append(
                {
                    "frame": int(frame),
                    "legacy_index": legacy_index,
                    "name": LEGACY_NAMES[legacy_index],
                    "core23_index": int(indices[legacy_index]),
                    "legacy_x": float(legacy_xy[0]),
                    "legacy_y": float(legacy_xy[1]),
                    "projected_x": float(projected_xy[0]),
                    "projected_y": float(projected_xy[1]),
                    "dx": float(vector[0]),
                    "dy": float(vector[1]),
                    "error_px": float(error),
                    "image": str(scene_root / "blended_images" / "00000000.jpg"),
                }
            )
        max_index = int(np.argmax(errors))
        frame_records.append(
            {
                "frame": int(frame),
                "mean_error_px": float(errors.mean()),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(errors[max_index]),
                "max_error_name": LEGACY_NAMES[max_index],
                "image": str(scene_root / "blended_images" / "00000000.jpg"),
            }
        )
    return point_records, frame_records, metadata


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def make_point_sample(record: dict, zoom_radius: int) -> Image.Image:
    source = Image.open(record["image"]).convert("RGB")
    legacy_xy = (record["legacy_x"], record["legacy_y"])
    projected_xy = (record["projected_x"], record["projected_y"])

    context = source.copy()
    draw_markers(context, legacy_xy, projected_xy, radius=7, width=3)
    context.thumbnail((350, 350), Image.Resampling.LANCZOS)

    cx = (legacy_xy[0] + projected_xy[0]) * 0.5
    cy = (legacy_xy[1] + projected_xy[1]) * 0.5
    radius = max(zoom_radius, int(math.ceil(record["error_px"] * 2.5)))
    crop_box = (
        int(round(cx - radius)),
        int(round(cy - radius)),
        int(round(cx + radius)),
        int(round(cy + radius)),
    )
    marked = source.copy()
    draw_markers(marked, legacy_xy, projected_xy, radius=2, width=1)
    zoom = marked.crop(crop_box).resize((350, 350), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (760, 430), (22, 22, 24))
    canvas.paste(context, (10, 64))
    canvas.paste(zoom, (400, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"frame {record['frame']:03d} | {record['name']} | {record['error_px']:.2f} px",
        font=font(24),
        fill=TEXT_COLOR,
    )
    draw.text((12, 38), "full-frame context", font=font(16), fill=(180, 180, 180))
    draw.text((402, 38), f"local {radius * 2}x{radius * 2} px crop", font=font(16), fill=(180, 180, 180))
    draw.text(
        (402, 400),
        f"vector: dx={record['dx']:+.2f}, dy={record['dy']:+.2f} px",
        font=font(16),
        fill=TEXT_COLOR,
    )
    return canvas


def make_contact_sheet(images: list[Image.Image], title: str, columns: int = 2) -> Image.Image:
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    header = 74
    sheet = Image.new("RGB", (tile_width * columns, header + tile_height * rows), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), title, font=font(28), fill=TEXT_COLOR)
    draw.text(
        (16, 45),
        "cyan circle: legacy 2D | magenta cross: Blender projection | yellow: error vector",
        font=font(17),
        fill=(195, 195, 195),
    )
    for index, image in enumerate(images):
        x = (index % columns) * tile_width
        y = header + (index // columns) * tile_height
        sheet.paste(image, (x, y))
    return sheet


def make_frame_panel(root: Path, frame_record: dict, frame_points: list[dict]) -> Image.Image:
    source = Image.open(frame_record["image"]).convert("RGB")
    for record in frame_points:
        draw_markers(
            source,
            (record["legacy_x"], record["legacy_y"]),
            (record["projected_x"], record["projected_y"]),
            radius=6,
            width=2,
        )
    source.thumbnail((500, 500), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (520, 560), (22, 22, 24))
    panel.paste(source, ((520 - source.width) // 2, 48))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (10, 8),
        f"frame {frame_record['frame']:03d} | mean {frame_record['mean_error_px']:.2f} px | "
        f"max {frame_record['max_error_px']:.2f} px ({frame_record['max_error_name']})",
        font=font(18),
        fill=TEXT_COLOR,
    )
    return panel


def make_distribution_plot(point_records: list[dict], output: Path) -> None:
    errors = np.asarray([record["error_px"] for record in point_records])
    canvas = Image.new("RGB", (1600, 720), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), "Reprojection error distribution", font=font(32), fill=(25, 25, 25))

    hist_left, hist_top, hist_right, hist_bottom = 70, 100, 760, 640
    counts, edges = np.histogram(errors, bins=36, range=(0.0, float(math.ceil(errors.max()))))
    max_count = int(counts.max())
    draw.rectangle((hist_left, hist_top, hist_right, hist_bottom), outline=(70, 70, 70), width=2)
    bar_width = (hist_right - hist_left) / len(counts)
    for index, count in enumerate(counts):
        x0 = hist_left + index * bar_width
        x1 = hist_left + (index + 1) * bar_width - 1
        y0 = hist_bottom - (hist_bottom - hist_top) * count / max_count
        draw.rectangle((x0, y0, x1, hist_bottom), fill=(22, 199, 206), outline=(35, 60, 60))
    max_error = float(edges[-1])
    p95 = float(np.percentile(errors, 95))
    p95_x = hist_left + (hist_right - hist_left) * p95 / max_error
    draw.line((p95_x, hist_top, p95_x, hist_bottom), fill=PROJECTED_COLOR, width=4)
    draw.text((p95_x + 6, hist_top + 8), f"p95={p95:.2f}px", font=font(17), fill=(180, 20, 110))
    for value in np.linspace(0.0, max_error, 6):
        x = hist_left + (hist_right - hist_left) * value / max_error
        draw.line((x, hist_bottom, x, hist_bottom + 7), fill=(50, 50, 50), width=2)
        draw.text((x - 10, hist_bottom + 12), f"{value:.1f}", font=font(15), fill=(40, 40, 40))
    draw.text((hist_left + 220, 678), "Reprojection error (pixels)", font=font(19), fill=(30, 30, 30))
    draw.text((hist_left, 70), f"All {len(errors):,} keypoint instances", font=font(20), fill=(40, 40, 40))

    grouped = defaultdict(list)
    for record in point_records:
        grouped[record["name"]].append(record["error_px"])
    box_left, box_top, box_right, box_bottom = 980, 105, 1540, 645
    draw.text((840, 70), "Error by keypoint", font=font(20), fill=(40, 40, 40))
    draw.rectangle((box_left, box_top, box_right, box_bottom), outline=(70, 70, 70), width=2)
    for value in np.linspace(0.0, max_error, 6):
        x = box_left + (box_right - box_left) * value / max_error
        draw.line((x, box_top, x, box_bottom), fill=(215, 215, 215), width=1)
        draw.text((x - 10, box_bottom + 12), f"{value:.1f}", font=font(15), fill=(40, 40, 40))
    row_height = (box_bottom - box_top) / len(LEGACY_NAMES)
    for index, name in enumerate(LEGACY_NAMES):
        values = np.asarray(grouped[name])
        low, q1, median, q3, high = np.percentile(values, (0, 25, 50, 75, 100))
        y = box_top + (index + 0.5) * row_height

        def scale(value: float) -> float:
            return box_left + (box_right - box_left) * value / max_error

        draw.text((805, y - 9), name, font=font(15), fill=(35, 35, 35))
        draw.line((scale(low), y, scale(high), y), fill=(65, 65, 65), width=2)
        draw.rectangle((scale(q1), y - 10, scale(q3), y + 10), fill=(22, 199, 206), outline=(35, 60, 60), width=2)
        draw.line((scale(median), y - 11, scale(median), y + 11), fill=PROJECTED_COLOR, width=3)
    draw.text((1140, 678), "Reprojection error (pixels)", font=font(19), fill=(30, 30, 30))
    canvas.save(output)


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    output = args.output_dir.resolve()
    samples = output / "largest_point_samples"
    samples.mkdir(parents=True, exist_ok=True)

    point_records, frame_records, metadata = collect_records(root)
    point_ranked = sorted(point_records, key=lambda row: row["error_px"], reverse=True)
    frame_ranked = sorted(frame_records, key=lambda row: row["max_error_px"], reverse=True)

    write_csv(
        output / "all_point_errors.csv",
        point_ranked,
        [
            "frame",
            "legacy_index",
            "name",
            "core23_index",
            "legacy_x",
            "legacy_y",
            "projected_x",
            "projected_y",
            "dx",
            "dy",
            "error_px",
            "image",
        ],
    )
    write_csv(
        output / "frame_error_ranking.csv",
        frame_ranked,
        ["frame", "mean_error_px", "median_error_px", "max_error_px", "max_error_name", "image"],
    )

    per_keypoint = []
    for name in LEGACY_NAMES:
        values = np.asarray([record["error_px"] for record in point_records if record["name"] == name])
        per_keypoint.append(
            {
                "name": name,
                "count": len(values),
                "mean_error_px": float(values.mean()),
                "median_error_px": float(np.median(values)),
                "p95_error_px": float(np.percentile(values, 95)),
                "max_error_px": float(values.max()),
            }
        )
    write_csv(
        output / "per_keypoint_summary.csv",
        per_keypoint,
        ["name", "count", "mean_error_px", "median_error_px", "p95_error_px", "max_error_px"],
    )

    point_images = []
    for rank, record in enumerate(point_ranked[: args.top_points], start=1):
        image = make_point_sample(record, args.zoom_radius)
        image.save(samples / f"rank_{rank:02d}_frame_{record['frame']:06d}_{record['name']}.jpg", quality=95)
        point_images.append(image)
    make_contact_sheet(point_images, "Largest individual reprojection errors").save(
        output / "largest_point_errors_contact_sheet.jpg", quality=95
    )

    points_by_frame = defaultdict(list)
    for record in point_records:
        points_by_frame[record["frame"]].append(record)
    frame_panels = [
        make_frame_panel(root, record, points_by_frame[record["frame"]])
        for record in frame_ranked[: args.top_frames]
    ]
    make_contact_sheet(frame_panels, "Frames with the largest maximum reprojection errors", columns=3).save(
        output / "largest_frame_errors_contact_sheet.jpg", quality=95
    )
    make_distribution_plot(point_records, output / "error_distribution.png")

    errors = np.asarray([record["error_px"] for record in point_records])
    vectors = np.asarray([[record["dx"], record["dy"]] for record in point_records])
    summary = {
        "dataset": metadata["dataset_name"],
        "frame_count": len(frame_records),
        "point_instance_count": len(point_records),
        "error_px": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99)),
            "max": float(errors.max()),
        },
        "mean_signed_vector_px": {"dx": float(vectors[:, 0].mean()), "dy": float(vectors[:, 1].mean())},
        "largest_point_errors": point_ranked[: args.top_points],
        "largest_frame_errors": frame_ranked[: args.top_frames],
        "per_keypoint": per_keypoint,
        "legend": {
            "legacy_2d": "cyan circle",
            "blender_projection": "magenta cross",
            "error_vector": "yellow line",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["error_px"], indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()

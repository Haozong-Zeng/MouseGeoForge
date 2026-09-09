#!/usr/bin/env python3
"""Create a mouse-scale comparison sheet for C57 ear material variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def padded_box(mask: Image.Image, factor: float = 0.22) -> tuple[int, int, int, int]:
    box = mask.convert("L").getbbox()
    if box is None:
        return (0, 0, mask.width, mask.height)
    left, top, right, bottom = box
    pad = int(max(right - left, bottom - top) * factor)
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(mask.width, right + pad),
        min(mask.height, bottom + pad),
    )


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, (45, 45, 45))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = ["baseline", "subtle", "balanced", "warm"]
    tile_size = (480, 360)
    label_height = 42
    sheet = Image.new("RGB", (tile_size[0] * 4, (tile_size[1] + label_height) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    view_names = {1: "side_neg_x", 2: "side_neg_y", 3: "side_pos_y"}
    for row, view_id in enumerate((1, 2, 3)):
        mask = Image.open(
            args.dataset_root / "frame_000001" / "masks" / f"{view_id:08d}.png"
        )
        box = padded_box(mask)
        for column, label in enumerate(labels):
            if label == "baseline":
                source = (
                    args.dataset_root
                    / "frame_000001"
                    / "blended_images"
                    / f"{view_id:08d}.jpg"
                )
            else:
                source = args.variants_root / label / f"{view_names[view_id]}.png"
            crop = Image.open(source).convert("RGB").crop(box)
            x = column * tile_size[0]
            y = row * (tile_size[1] + label_height)
            sheet.paste(fit(crop, tile_size), (x, y))
            draw.text((x + 8, y + tile_size[1] + 7), f"{label} / view {view_id}", fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=96)

    head_boxes = {
        1: (535, 330, 660, 470),
        2: (675, 325, 820, 475),
        3: (435, 330, 585, 480),
    }
    head_tile = (420, 420)
    head_sheet = Image.new(
        "RGB", (head_tile[0] * 4, (head_tile[1] + label_height) * 3), "white"
    )
    head_draw = ImageDraw.Draw(head_sheet)
    for row, view_id in enumerate((1, 2, 3)):
        for column, label in enumerate(labels):
            if label == "baseline":
                source = (
                    args.dataset_root
                    / "frame_000001"
                    / "blended_images"
                    / f"{view_id:08d}.jpg"
                )
            else:
                source = args.variants_root / label / f"{view_names[view_id]}.png"
            crop = Image.open(source).convert("RGB").crop(head_boxes[view_id])
            x = column * head_tile[0]
            y = row * (head_tile[1] + label_height)
            head_sheet.paste(fit(crop, head_tile), (x, y))
            head_draw.text(
                (x + 8, y + head_tile[1] + 7),
                f"{label} / ear view {view_id}",
                fill="black",
                font=font,
            )
    head_sheet.save(args.output.with_name(args.output.stem + "_head_crops.jpg"), quality=96)


if __name__ == "__main__":
    main()

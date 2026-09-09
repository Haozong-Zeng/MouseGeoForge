#!/usr/bin/env python3
"""Build close-crop contact sheets for Homecage ear-material review."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def crop_from_mask(image: Image.Image, mask: Image.Image, padding: float = 0.35) -> Image.Image:
    box = mask.convert("L").getbbox()
    if box is None:
        return image.copy()
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    pad = int(max(width, height) * padding)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    return image.crop((left, top, right, bottom))


def fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, (55, 55, 55))
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    tile.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return tile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=22)
    tile_size = (512, 384)
    label_height = 38
    for view_id in (1, 2, 3):
        sheet = Image.new(
            "RGB",
            (tile_size[0] * 2, (tile_size[1] + label_height) * ((len(args.frames) + 1) // 2)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(args.frames):
            root = args.dataset_root / f"frame_{frame:06d}"
            image = Image.open(root / "blended_images" / f"{view_id:08d}.jpg").convert("RGB")
            mask = Image.open(root / "masks" / f"{view_id:08d}.png")
            tile = fit_tile(crop_from_mask(image, mask), tile_size)
            column = index % 2
            row = index // 2
            x = column * tile_size[0]
            y = row * (tile_size[1] + label_height)
            sheet.paste(tile, (x, y))
            draw.text((x + 8, y + tile_size[1] + 6), f"frame {frame:03d} / view {view_id}", fill="black", font=font)
        sheet.save(args.output_dir / f"ear_review_view{view_id}.jpg", quality=95)


if __name__ == "__main__":
    main()

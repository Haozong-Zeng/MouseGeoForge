#!/usr/bin/env python3
"""Create RGB, depth, mask, and Core-23 overlay sheets for one dataset frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--frame", type=int, default=1)
    return parser.parse_args()


def make_sheet(images, labels, output, tile_size=320):
    columns = 4
    label_height = 28
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, columns)
        x = column * tile_size
        y = row * (tile_size + label_height)
        image = image.convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (tile_size - image.width) // 2, y + label_height))
        draw.text((x + 8, y + 8), label, fill="black", font=font)
    sheet.save(output, quality=94, subsampling=0)


def main():
    args = parse_args()
    root = args.dataset_root.resolve()
    frame_root = root / f"frame_{args.frame:06d}"
    keypoints = json.loads((frame_root / "keypoints.json").read_text(encoding="utf-8"))
    views = keypoints["views"]
    labels = [f"view {index:02d} | {view['camera']}" for index, view in enumerate(views)]

    rgb = []
    depth = []
    masks = []
    overlays = []
    for index, view in enumerate(views):
        stem = f"{index:08d}"
        source = Image.open(frame_root / "blended_images" / f"{stem}.jpg").convert("RGB")
        rgb.append(source.copy())
        depth.append(Image.open(frame_root / "rendered_depth_maps" / f"{stem}_preview.png"))
        masks.append(Image.open(frame_root / "masks" / f"{stem}.png"))
        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        for keypoint in view["keypoints"]:
            if not keypoint.get("in_frame", False):
                continue
            x, y = keypoint["uv"]
            color = (30, 220, 80) if keypoint.get("visible", False) else (255, 170, 30)
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        overlays.append(overlay)

    make_sheet(rgb, labels, root / "pilot_rgb_sheet.jpg")
    make_sheet(depth, labels, root / "pilot_depth_sheet.jpg")
    make_sheet(masks, labels, root / "pilot_mask_sheet.jpg")
    make_sheet(overlays, labels, root / "pilot_core23_overlay_sheet.jpg")
    print(root / "pilot_rgb_sheet.jpg")
    print(root / "pilot_depth_sheet.jpg")
    print(root / "pilot_mask_sheet.jpg")
    print(root / "pilot_core23_overlay_sheet.jpg")


if __name__ == "__main__":
    main()

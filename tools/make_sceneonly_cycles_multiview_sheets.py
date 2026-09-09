#!/usr/bin/env python3
"""Build full-frame and mouse-detail sheets from SceneOnly Cycles renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Normalized camera-view bounds from Blender's evaluated frame-100 mouse bbox.
# Coordinates use Blender camera convention: origin at bottom left.
MOUSE_BOUNDS = {
    "Camera.012": ((0.4102, 0.2602), (0.5898, 0.7398)),
    "Camera.013": ((0.3311, 0.1888), (0.6664, 0.4489)),
    "Camera.014": ((0.3336, 0.1888), (0.6689, 0.4489)),
    "Camera.015": ((0.4156, 0.1824), (0.5844, 0.4524)),
    "Camera.016": ((0.3221, 0.1466), (0.6750, 0.3478)),
    "Camera.017": ((0.3250, 0.1466), (0.6779, 0.3478)),
    "Camera.018": ((0.4315, 0.2282), (0.5685, 0.3848)),
    "Camera.019": ((0.4475, 0.3948), (0.5525, 0.6052)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path)
    return parser.parse_args()


def fit(image: Image.Image, size: tuple[int, int], background=(238, 238, 238)) -> Image.Image:
    canvas = Image.new("RGB", size, background)
    source = image.convert("RGB")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
    return canvas


def crop_mouse(image: Image.Image, camera: str, padding: float = 0.10) -> Image.Image:
    (x0, y0), (x1, y1) = MOUSE_BOUNDS[camera]
    x0, x1 = x0 - padding, x1 + padding
    y0, y1 = y0 - padding, y1 + padding
    left = max(0, round(x0 * image.width))
    right = min(image.width, round(x1 * image.width))
    top = max(0, round((1.0 - y1) * image.height))
    bottom = min(image.height, round((1.0 - y0) * image.height))
    return image.crop((left, top, right, bottom))


def sheet(records: list[dict], render_dir: Path, output: Path, detail: bool) -> None:
    columns = 4
    tile = 420 if detail else 360
    label_height = 34
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_height)), (244, 244, 244))
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = column * tile
        y = row * (tile + label_height)
        camera = record["camera"]
        source_path = render_dir / Path(str(record["output"]).replace("\\", "/")).name
        with Image.open(source_path) as image:
            image.load()
            source = crop_mouse(image, camera) if detail else image
            tile_image = fit(source, (tile, tile))
        canvas.paste(tile_image, (x, y + label_height))
        label = f"{camera} | frame 100" + (" | fur detail crop" if detail else "")
        draw.text((x + 10, y + 9), label, fill=(24, 24, 24), font=ImageFont.load_default())
    canvas.save(output, quality=94, subsampling=0)


def main() -> None:
    args = parse_args()
    render_dir = args.render_dir.resolve()
    manifest = json.loads((render_dir / "render_manifest.json").read_text(encoding="utf-8"))
    records = manifest["cameras"]
    sheet(records, render_dir, render_dir / "sceneonly_frame0100_multiview_full.jpg", detail=False)
    sheet(records, render_dir, render_dir / "sceneonly_frame0100_multiview_fur_detail.jpg", detail=True)
    print(render_dir / "sceneonly_frame0100_multiview_full.jpg")
    print(render_dir / "sceneonly_frame0100_multiview_fur_detail.jpg")


if __name__ == "__main__":
    main()

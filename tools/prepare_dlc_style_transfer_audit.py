#!/usr/bin/env python3
"""Prepare matched raw/styled/GT images for DeepLabCut synthetic-data auditing."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--views", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--camera-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help=(
            "Map Blender camera names to DLC views, e.g. "
            "Camera.003=CameraA. Repeat for each camera."
        ),
    )
    parser.add_argument("--image-prep", default="resize_pad_576x384")
    parser.add_argument(
        "--prompt",
        default="a real camera photograph of a laboratory mouse reaching for water",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_camera_map(items: list[str]) -> dict[str, str]:
    mapping = {}
    for item in items:
        source, separator, target = item.partition("=")
        if not separator or not source or not target:
            raise ValueError(f"Invalid --camera-map value: {item!r}")
        mapping[source] = target
    return mapping


def reset_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite")
        resolved = path.resolve()
        if len(resolved.parts) < 5 or resolved == resolved.parent:
            raise ValueError(f"Refuse to overwrite suspicious path: {resolved}")
        shutil.rmtree(resolved)
    for condition in ("raw", "styled", "annotations"):
        (path / condition).mkdir(parents=True, exist_ok=True)


def frame_number(path: Path) -> int:
    return int(path.name.removeprefix("frame_"))


def choose_frames(dataset: Path, requested: list[int] | None, count: int) -> list[Path]:
    frames = sorted(
        (path for path in dataset.glob("frame_*") if path.is_dir()),
        key=frame_number,
    )
    if requested:
        by_number = {frame_number(path): path for path in frames}
        missing = sorted(set(requested) - set(by_number))
        if missing:
            raise FileNotFoundError(f"Requested frames not found: {missing}")
        return [by_number[number] for number in requested]
    if count < 1:
        raise ValueError("--sample-count must be positive")
    if len(frames) <= count:
        return frames
    indices = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    return [frames[index] for index in dict.fromkeys(indices)]


def load_model(args: argparse.Namespace, repository_root: Path):
    turbo_src = repository_root / "img2img-turbo" / "src"
    sys.path.insert(0, str(turbo_src))
    from cyclegan_turbo import CycleGAN_Turbo
    from my_utils.training_utils import build_transform

    model = CycleGAN_Turbo(pretrained_path=str(args.checkpoint))
    model.eval().to(args.device)
    if args.use_fp16:
        model.half()
    transform = build_transform(args.image_prep)
    with torch.no_grad():
        tokens = model.tokenizer(
            args.prompt,
            max_length=model.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(args.device)
        caption_embedding = model.text_encoder(tokens)[0]
        if args.use_fp16:
            caption_embedding = caption_embedding.half()
    return model, transform, caption_embedding


def stylize(
    model,
    transform,
    caption_embedding,
    image: Image.Image,
    device: str,
    use_fp16: bool,
) -> Image.Image:
    prepared = transform(image)
    tensor = transforms.ToTensor()(prepared)
    tensor = transforms.Normalize([0.5], [0.5])(tensor).unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()
    with torch.no_grad():
        output = model(tensor, direction="a2b", caption_emb=caption_embedding)
    styled = transforms.ToPILImage()(output[0].float().cpu() * 0.5 + 0.5)
    return styled.resize(image.size, Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    reset_output(args.output, args.overwrite)
    frames = choose_frames(args.dataset, args.frames, args.sample_count)
    camera_map = parse_camera_map(args.camera_map)
    model, transform, caption_embedding = load_model(args, repository_root)

    rows = []
    for frame_dir in frames:
        frame = frame_number(frame_dir)
        for view_id in args.views:
            annotation_source = frame_dir / "annotations" / f"{view_id:08d}.json"
            if not annotation_source.is_file():
                raise FileNotFoundError(annotation_source)
            annotation = json.loads(annotation_source.read_text(encoding="utf-8"))
            image_source = frame_dir.joinpath(
                *annotation["image"].replace("\\", "/").split("/")
            )
            if not image_source.is_file():
                raise FileNotFoundError(image_source)
            source_camera = annotation["camera"]
            camera = camera_map.get(source_camera, source_camera)
            camera_suffix = camera.removeprefix("Camera")
            stem = f"frame_{frame:06d}_cam{camera_suffix}"
            raw_path = args.output / "raw" / f"{stem}.png"
            styled_path = args.output / "styled" / f"{stem}.png"
            annotation_path = args.output / "annotations" / f"{stem}.json"

            image = Image.open(image_source).convert("RGB")
            image.save(raw_path)
            stylize(
                model,
                transform,
                caption_embedding,
                image,
                args.device,
                args.use_fp16,
            ).save(styled_path)
            shutil.copy2(annotation_source, annotation_path)
            rows.append(
                {
                    "dataset": args.dataset.name,
                    "frame": frame,
                    "view_id": view_id,
                    "camera": camera,
                    "source_camera": source_camera,
                    "raw_image": str(raw_path),
                    "styled_image": str(styled_path),
                    "annotation": str(annotation_path),
                    "width": image.width,
                    "height": image.height,
                }
            )
            print(f"prepared frame={frame} camera={camera}", flush=True)

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "image_prep": args.image_prep,
        "frames": [frame_number(path) for path in frames],
        "views": args.views,
        "camera_map": camera_map,
        "sample_count": len(rows),
        "output": str(args.output),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

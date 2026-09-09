#!/usr/bin/env python3
"""Replace paired synthetic RGB/depth images with later Blender dataset renders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FRAME_PATTERN = re.compile(r"^(?:rgb|depth)_(\d+)$")


@dataclass(frozen=True)
class SequenceConfig:
    label: str
    camera: str
    dataset_name: str
    view_id: int
    resize_mode: str
    crop_x: int | None = None
    crop_width: int | None = None

    @property
    def pair_dir_name(self) -> str:
        return f"water-reaching_{self.label}_cam{self.camera}_FrameToFrame"


SEQUENCES = (
    SequenceConfig(
        label="FG",
        camera="B",
        dataset_name="PyMouseStudio-BrowmFur-WaterReaching-FrameToFrame-FGcamB",
        view_id=1,
        resize_mode="center-square",
    ),
    SequenceConfig(
        label="FS",
        camera="C",
        dataset_name="PyMouseStudio-BrowmFur-WaterReaching-FrameToFrame-FScamC-mapping",
        view_id=2,
        resize_mode="horizontal-window",
        crop_x=240,
        crop_width=925,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def frame_numbers(directory: Path, prefix: str, suffix: str) -> set[int]:
    result = set()
    for path in directory.glob(f"{prefix}_*{suffix}"):
        match = FRAME_PATTERN.match(path.stem)
        if match:
            result.add(int(match.group(1)))
    return result


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header {header!r}: {path}")
        width, height = (int(value) for value in handle.readline().split())
        scale = float(handle.readline())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(handle, dtype=endian + "f4")
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM size mismatch in {path}: {data.size} != {expected}")
    data = np.flipud(data.reshape(height, width, channels))
    return data[..., 0]


def prepare_image(
    image: np.ndarray,
    resize_mode: str,
    interpolation: int,
    crop_x: int | None = None,
    crop_width: int | None = None,
) -> np.ndarray:
    if resize_mode == "full":
        return cv2.resize(image, (512, 512), interpolation=interpolation)
    if resize_mode == "horizontal-window":
        if crop_x is None or crop_width is None:
            raise ValueError("horizontal-window requires crop_x and crop_width")
        if crop_x < 0 or crop_x + crop_width > image.shape[1]:
            raise ValueError(
                f"Horizontal window [{crop_x}, {crop_x + crop_width}) "
                f"exceeds image width {image.shape[1]}"
            )
        crop = image[:, crop_x : crop_x + crop_width]
        return cv2.resize(crop, (512, 512), interpolation=interpolation)
    if resize_mode != "center-square":
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    height, width = image.shape[:2]
    size = min(height, width)
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    crop = image[y0 : y0 + size, x0 : x0 + size]
    return cv2.resize(crop, (512, 512), interpolation=interpolation)


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def fit_depth_encoding(
    synthetic_dir: Path,
    dataset_dir: Path,
    view_id: int,
    resize_mode: str,
    crop_x: int | None,
    crop_width: int | None,
    frames: list[int],
) -> dict[str, float | int]:
    depth_values = []
    grayscale_values = []
    rng = np.random.default_rng(20260722)
    for frame in evenly_spaced(frames, 24):
        old_rgb = cv2.imread(
            str(synthetic_dir / f"rgb_{frame:012d}.jpg"), cv2.IMREAD_COLOR
        )
        old_depth = cv2.imread(
            str(synthetic_dir / f"depth_{frame:012d}.png"), cv2.IMREAD_GRAYSCALE
        )
        frame_dir = dataset_dir / f"frame_{frame:06d}"
        new_rgb = cv2.imread(
            str(frame_dir / "blended_images" / f"{view_id:08d}.jpg"),
            cv2.IMREAD_COLOR,
        )
        depth = read_pfm(
            frame_dir / "rendered_depth_maps" / f"{view_id:08d}.pfm"
        )
        if old_rgb is None or old_depth is None or new_rgb is None:
            continue
        new_rgb = prepare_image(
            new_rgb, resize_mode, cv2.INTER_AREA, crop_x, crop_width
        )
        depth = prepare_image(
            depth, resize_mode, cv2.INTER_NEAREST, crop_x, crop_width
        )

        # Pixels whose RGB barely changed are dominated by the static apparatus.
        # They let us retain the existing grayscale-depth convention while the
        # mouse geometry itself changes between the two render generations.
        rgb_delta = np.max(
            np.abs(new_rgb.astype(np.int16) - old_rgb.astype(np.int16)), axis=2
        )
        valid = (
            (rgb_delta <= 5)
            & np.isfinite(depth)
            & (depth > 0)
            & (old_depth < 245)
        )
        y, x = np.nonzero(valid)
        if len(x) > 30_000:
            selected = rng.choice(len(x), size=30_000, replace=False)
            y = y[selected]
            x = x[selected]
        depth_values.append(depth[y, x].astype(np.float64))
        grayscale_values.append(old_depth[y, x].astype(np.float64))

    if not depth_values:
        raise RuntimeError(f"Could not estimate depth encoding for {synthetic_dir}")
    source = np.concatenate(depth_values)
    target = np.concatenate(grayscale_values)
    keep = np.ones(len(source), dtype=bool)
    coefficients = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(5):
        design = np.column_stack([source[keep], np.ones(int(keep.sum()))])
        coefficients = np.linalg.lstsq(design, target[keep], rcond=None)[0]
        residual = np.abs(coefficients[0] * source + coefficients[1] - target)
        threshold = max(2.0, float(np.percentile(residual[keep], 80)))
        keep = residual <= threshold

    residual = np.abs(coefficients[0] * source + coefficients[1] - target)
    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "sample_count": int(len(source)),
        "inlier_count": int(keep.sum()),
        "median_abs_residual": float(np.median(residual[keep])),
        "p95_abs_residual": float(np.percentile(residual[keep], 95)),
    }


def encode_depth(depth: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    output = np.full(depth.shape, 255, dtype=np.uint8)
    encoded = np.rint(slope * depth[valid] + intercept)
    output[valid] = np.clip(encoded, 0, 255).astype(np.uint8)
    return output


def replace_sequence(
    project_root: Path,
    backup_root: Path,
    config: SequenceConfig,
    dry_run: bool,
) -> dict:
    archived_blender_data = project_root / "backup" / "blender"
    real_dir = archived_blender_data / "real" / config.pair_dir_name
    synthetic_dir = archived_blender_data / "synthetic" / config.pair_dir_name
    dataset_dir = (
        project_root / "dataset" / "BlenderPyMouse" / config.dataset_name
    )
    backup_dir = backup_root / config.pair_dir_name

    synthetic_rgb = frame_numbers(synthetic_dir, "rgb", ".jpg")
    synthetic_depth = frame_numbers(synthetic_dir, "depth", ".png")
    real_rgb = frame_numbers(real_dir, "rgb", ".jpg")
    real_depth = frame_numbers(real_dir, "depth", ".png")
    dataset_frames = {
        int(path.name.removeprefix("frame_"))
        for path in dataset_dir.glob("frame_*")
        if path.is_dir()
    }
    paired_frames = sorted(
        synthetic_rgb
        & synthetic_depth
        & real_rgb
        & real_depth
        & dataset_frames
    )
    if not paired_frames:
        raise RuntimeError(f"No paired replacement frames for {config.label}")

    encoding = fit_depth_encoding(
        synthetic_dir,
        dataset_dir,
        config.view_id,
        config.resize_mode,
        config.crop_x,
        config.crop_width,
        paired_frames,
    )
    rows = []
    if not dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for frame in paired_frames:
        rgb_target = synthetic_dir / f"rgb_{frame:012d}.jpg"
        depth_target = synthetic_dir / f"depth_{frame:012d}.png"
        rgb_source = (
            dataset_dir
            / f"frame_{frame:06d}"
            / "blended_images"
            / f"{config.view_id:08d}.jpg"
        )
        depth_source = (
            dataset_dir
            / f"frame_{frame:06d}"
            / "rendered_depth_maps"
            / f"{config.view_id:08d}.pfm"
        )
        if not rgb_source.is_file() or not depth_source.is_file():
            raise FileNotFoundError(f"Missing dataset source for frame {frame}")

        if not dry_run:
            rgb_backup = backup_dir / rgb_target.name
            depth_backup = backup_dir / depth_target.name
            if not rgb_backup.exists():
                shutil.copy2(rgb_target, rgb_backup)
            if not depth_backup.exists():
                shutil.copy2(depth_target, depth_backup)

            rgb = cv2.imread(str(rgb_source), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Could not read {rgb_source}")
            rgb = prepare_image(
                rgb,
                config.resize_mode,
                cv2.INTER_AREA,
                config.crop_x,
                config.crop_width,
            )
            if not cv2.imwrite(
                str(rgb_target), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Could not write {rgb_target}")

            depth = prepare_image(
                read_pfm(depth_source),
                config.resize_mode,
                cv2.INTER_NEAREST,
                config.crop_x,
                config.crop_width,
            )
            encoded = encode_depth(
                depth, float(encoding["slope"]), float(encoding["intercept"])
            )
            if not cv2.imwrite(str(depth_target), encoded):
                raise RuntimeError(f"Could not write {depth_target}")

        rows.append(
            {
                "frame": frame,
                "rgb_target": str(rgb_target),
                "depth_target": str(depth_target),
                "rgb_source": str(rgb_source),
                "depth_source": str(depth_source),
            }
        )

    return {
        "label": config.label,
        "camera": config.camera,
        "view_id": config.view_id,
        "resize_mode": config.resize_mode,
        "crop_x": config.crop_x,
        "crop_width": config.crop_width,
        "dataset": config.dataset_name,
        "replacement_count": len(rows),
        "first_frame": paired_frames[0],
        "last_frame": paired_frames[-1],
        "depth_encoding": encoding,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    backup_root = args.backup_dir.resolve()
    report_path = args.report.resolve()
    results = [
        replace_sequence(project_root, backup_root, config, args.dry_run)
        for config in SEQUENCES
    ]
    report = {
        "project_root": str(project_root),
        "backup_root": str(backup_root),
        "dry_run": bool(args.dry_run),
        "replacement_policy": "real_and_synthetic_and_dataset_intersection",
        "rgb_transform": "per-sequence resize_mode, then 512x512 INTER_AREA",
        "depth_transform": "per-sequence resize_mode, then 512x512 INTER_NEAREST",
        "invalid_depth_value": 255,
        "sequences": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "dry_run": args.dry_run,
        "sequences": [
            {
                "label": result["label"],
                "replacement_count": result["replacement_count"],
                "depth_encoding": result["depth_encoding"],
            }
            for result in results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

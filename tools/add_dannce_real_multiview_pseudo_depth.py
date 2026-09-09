#!/usr/bin/env python3
"""Add SAM2 masks and calibrated visual-hull pseudo-depth to DANNCE scenes.

The DANNCE RealMultiview datasets contain sparse, widely separated labeled
times.  A single temporal 4D-GS model is therefore a poor fit for the full
dataset.  This tool keeps the successful foreground-V3 ingredients (SAM2
foregrounds, calibrated multi-view agreement, and a 2 mm visual hull), but
computes an independent hull for every synchronized scene.  It writes the
native BlenderPyMouse/BlendedMVS layout: float32 millimetre PFM depth and PNG
mouse masks at exactly the RGB resolution.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


METHOD = "sam2.1_hiera_small_calibrated_visual_hull_2mm_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("third_party/checkpoints/sam2.1_hiera_small.pt"),
    )
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--bounds-padding-mm", type=float, default=18.0)
    parser.add_argument("--minimum-views", type=int, default=4)
    parser.add_argument("--box-padding-px", type=float, default=28.0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--overwrite-depth", action="store_true")
    parser.add_argument("--review-scenes", type=int, default=3)
    return parser.parse_args()


def write_pfm(path: Path, depth_mm: np.ndarray) -> None:
    if depth_mm.ndim != 2:
        raise ValueError(f"Expected grayscale depth, got {depth_mm.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.flipud(depth_mm).astype("<f4", copy=False)
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        handle.write(b"-1.0\n")
        image.tofile(handle)


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = (int(value) for value in handle.readline().split())
        scale = float(handle.readline())
        endian = "<" if scale < 0 else ">"
        image = np.fromfile(handle, dtype=endian + "f4", count=width * height)
    return np.flipud(image.reshape(height, width))


def farthest_points(points: np.ndarray, count: int = 10) -> np.ndarray:
    center = np.median(points, axis=0, keepdims=True)
    candidates = np.concatenate((points, center), axis=0)
    chosen = [len(candidates) - 1]
    while len(chosen) < min(count, len(candidates)):
        distance = np.min(
            np.linalg.norm(candidates[:, None] - candidates[chosen][None], axis=-1),
            axis=1,
        )
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return candidates[chosen].astype(np.float32)


def prompts_for_view(
    uv: np.ndarray,
    in_frame: np.ndarray,
    width: int,
    height: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(uv[in_frame], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 4:
        raise ValueError("Fewer than four in-frame keypoints cannot prompt SAM2")
    low = np.maximum(points.min(axis=0) - padding, [0.0, 0.0])
    high = np.minimum(points.max(axis=0) + padding, [width - 1.0, height - 1.0])
    box = np.asarray([low[0], low[1], high[0], high[1]], dtype=np.float32)
    return box, farthest_points(points)


def clean_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    binary = np.uint8(mask) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    prompt_components = set()
    for x, y in np.rint(points).astype(np.int32):
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            component = int(labels[y, x])
            if component:
                prompt_components.add(component)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest, cv2.CC_STAT_AREA])
    keep = np.zeros_like(binary)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if component == largest or component in prompt_components or area >= max(24, largest_area * 0.002):
            keep[labels == component] = 255
    return keep


def segment_scene(predictor, images: list[np.ndarray], boxes, points) -> list[np.ndarray]:
    rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image_batch(rgb_images)
        masks, scores, _ = predictor.predict_batch(
            point_coords_batch=points,
            point_labels_batch=[np.ones(len(item), dtype=np.int32) for item in points],
            box_batch=boxes,
            multimask_output=True,
        )
    result = []
    for candidates, candidate_scores, positive_points in zip(masks, scores, points, strict=True):
        order = np.argsort(candidate_scores)[::-1]
        selected = candidates[int(order[0])] > 0
        result.append(clean_mask(selected, positive_points))
    return result


def project(points_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray):
    camera = points_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    homogeneous = camera @ intrinsic.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv, camera[:, 2]


def visual_hull(
    keypoints_world: np.ndarray,
    masks: list[np.ndarray],
    intrinsics: list[np.ndarray],
    extrinsics: list[np.ndarray],
    voxel_mm: float,
    padding_mm: float,
    minimum_views: int,
) -> tuple[np.ndarray, dict]:
    low = keypoints_world.min(axis=0) - padding_mm
    high = keypoints_world.max(axis=0) + padding_mm
    axes = [np.arange(low[i], high[i] + voxel_mm * 0.5, voxel_mm) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)
    votes = np.zeros(len(candidates), dtype=np.uint8)
    for mask, intrinsic, extrinsic in zip(masks, intrinsics, extrinsics, strict=True):
        uv, z = project(candidates, intrinsic, extrinsic)
        x = np.rint(uv[:, 0]).astype(np.int32)
        y = np.rint(uv[:, 1]).astype(np.int32)
        inside = (
            (z > 0)
            & (x >= 0)
            & (x < mask.shape[1])
            & (y >= 0)
            & (y < mask.shape[0])
        )
        hit = np.zeros(len(candidates), dtype=bool)
        hit[inside] = mask[y[inside], x[inside]] > 127
        votes += hit
    occupied = candidates[votes >= minimum_views]
    if len(occupied) < 100:
        raise RuntimeError(f"Visual hull has only {len(occupied)} voxels")
    return occupied, {
        "candidate_voxels": int(len(candidates)),
        "occupied_voxels": int(len(occupied)),
        "bounds_mm": [low.tolist(), high.tolist()],
    }


def rasterize_depth(
    points_world: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    voxel_mm: float,
) -> np.ndarray:
    uv, z = project(points_world, intrinsic, extrinsic)
    x = np.rint(uv[:, 0]).astype(np.int32)
    y = np.rint(uv[:, 1]).astype(np.int32)
    inside = (
        (z > voxel_mm)
        & (x >= 0)
        & (x < mask.shape[1])
        & (y >= 0)
        & (y < mask.shape[0])
    )
    x, y, z = x[inside], y[inside], z[inside]
    sparse = np.full(mask.shape, np.inf, dtype=np.float32)
    np.minimum.at(sparse.ravel(), y * mask.shape[1] + x, z.astype(np.float32) - voxel_mm * 0.5)
    median_z = max(float(np.median(z)), voxel_mm)
    radius = int(np.ceil(max(intrinsic[0, 0], intrinsic[1, 1]) * voxel_mm * 0.9 / median_z))
    radius = int(np.clip(radius, 1, 16))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dense = cv2.erode(sparse, kernel)
    # OpenCV may turn +inf into float32 max during erosion; explicitly reject
    # values outside any physically plausible calibrated DANNCE depth.
    valid = np.isfinite(dense) & (dense < 10_000.0) & (mask > 127)
    depth = np.zeros(mask.shape, dtype=np.float32)
    depth[valid] = dense[valid]
    # Suppress visible 2 mm voxel banding without leaking depth through the
    # silhouette.  The normalized filter only mixes already valid foreground.
    weight = valid.astype(np.float32)
    numerator = cv2.GaussianBlur(depth, (0, 0), 1.2)
    denominator = cv2.GaussianBlur(weight, (0, 0), 1.2)
    depth[valid] = numerator[valid] / np.maximum(denominator[valid], 1e-6)
    return depth


def depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.count_nonzero(valid) < 16:
        return result
    near, far = np.percentile(depth[valid], [2, 98])
    normalized = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    result = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def update_json(path: Path, updates: dict) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_scene_meta(path: Path, view_count: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "depth_available": True,
            "depth_units": "millimeter",
            "depth_is_pseudo_label": True,
            "depth_method": METHOD,
            "mask_available": True,
            "mask_is_pseudo_label": True,
        }
    )
    for view in payload.get("views", []):
        view_id = int(view["view_id"])
        if not 0 <= view_id < view_count:
            continue
        view.update(
            {
                "depth_available": True,
                "depth": f"rendered_depth_maps/{view_id:08d}.pfm",
                "depth_units": "millimeter",
                "depth_is_pseudo_label": True,
                "depth_method": METHOD,
                "mask_available": True,
                "mask": f"masks/{view_id:08d}.png",
                "mask_is_pseudo_label": True,
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def review_sheet(scene: Path, view_count: int) -> np.ndarray:
    rows = []
    for view_id in range(view_count):
        stem = f"{view_id:08d}"
        image = cv2.imread(str(scene / "blended_images" / f"{stem}.jpg"))
        mask = cv2.imread(str(scene / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        depth = read_pfm(scene / "rendered_depth_maps" / f"{stem}.pfm")
        overlay = image.copy()
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        selector = mask > 127
        overlay[selector] = cv2.addWeighted(overlay, 0.45, green, 0.55, 0)[selector]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        panels = [image, overlay, depth_preview(depth)]
        labels = [f"view {view_id} RGB", "SAM2 mask", "visual-hull depth (mm)"]
        tiles = []
        for panel, text in zip(panels, labels, strict=True):
            tile = cv2.resize(panel, (360, 320), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (300, 30), (15, 15, 15), -1)
            cv2.putText(tile, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def process_dataset(args: argparse.Namespace, predictor, dataset: Path) -> dict:
    dataset = dataset.resolve()
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scenes = [dataset / name for name in metadata["scene_folders"]]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    view_count = int(metadata["views_per_scene"])
    review_indices = set(
        np.linspace(0, max(len(scenes) - 1, 0), min(args.review_scenes, len(scenes)), dtype=int).tolist()
    )
    review_root = dataset / "pseudo_depth_review"
    review_root.mkdir(exist_ok=True)
    scene_records = []

    for scene_index, scene in enumerate(scenes):
        keypoint_path = scene / "keypoints.npz"
        with np.load(keypoint_path, allow_pickle=False) as arrays:
            uv = arrays["uv"][:, 0]
            in_frame = arrays["in_frame"][:, 0]
            source_world = arrays["source_joint_world_xyz"].astype(np.float32)
            source_valid = arrays["source_world_valid"].astype(bool)
        keypoints_world = source_world[source_valid]
        annotations = [
            json.loads((scene / "annotations" / f"{view_id:08d}.json").read_text(encoding="utf-8"))
            for view_id in range(view_count)
        ]
        images = [
            cv2.imread(str(scene / "blended_images" / f"{view_id:08d}.jpg"))
            for view_id in range(view_count)
        ]
        if any(image is None for image in images):
            raise FileNotFoundError(f"Missing RGB image under {scene}")
        shapes = {image.shape[:2] for image in images}
        if len(shapes) != 1:
            raise ValueError(f"Views have inconsistent sizes in {scene}: {shapes}")
        height, width = images[0].shape[:2]
        boxes, point_prompts = zip(
            *[
                prompts_for_view(uv[view_id], in_frame[view_id], width, height, args.box_padding_px)
                for view_id in range(view_count)
            ],
            strict=True,
        )
        mask_paths = [scene / "masks" / f"{view_id:08d}.png" for view_id in range(view_count)]
        depth_paths = [scene / "rendered_depth_maps" / f"{view_id:08d}.pfm" for view_id in range(view_count)]
        if args.overwrite or args.overwrite_masks or not all(path.is_file() for path in mask_paths):
            masks = segment_scene(predictor, images, list(boxes), list(point_prompts))
            for path, mask in zip(mask_paths, masks, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), mask):
                    raise RuntimeError(f"Could not write {path}")
        else:
            masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in mask_paths]

        intrinsics = [np.asarray(item["intrinsic"], dtype=np.float64) for item in annotations]
        extrinsics = [np.asarray(item["extrinsic"], dtype=np.float64) for item in annotations]
        hull, hull_stats = visual_hull(
            keypoints_world,
            masks,
            intrinsics,
            extrinsics,
            args.voxel_mm,
            args.bounds_padding_mm,
            args.minimum_views,
        )
        coverage = []
        ranges = []
        for view_id, (mask, intrinsic, extrinsic, depth_path) in enumerate(
            zip(masks, intrinsics, extrinsics, depth_paths, strict=True)
        ):
            if args.overwrite or args.overwrite_depth or not depth_path.is_file():
                depth = rasterize_depth(hull, mask, intrinsic, extrinsic, args.voxel_mm)
                write_pfm(depth_path, depth)
                cv2.imwrite(
                    str(depth_path.with_name(depth_path.stem + "_preview.png")),
                    depth_preview(depth),
                )
            else:
                depth = read_pfm(depth_path)
            valid = depth > 0
            coverage.append(float(np.count_nonzero(valid) / max(np.count_nonzero(mask), 1)))
            ranges.append(
                [float(depth[valid].min()), float(depth[valid].max())]
                if np.any(valid)
                else [0.0, 0.0]
            )
            annotation_path = scene / "annotations" / f"{view_id:08d}.json"
            update_json(
                annotation_path,
                {
                    "depth_available": True,
                    "depth": f"rendered_depth_maps/{view_id:08d}.pfm",
                    "depth_units": "millimeter",
                    "depth_is_pseudo_label": True,
                    "depth_method": METHOD,
                    "mask_available": True,
                    "mask": f"masks/{view_id:08d}.png",
                    "mask_is_pseudo_label": True,
                },
            )
        update_scene_meta(scene / "scene_meta.json", view_count)
        if scene_index in review_indices:
            cv2.imwrite(str(review_root / f"{scene.name}_sixview.jpg"), review_sheet(scene, view_count))
        record = {
            "scene": scene.name,
            **hull_stats,
            "mask_pixels": [int(np.count_nonzero(mask)) for mask in masks],
            "depth_mask_coverage": coverage,
            "depth_range_mm": ranges,
        }
        scene_records.append(record)
        print(
            f"[{dataset.name}] {scene_index + 1}/{len(scenes)} {scene.name}: "
            f"hull={len(hull)} coverage={np.mean(coverage):.3f}",
            flush=True,
        )

    complete = len(scenes) == int(metadata["scene_count"])
    manifest = {
        "format": "dannce_real_multiview_pseudo_depth_v1",
        "dataset": str(dataset),
        "method": METHOD,
        "scene_count_processed": len(scenes),
        "scene_count_expected": int(metadata["scene_count"]),
        "complete": complete,
        "views_per_scene": view_count,
        "resolution": list(reversed(images[0].shape[:2])),
        "mask_encoding": "uint8 PNG; mouse=255, background=0",
        "depth_encoding": "grayscale float32 PFM, vertically stored per PFM convention",
        "depth_units": "millimeter",
        "depth_semantics": "first occupied visual-hull surface along the calibrated camera ray",
        "pseudo_label": True,
        "voxel_mm": args.voxel_mm,
        "minimum_silhouette_views": args.minimum_views,
        "bounds_padding_mm": args.bounds_padding_mm,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean_depth_mask_coverage": float(
            np.mean([value for row in scene_records for value in row["depth_mask_coverage"]])
        ),
        "scenes": scene_records,
    }
    (dataset / "pseudo_depth_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if complete:
        metadata.update(
            {
                "depth_available": True,
                "depth_format": "PFM float32",
                "depth_units": "millimeter",
                "depth_is_pseudo_label": True,
                "depth_method": METHOD,
                "mask_available": True,
                "mask_format": "PNG uint8 mouse foreground",
                "mask_is_pseudo_label": True,
                "depth_resolution": list(reversed(images[0].shape[:2])),
                "pseudo_depth_manifest": "pseudo_depth_manifest.json",
            }
        )
        layout = dict(metadata.get("layout", {}))
        layout.update(
            {
                "depths": "frame_xxxxxx/rendered_depth_maps/00000000.pfm",
                "depth_previews": "frame_xxxxxx/rendered_depth_maps/00000000_preview.png",
                "masks": "frame_xxxxxx/masks/00000000.png",
            }
        )
        metadata["layout"] = layout
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> None:
    args = parse_args()
    if not 1 <= args.minimum_views <= 6:
        raise ValueError("--minimum-views must be between 1 and 6")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(args.checkpoint.resolve()),
        device="cuda",
    )
    predictor = SAM2ImagePredictor(model)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    summaries = [process_dataset(args, predictor, dataset) for dataset in args.dataset]
    print(json.dumps([{k: v for k, v in row.items() if k != "scenes"} for row in summaries], indent=2))


if __name__ == "__main__":
    main()

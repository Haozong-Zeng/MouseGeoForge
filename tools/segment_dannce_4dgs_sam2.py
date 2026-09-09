#!/usr/bin/env python3
"""Track the DANNCE mouse foreground in the prepared six-view 4D-GS clip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/DANNCE-4DGS-Pilot-mm1-f0000-0079"),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("dataset/DANNCE/raw/markerless_mouse_1/label3d_dannce.mat"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("third_party/checkpoints/sam2.1_hiera_small.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/DANNCE-4DGS-Pilot-mm1-f0000-0079/sam2_masks_v1"),
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--box-padding", type=float, default=18.0)
    return parser.parse_args()


def project_points(points_world: np.ndarray, manifest: dict, camera_index: int) -> np.ndarray:
    camera = manifest["cameras"][camera_index]
    center = np.asarray(manifest["world_center_mm"], dtype=np.float64)
    scale = float(manifest["world_scale_mm"])
    xyz = (points_world - center) / scale
    rotation = np.asarray(camera["rotation_world_to_camera"], dtype=np.float64)
    translation = np.asarray(
        camera["translation_world_to_camera_normalized"], dtype=np.float64
    )
    camera_xyz = xyz @ rotation.T + translation
    focal = float(manifest["virtual_focal_px"])
    width, height = manifest["resolution"]
    uv = np.stack(
        (
            focal * camera_xyz[:, 0] / camera_xyz[:, 2] + width * 0.5,
            focal * camera_xyz[:, 1] / camera_xyz[:, 2] + height * 0.5,
        ),
        axis=-1,
    )
    return uv


def prompt_from_keypoints(
    points_world: np.ndarray,
    manifest: dict,
    camera_index: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points_world).all(axis=1) & (np.linalg.norm(points_world, axis=1) > 1e-5)
    uv = project_points(points_world[valid], manifest, camera_index)
    width, height = manifest["resolution"]
    in_frame = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    uv = uv[in_frame]
    if len(uv) < 4:
        raise RuntimeError(f"Too few projected keypoints for camera {camera_index + 1}")
    low = np.maximum(uv.min(axis=0) - padding, [0.0, 0.0])
    high = np.minimum(uv.max(axis=0) + padding, [width - 1.0, height - 1.0])
    box = np.asarray([low[0], low[1], high[0], high[1]], dtype=np.float32)
    # Dense positive points can make SAM overfit individual limbs. Keep spatially
    # separated points plus the center to cover the trunk, head, paws, and tail.
    center = np.median(uv, axis=0, keepdims=True)
    candidates = np.concatenate((uv, center), axis=0)
    chosen = [len(candidates) - 1]
    while len(chosen) < min(10, len(candidates)):
        distance = np.min(
            np.linalg.norm(candidates[:, None] - candidates[chosen][None], axis=-1),
            axis=1,
        )
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return box, candidates[chosen].astype(np.float32)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.uint8(mask) * 255
    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.max(areas))
    keep = np.zeros_like(mask_u8)
    for component, area in enumerate(areas, start=1):
        if area >= max(12, largest * 0.002):
            keep[labels == component] = 255
    return keep


def overlay(image: np.ndarray, mask: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
    result = image.copy()
    green = np.zeros_like(result)
    green[..., 1] = 255
    selector = mask > 0
    result[selector] = cv2.addWeighted(result, 0.45, green, 0.55, 0)[selector]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    if box is not None:
        x0, y0, x1, y1 = np.rint(box).astype(int)
        cv2.rectangle(result, (x0, y0), (x1, y1), (255, 128, 0), 1)
    return result


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    mask_root = output / "masks"
    numeric_root = output / "sam2_numeric_frames"
    overlay_root = output / "overlays"
    for path in (mask_root, numeric_root, overlay_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dataset / "pilot_manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest["frame_count"])
    mat = loadmat(args.annotation, squeeze_me=True, struct_as_record=False)
    label_entry = list(np.asarray(mat["labelData"], dtype=object).reshape(-1))[0]
    label_frames = np.asarray(label_entry.data_frame, dtype=np.int64)
    label_xyz = np.asarray(label_entry.data_3d, dtype=np.float64).reshape(-1, 22, 3)
    prompt_frames = [
        int(frame) for frame in label_frames if 0 <= int(frame) < frame_count
    ]
    if not prompt_frames:
        raise RuntimeError("No labeled prompt frames overlap the pilot clip")
    labels_by_frame = {
        int(frame): label_xyz[index]
        for index, frame in enumerate(label_frames)
        if 0 <= int(frame) < frame_count
    }

    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(args.checkpoint.resolve()),
        device="cuda",
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    camera_records = []
    preview_frames = [0, *prompt_frames, 40, frame_count - 1]
    preview_frames = sorted(set(frame for frame in preview_frames if 0 <= frame < frame_count))
    for camera_number in args.cameras:
        camera_index = camera_number - 1
        source = dataset / f"cam{camera_number:02d}"
        numeric = numeric_root / f"cam{camera_number:02d}"
        numeric.mkdir(parents=True, exist_ok=True)
        for frame in range(frame_count):
            link = numeric / f"{frame:05d}.jpg"
            target = source / f"frame_{frame + 1:05d}.jpg"
            if not link.exists():
                link.symlink_to(target)

        state = predictor.init_state(
            video_path=str(numeric),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        prompts = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame in prompt_frames:
                box, positive_points = prompt_from_keypoints(
                    labels_by_frame[frame], manifest, camera_index, args.box_padding
                )
                prompts[frame] = {"box": box, "points": positive_points}
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame,
                    obj_id=1,
                    points=positive_points,
                    labels=np.ones(len(positive_points), dtype=np.int32),
                    box=box,
                )

            masks: dict[int, np.ndarray] = {}
            start = min(prompt_frames)
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start
            ):
                object_index = list(object_ids).index(1)
                masks[int(frame_index)] = clean_mask(
                    (logits[object_index, 0] > 0).detach().cpu().numpy()
                )
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, reverse=True
            ):
                object_index = list(object_ids).index(1)
                masks[int(frame_index)] = clean_mask(
                    (logits[object_index, 0] > 0).detach().cpu().numpy()
                )

        if len(masks) != frame_count:
            missing = sorted(set(range(frame_count)) - set(masks))
            raise RuntimeError(f"Camera {camera_number} missing masks: {missing}")
        camera_mask_root = mask_root / f"cam{camera_number:02d}"
        camera_overlay_root = overlay_root / f"cam{camera_number:02d}"
        camera_mask_root.mkdir(parents=True, exist_ok=True)
        camera_overlay_root.mkdir(parents=True, exist_ok=True)
        areas = []
        for frame in range(frame_count):
            mask = masks[frame]
            areas.append(int(np.count_nonzero(mask)))
            cv2.imwrite(str(camera_mask_root / f"frame_{frame + 1:05d}.png"), mask)
            if frame in preview_frames:
                image = cv2.imread(str(source / f"frame_{frame + 1:05d}.jpg"))
                prompt_box = prompts.get(frame, {}).get("box")
                cv2.imwrite(
                    str(camera_overlay_root / f"frame_{frame:04d}.jpg"),
                    overlay(image, mask, prompt_box),
                    [cv2.IMWRITE_JPEG_QUALITY, 94],
                )
        camera_records.append(
            {
                "camera": camera_number,
                "area_min": min(areas),
                "area_median": float(np.median(areas)),
                "area_max": max(areas),
                "area_fraction_median": float(np.median(areas) / np.prod(manifest["resolution"])),
                "prompts": {
                    str(frame): {
                        "box": prompts[frame]["box"].tolist(),
                        "positive_points": prompts[frame]["points"].tolist(),
                    }
                    for frame in prompt_frames
                },
            }
        )
        predictor.reset_state(state)
        print(json.dumps(camera_records[-1], indent=2))

    record = {
        "format": "dannce_sam2_mouse_masks_v1",
        "source_dataset": str(dataset),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_frames_zero_based": prompt_frames,
        "mask_semantics": "mouse foreground=255, background=0",
        "cameras": camera_records,
    }
    (output / "mask_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    preview_root = output / "sixview_previews"
    preview_root.mkdir(exist_ok=True)
    if set(args.cameras) == set(range(1, int(manifest["camera_count"]) + 1)):
        for frame in preview_frames:
            tiles = []
            for camera_number in args.cameras:
                tile = cv2.imread(
                    str(overlay_root / f"cam{camera_number:02d}" / f"frame_{frame:04d}.jpg")
                )
                tile = cv2.resize(tile, (256, 256), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tile, (0, 0), (150, 27), (15, 15, 15), -1)
                cv2.putText(
                    tile,
                    f"cam{camera_number:02d} frame{frame:04d}",
                    (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            montage = np.concatenate(
                (np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)),
                axis=0,
            )
            cv2.imwrite(
                str(preview_root / f"frame_{frame:04d}_sixview.jpg"),
                montage,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )


if __name__ == "__main__":
    main()

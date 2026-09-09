#!/usr/bin/env python3
"""Evaluate a trained VGGT-Omega keypoint head by semantic keypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.finetune_blendedmvs import (
    load_checkpoint_chain,
    make_loader,
    make_real_geometry_loader,
    remap_real_keypoint_targets,
    to_device,
)
from training.keypoint_head_losses import (
    heatmap_soft_argmax_with_covariance,
    triangulate_gaussian_rays,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-batches", type=int, default=32)
    parser.add_argument("--real-batches", type=int, default=64)
    parser.add_argument("--overlay-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_training_args(checkpoint: Path) -> tuple[SimpleNamespace, list[str]]:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "args" not in payload:
        raise ValueError(f"Checkpoint does not contain saved training args: {checkpoint}")
    values = dict(payload["args"])
    names = [str(name) for name in values.get("resolved_keypoint_names", [])]
    if not names:
        raise ValueError(f"Checkpoint does not record resolved_keypoint_names: {checkpoint}")
    values["num_workers"] = 0
    return SimpleNamespace(**values), names


def semantic_group(name: str) -> str:
    if "Finger" in name and not name.startswith("Fingers1_"):
        return "finger"
    if name.startswith("Fingers1_"):
        return "palm"
    if "Wrist" in name:
        return "wrist"
    if any(token in name for token in ("Elbow", "Scapula", "Shoulder")):
        return "arm"
    if any(token in name for token in ("Head", "Snout", "Nose")):
        return "head"
    return "other"


def predict_geometry(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    target_xy: torch.Tensor,
    point_valid: torch.Tensor,
    view_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    image_size = batch["images"].shape[-2:]
    xy, peak, heatmap_covariance = heatmap_soft_argmax_with_covariance(
        predictions["keypoint_heatmap_logits"],
        image_size,
    )
    visibility = predictions["keypoint_visibility_logits"]
    sigma = predictions["keypoint_pixel_sigma"]
    camera_available = torch.ones(
        batch["images"].shape[:2],
        dtype=torch.bool,
        device=batch["images"].device,
    )
    fusion_valid = view_valid & point_valid[:, None, :] & camera_available[:, :, None]
    fixed_center, _, fixed_count = triangulate_gaussian_rays(
        xy,
        visibility,
        sigma,
        batch["extrinsics"],
        batch["intrinsics"],
        fusion_valid,
        pixel_covariance=heatmap_covariance,
    )
    predicted_extrinsics, predicted_intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        image_size,
        build_intrinsics=True,
    )
    predicted_center, _, predicted_count = triangulate_gaussian_rays(
        xy,
        visibility,
        sigma,
        predicted_extrinsics,
        predicted_intrinsics,
        fusion_valid,
        pixel_covariance=heatmap_covariance,
    )
    return {
        "xy": xy,
        "peak": peak,
        "fixed_center": fixed_center,
        "predicted_center": predicted_center,
        "center_valid": point_valid & (fixed_count >= 2) & (predicted_count >= 2),
        "pixel_valid": view_valid & point_valid[:, None, :],
        "target_xy": target_xy,
    }


def append_metrics(
    rows: dict[str, dict[str, list[float]]],
    names: list[str],
    geometry: dict[str, torch.Tensor],
    target_world: torch.Tensor,
    scale: torch.Tensor | None,
) -> None:
    pixel_error = torch.linalg.vector_norm(
        geometry["xy"] - geometry["target_xy"],
        dim=-1,
    )
    fixed_error = torch.linalg.vector_norm(
        geometry["fixed_center"] - target_world,
        dim=-1,
    )
    predicted_error = torch.linalg.vector_norm(
        geometry["predicted_center"] - target_world,
        dim=-1,
    )
    for index, name in enumerate(names):
        pixel_values = pixel_error[..., index][geometry["pixel_valid"][..., index]]
        valid_3d = geometry["center_valid"][:, index]
        fixed_values = fixed_error[:, index][valid_3d]
        predicted_values = predicted_error[:, index][valid_3d]
        if scale is not None and valid_3d.any():
            point_scale = scale[valid_3d].to(dtype=fixed_values.dtype)
            fixed_values = fixed_values * point_scale
            predicted_values = predicted_values * point_scale
        rows[name]["pixel"].extend(pixel_values.detach().cpu().tolist())
        rows[name]["fixed3d"].extend(fixed_values.detach().cpu().tolist())
        rows[name]["predicted3d"].extend(predicted_values.detach().cpu().tolist())


def draw_overlay(
    output: Path,
    batch: dict[str, Any],
    geometry: dict[str, torch.Tensor],
    names: list[str],
    prefix: str,
    batch_index: int,
) -> None:
    colors = {
        "finger": (255, 80, 75),
        "palm": (40, 220, 110),
        "wrist": (255, 210, 45),
        "arm": (90, 180, 255),
        "head": (220, 80, 255),
        "other": (255, 255, 255),
    }
    images = batch["images"][0].detach().cpu().numpy()
    predicted = geometry["xy"][0].detach().cpu().numpy()
    target = geometry["target_xy"][0].detach().cpu().numpy()
    valid = geometry["pixel_valid"][0].detach().cpu().numpy()
    for view in range(images.shape[0]):
        image = np.clip(images[view].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for index, name in enumerate(names):
            if not valid[view, index]:
                continue
            tx, ty = target[view, index]
            px, py = predicted[view, index]
            color = colors[semantic_group(name)]
            draw.line((tx, ty, px, py), fill=(255, 255, 255), width=1)
            draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), outline=(40, 255, 80), width=2)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=color, width=2)
        canvas.save(output / f"{prefix}_batch{batch_index:03d}_view{view}.png")


def summarize_rows(
    rows: dict[str, dict[str, list[float]]],
    names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_keypoint = []
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for name in names:
        group = semantic_group(name)
        values = rows[name]
        row = {"name": name, "group": group}
        for metric in ("pixel", "fixed3d", "predicted3d"):
            metric_values = np.asarray(values[metric], dtype=np.float64)
            row[f"{metric}_count"] = int(metric_values.size)
            row[f"{metric}_mean"] = (
                float(metric_values.mean()) if metric_values.size else None
            )
            row[f"{metric}_median"] = (
                float(np.median(metric_values)) if metric_values.size else None
            )
            groups[group][metric].extend(metric_values.tolist())
        per_keypoint.append(row)

    summary: dict[str, Any] = {}
    for group, metrics in groups.items():
        summary[group] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            summary[group][metric] = {
                "count": int(array.size),
                "mean": float(array.mean()) if array.size else None,
                "median": float(np.median(array)) if array.size else None,
            }
    return per_keypoint, summary


@torch.no_grad()
def evaluate_loader(
    model: VGGTOmega,
    loader: torch.utils.data.DataLoader,
    names: list[str],
    device: torch.device,
    output: Path,
    prefix: str,
    max_batches: int,
    overlay_batches: int,
    real_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = to_device(batch, device)
        predictions = model(batch["images"])
        if real_names is None:
            target_xy = batch["keypoint_uv"]
            target_world = batch["keypoint_joint_world_xyz"]
            point_valid = batch["keypoint_valid"]
            view_valid = batch["keypoint_in_frame"]
            scale = None
        else:
            remapped = remap_real_keypoint_targets(
                batch,
                real_names,
                names,
            )
            target_xy = remapped["target_xy"]
            target_world = remapped["target_world"]
            point_valid = remapped["point_valid"]
            view_valid = remapped["view_valid"]
            scale = batch["scale"]
        geometry = predict_geometry(
            predictions,
            batch,
            target_xy,
            point_valid,
            view_valid,
        )
        append_metrics(
            rows,
            names,
            geometry,
            target_world,
            scale,
        )
        if batch_index < overlay_batches:
            draw_overlay(
                output,
                batch,
                geometry,
                names,
                prefix,
                batch_index,
            )
    return summarize_rows(rows, names)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    training_args, names = load_training_args(cli.checkpoint)
    synthetic_loader = make_loader(training_args, "val")
    real_loader = make_real_geometry_loader(training_args, "val")
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    model = VGGTOmega(
        enable_depth=False,
        enable_keypoints=True,
        num_keypoints=len(names),
    ).to(device)
    load_checkpoint_chain(model, cli.checkpoint)
    model.eval()

    synthetic_rows, synthetic_summary = evaluate_loader(
        model,
        synthetic_loader,
        names,
        device,
        cli.output,
        "synthetic",
        cli.synthetic_batches,
        cli.overlay_batches,
    )
    write_csv(cli.output / "synthetic_per_keypoint.csv", synthetic_rows)

    real_rows: list[dict[str, Any]] = []
    real_summary: dict[str, Any] = {}
    if real_loader is not None:
        real_names = [str(name) for name in real_loader.dataset.keypoints]
        real_rows, real_summary = evaluate_loader(
            model,
            real_loader,
            names,
            device,
            cli.output,
            "real",
            cli.real_batches,
            cli.overlay_batches,
            real_names=real_names,
        )
        write_csv(cli.output / "real_per_keypoint.csv", real_rows)

    payload = {
        "checkpoint": str(cli.checkpoint.resolve()),
        "keypoint_names": names,
        "synthetic": synthetic_summary,
        "real": real_summary,
        "units": {
            "pixel": "input pixels",
            "synthetic_3d": "normalized scene units",
            "real_3d": "millimeters",
        },
    }
    (cli.output / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"evaluation_complete={cli.output}")


if __name__ == "__main__":
    main()

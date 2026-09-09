#!/usr/bin/env python3
"""Run camera-specific DLC models and score raw/styled synthetic images against GT."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import deeplabcut
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


def build_keypoint_mapping() -> dict[str, str]:
    # Human-labeled CamB frames show digit 1 at the lateral/pinky side and digit 4
    # at the medial/index side for both paws.
    digits = ("Pinky", "Ring", "Middle", "Index")
    mapping = {}
    for side_short, side_long in (("r", "R"), ("l", "L")):
        for digit_index, digit_name in enumerate(digits, start=1):
            mapping[f"{side_short}{digit_index}_tip_A"] = (
                f"{digit_name}Finger3_{side_long}"
            )
            mapping[f"{side_short}{digit_index}_mid_A"] = (
                f"{digit_name}Finger2_{side_long}"
            )
            mapping[f"{side_short}{digit_index}_joint_A"] = (
                f"{digit_name}Finger1_{side_long}"
            )
        mapping[f"{side_short}_mid_A"] = f"Fingers1_{side_long}"
    mapping["snout_A"] = "HeadEnd_M"
    return mapping


DLC_TO_BLENDER = build_keypoint_mapping()
COLORS = {
    "gt": (40, 230, 90),
    "prediction": (255, 70, 55),
    "line": (255, 215, 50),
    "text": (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="VIEW=PATH",
        help="Repeat for each available camera-specific local config, e.g. A=/.../config.yaml.",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        metavar="VIEW=INDEX",
        help="Optional per-view snapshot index. Defaults to -1 (latest).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pcutoff", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_configs(items: list[str]) -> dict[str, Path]:
    configs = {}
    for item in items:
        view, separator, path_text = item.partition("=")
        view = view.upper().removeprefix("CAMERA").removeprefix("CAM")
        if not separator or view not in {"A", "B", "C"}:
            raise ValueError(f"Invalid --config value: {item!r}")
        path = Path(path_text).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        configs[view] = path
    return configs


def parse_snapshots(items: list[str]) -> dict[str, int]:
    snapshots = {}
    for item in items:
        view, separator, index_text = item.partition("=")
        view = view.upper().removeprefix("CAMERA").removeprefix("CAM")
        if not separator or view not in {"A", "B", "C"}:
            raise ValueError(f"Invalid --snapshot value: {item!r}")
        snapshots[view] = int(index_text)
    return snapshots


def reset_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite")
        resolved = path.resolve()
        if len(resolved.parts) < 5 or resolved == resolved.parent:
            raise ValueError(f"Refuse to overwrite suspicious path: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True)


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    for row in rows:
        row["frame"] = int(row["frame"])
        row["view_id"] = int(row["view_id"])
        row["width"] = int(row["width"])
        row["height"] = int(row["height"])
        row["camera_suffix"] = row["camera"].removeprefix("Camera").upper()
    return rows


def dataframe_predictions(dataframe: pd.DataFrame) -> dict[str, dict[str, dict]]:
    predictions = {}
    for image_path, row in dataframe.iterrows():
        bodyparts = defaultdict(dict)
        for column, value in row.items():
            bodypart = column[-2]
            coordinate = column[-1]
            bodyparts[bodypart][coordinate] = float(value)
        predictions[str(Path(image_path).resolve())] = dict(bodyparts)
    return predictions


def run_condition(
    config: Path,
    images: list[Path],
    output: Path,
    device: str,
    pcutoff: float,
    seed: int,
    snapshot_index: int,
) -> dict[str, dict[str, dict]]:
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deeplabcut.analyze_images(
        config=str(config),
        images=[str(path) for path in images],
        destfolder=str(output),
        shuffle=1,
        snapshot_index=snapshot_index,
        device=device,
        save_as_csv=True,
        plotting=True,
        pcutoff=pcutoff,
        plot_skeleton=False,
    )
    h5_files = list(output.glob("image_predictions_*.h5"))
    if len(h5_files) != 1:
        raise RuntimeError(f"Expected one prediction H5 under {output}, found {h5_files}")
    return dataframe_predictions(pd.read_hdf(h5_files[0]))


def load_gt(annotation_path: Path) -> dict[str, dict]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in annotation.get("keypoints", [])}


def finite_prediction(prediction: dict | None) -> bool:
    return (
        prediction is not None
        and math.isfinite(prediction.get("x", math.nan))
        and math.isfinite(prediction.get("y", math.nan))
    )


def metric_rows(
    manifest_rows: list[dict],
    all_predictions: dict[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for sample in manifest_rows:
        camera = sample["camera_suffix"]
        gt_by_name = load_gt(Path(sample["annotation"]))
        for condition in ("raw", "styled"):
            image_path = str(Path(sample[f"{condition}_image"]).resolve())
            predictions = all_predictions.get((camera, condition), {})
            image_predictions = predictions.get(image_path, {})
            for dlc_name, blender_name in DLC_TO_BLENDER.items():
                gt = gt_by_name.get(blender_name)
                prediction = image_predictions.get(dlc_name)
                gt_visible = bool(gt and gt.get("visible") and "x" in gt and "y" in gt)
                predicted = finite_prediction(prediction)
                likelihood = (
                    float(prediction.get("likelihood", 0.0)) if prediction else 0.0
                )
                error = math.nan
                if gt_visible and predicted:
                    error = math.hypot(
                        prediction["x"] - float(gt["x"]),
                        prediction["y"] - float(gt["y"]),
                    )
                rows.append(
                    {
                        "frame": sample["frame"],
                        "camera": sample["camera"],
                        "condition": condition,
                        "dlc_keypoint": dlc_name,
                        "blender_keypoint": blender_name,
                        "gt_visible": gt_visible,
                        "gt_x": float(gt["x"]) if gt and "x" in gt else math.nan,
                        "gt_y": float(gt["y"]) if gt and "y" in gt else math.nan,
                        "prediction_available": predicted,
                        "pred_x": (
                            float(prediction["x"]) if predicted else math.nan
                        ),
                        "pred_y": (
                            float(prediction["y"]) if predicted else math.nan
                        ),
                        "likelihood": likelihood,
                        "error_px": error,
                        "error_fraction_diagonal": (
                            error / math.hypot(sample["width"], sample["height"])
                            if math.isfinite(error)
                            else math.nan
                        ),
                    }
                )
    return rows


def summarize(metrics: list[dict], pcutoff: float) -> list[dict]:
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["camera"], row["condition"])].append(row)
    summaries = []
    for (camera, condition), rows in sorted(grouped.items()):
        visible = [row for row in rows if row["gt_visible"]]
        errors = np.asarray(
            [row["error_px"] for row in visible if math.isfinite(row["error_px"])],
            dtype=np.float64,
        )
        confident = [
            row
            for row in visible
            if row["prediction_available"] and row["likelihood"] >= pcutoff
        ]
        confident_errors = np.asarray(
            [row["error_px"] for row in confident if math.isfinite(row["error_px"])],
            dtype=np.float64,
        )
        summaries.append(
            {
                "camera": camera,
                "condition": condition,
                "visible_gt_count": len(visible),
                "prediction_count": len(errors),
                "confident_prediction_count": len(confident),
                "coverage_at_pcutoff": (
                    len(confident) / len(visible) if visible else math.nan
                ),
                "coverage_at_0_5": (
                    sum(
                        row["prediction_available"] and row["likelihood"] >= 0.5
                        for row in visible
                    )
                    / len(visible)
                    if visible
                    else math.nan
                ),
                "coverage_at_0_9": (
                    sum(
                        row["prediction_available"] and row["likelihood"] >= 0.9
                        for row in visible
                    )
                    / len(visible)
                    if visible
                    else math.nan
                ),
                "mean_likelihood_visible_gt": (
                    float(np.mean([row["likelihood"] for row in visible]))
                    if visible
                    else math.nan
                ),
                "mean_error_px": float(np.mean(errors)) if len(errors) else math.nan,
                "median_error_px": (
                    float(np.median(errors)) if len(errors) else math.nan
                ),
                "rmse_px": (
                    float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan
                ),
                "mean_confident_error_px": (
                    float(np.mean(confident_errors))
                    if len(confident_errors)
                    else math.nan
                ),
                "pck_10px": (
                    sum(
                        math.isfinite(row["error_px"]) and row["error_px"] <= 10
                        for row in visible
                    )
                    / len(visible)
                    if visible
                    else math.nan
                ),
                "pck_20px": (
                    sum(
                        math.isfinite(row["error_px"]) and row["error_px"] <= 20
                        for row in visible
                    )
                    / len(visible)
                    if visible
                    else math.nan
                ),
                "pck_40px": (
                    sum(
                        math.isfinite(row["error_px"]) and row["error_px"] <= 40
                        for row in visible
                    )
                    / len(visible)
                    if visible
                    else math.nan
                ),
            }
        )
    return summaries


def camera_projection(annotation: dict) -> np.ndarray:
    intrinsic = np.asarray(annotation["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(annotation["extrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(
            f"Invalid camera matrices: K={intrinsic.shape}, E={extrinsic.shape}"
        )
    return intrinsic @ extrinsic[:3, :]


def project_point(projection: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.append(point_xyz, 1.0)
    if abs(homogeneous[2]) < 1e-12:
        return np.asarray([math.nan, math.nan])
    return homogeneous[:2] / homogeneous[2]


def triangulate_weighted_dlt(
    observations: list[tuple[np.ndarray, float, float, float]],
) -> np.ndarray:
    equations = []
    for projection, x, y, likelihood in observations:
        weight = math.sqrt(max(likelihood, 1e-6))
        equations.append(weight * (x * projection[2] - projection[0]))
        equations.append(weight * (y * projection[2] - projection[1]))
    _, _, vh = np.linalg.svd(np.asarray(equations), full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full(3, math.nan)
    return homogeneous[:3] / homogeneous[3]


def triangulation_rows(
    manifest_rows: list[dict],
    predictions: dict[tuple[str, str], dict],
    pcutoff: float,
) -> list[dict]:
    samples = {}
    for sample in manifest_rows:
        annotation = json.loads(
            Path(sample["annotation"]).read_text(encoding="utf-8")
        )
        samples[(sample["frame"], sample["camera_suffix"])] = {
            "sample": sample,
            "annotation": annotation,
            "gt": {item["name"]: item for item in annotation.get("keypoints", [])},
            "projection": camera_projection(annotation),
        }

    rows = []
    for frame in sorted({sample["frame"] for sample in manifest_rows}):
        for condition in ("raw", "styled"):
            for dlc_name, blender_name in DLC_TO_BLENDER.items():
                available = []
                gt_points = []
                gt_observations = []
                for camera in ("A", "B", "C"):
                    entry = samples.get((frame, camera))
                    if entry is None:
                        continue
                    image_path = str(
                        Path(entry["sample"][f"{condition}_image"]).resolve()
                    )
                    prediction = predictions.get((camera, condition), {}).get(
                        image_path, {}
                    ).get(dlc_name)
                    gt = entry["gt"].get(blender_name)
                    if gt and "joint_world_xyz" in gt:
                        gt_points.append(
                            np.asarray(gt["joint_world_xyz"], dtype=np.float64)
                        )
                    if (
                        gt
                        and gt.get("visible")
                        and "x" in gt
                        and "y" in gt
                    ):
                        gt_observations.append(
                            (
                                entry["projection"],
                                float(gt["x"]),
                                float(gt["y"]),
                                1.0,
                            )
                        )
                    if (
                        not finite_prediction(prediction)
                        or float(prediction.get("likelihood", 0.0)) < pcutoff
                        or gt is None
                        or "x" not in gt
                        or "y" not in gt
                    ):
                        continue
                    available.append(
                        {
                            "camera": camera,
                            "projection": entry["projection"],
                            "prediction": prediction,
                            "gt": gt,
                        }
                    )
                if len(available) < 2:
                    continue

                if gt_points:
                    gt_world = np.mean(np.stack(gt_points), axis=0)
                elif len(gt_observations) >= 2:
                    gt_world = triangulate_weighted_dlt(gt_observations)
                else:
                    continue
                if not np.isfinite(gt_world).all():
                    continue
                combinations = list(itertools.combinations(available, 2))
                if len(available) >= 3:
                    combinations.append(tuple(available))
                for combination in combinations:
                    observations = [
                        (
                            item["projection"],
                            float(item["prediction"]["x"]),
                            float(item["prediction"]["y"]),
                            float(item["prediction"]["likelihood"]),
                        )
                        for item in combination
                    ]
                    prediction_world = triangulate_weighted_dlt(observations)
                    if not np.isfinite(prediction_world).all():
                        continue
                    prediction_reprojection_errors = []
                    gt_reprojection_errors = []
                    likelihoods = []
                    for item in combination:
                        observed = np.asarray(
                            [
                                float(item["prediction"]["x"]),
                                float(item["prediction"]["y"]),
                            ]
                        )
                        prediction_projected = project_point(
                            item["projection"], prediction_world
                        )
                        prediction_reprojection_errors.append(
                            float(np.linalg.norm(prediction_projected - observed))
                        )
                        gt_projected = project_point(
                            item["projection"], gt_world
                        )
                        gt_uv = np.asarray(
                            [float(item["gt"]["x"]), float(item["gt"]["y"])]
                        )
                        gt_reprojection_errors.append(
                            float(np.linalg.norm(gt_projected - gt_uv))
                        )
                        likelihoods.append(
                            float(item["prediction"]["likelihood"])
                        )
                    error_3d = float(np.linalg.norm(prediction_world - gt_world))
                    rows.append(
                        {
                            "frame": frame,
                            "condition": condition,
                            "view_set": "".join(
                                item["camera"] for item in combination
                            ),
                            "view_count": len(combination),
                            "dlc_keypoint": dlc_name,
                            "blender_keypoint": blender_name,
                            "visible_view_count": sum(
                                bool(item["gt"].get("visible"))
                                for item in combination
                            ),
                            "mean_likelihood": float(np.mean(likelihoods)),
                            "min_likelihood": float(np.min(likelihoods)),
                            "gt_world_x": float(gt_world[0]),
                            "gt_world_y": float(gt_world[1]),
                            "gt_world_z": float(gt_world[2]),
                            "pred_world_x": float(prediction_world[0]),
                            "pred_world_y": float(prediction_world[1]),
                            "pred_world_z": float(prediction_world[2]),
                            "error_3d_world_units": error_3d,
                            "prediction_reprojection_rmse_px": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(
                                            prediction_reprojection_errors
                                        )
                                    )
                                )
                            ),
                            "gt_camera_matrix_reprojection_rmse_px": float(
                                np.sqrt(
                                    np.mean(np.square(gt_reprojection_errors))
                                )
                            ),
                        }
                    )
    return rows


def summarize_triangulation(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["view_set"])].append(row)
    summaries = []
    for (condition, view_set), group in sorted(grouped.items()):
        errors = np.asarray(
            [row["error_3d_world_units"] for row in group], dtype=np.float64
        )
        reprojection = np.asarray(
            [row["prediction_reprojection_rmse_px"] for row in group],
            dtype=np.float64,
        )
        camera_check = np.asarray(
            [row["gt_camera_matrix_reprojection_rmse_px"] for row in group],
            dtype=np.float64,
        )
        consistent = [
            row
            for row in group
            if row["prediction_reprojection_rmse_px"] <= 20.0
        ]
        consistent_errors = np.asarray(
            [row["error_3d_world_units"] for row in consistent],
            dtype=np.float64,
        )
        summaries.append(
            {
                "condition": condition,
                "view_set": view_set,
                "triangulated_count": len(group),
                "mean_error_3d_world_units": float(np.mean(errors)),
                "median_error_3d_world_units": float(np.median(errors)),
                "rmse_3d_world_units": float(np.sqrt(np.mean(errors**2))),
                "pck_1_world_unit": float(np.mean(errors <= 1.0)),
                "pck_2_world_units": float(np.mean(errors <= 2.0)),
                "pck_5_world_units": float(np.mean(errors <= 5.0)),
                "mean_prediction_reprojection_rmse_px": float(
                    np.mean(reprojection)
                ),
                "median_prediction_reprojection_rmse_px": float(
                    np.median(reprojection)
                ),
                "consistent_at_20px_count": len(consistent),
                "consistent_at_20px_fraction": len(consistent) / len(group),
                "consistent_mean_error_3d_world_units": (
                    float(np.mean(consistent_errors))
                    if len(consistent_errors)
                    else math.nan
                ),
                "consistent_median_error_3d_world_units": (
                    float(np.median(consistent_errors))
                    if len(consistent_errors)
                    else math.nan
                ),
                "mean_gt_camera_matrix_reprojection_rmse_px": float(
                    np.mean(camera_check)
                ),
            }
        )
    return summaries


def draw_overlay(
    image_path: Path,
    annotation_path: Path,
    predictions: dict[str, dict] | None,
    pcutoff: float,
    missing_text: str | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    gt_by_name = load_gt(annotation_path)
    predictions = predictions or {}
    for dlc_name, blender_name in DLC_TO_BLENDER.items():
        gt = gt_by_name.get(blender_name)
        if not gt or not gt.get("visible") or "x" not in gt or "y" not in gt:
            continue
        gx, gy = float(gt["x"]), float(gt["y"])
        draw.ellipse((gx - 4, gy - 4, gx + 4, gy + 4), fill=COLORS["gt"])
        prediction = predictions.get(dlc_name)
        if not finite_prediction(prediction):
            continue
        if float(prediction.get("likelihood", 0.0)) < pcutoff:
            continue
        px, py = float(prediction["x"]), float(prediction["y"])
        draw.line((gx, gy, px, py), fill=COLORS["line"], width=2)
        draw.line((px - 5, py - 5, px + 5, py + 5), fill=COLORS["prediction"], width=3)
        draw.line((px - 5, py + 5, px + 5, py - 5), fill=COLORS["prediction"], width=3)
    if missing_text:
        draw.rectangle((0, 0, 330, 34), fill=(0, 0, 0))
        draw.text((8, 8), missing_text, fill=COLORS["text"])
    return image


def labeled_thumbnail(image: Image.Image, label: str, width: int = 400) -> Image.Image:
    height = round(image.height * width / image.width)
    thumbnail = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 28), (15, 15, 15))
    canvas.paste(thumbnail, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill=COLORS["text"])
    return canvas


def write_contact_sheets(
    manifest_rows: list[dict],
    predictions: dict[tuple[str, str], dict],
    ready_views: set[str],
    output: Path,
    pcutoff: float,
) -> None:
    contact_dir = output / "contact_sheets"
    overlay_dir = output / "overlays"
    contact_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    by_frame = defaultdict(list)
    for row in manifest_rows:
        by_frame[row["frame"]].append(row)
    for frame, samples in sorted(by_frame.items()):
        camera_rows = []
        for sample in sorted(samples, key=lambda item: item["view_id"]):
            camera = sample["camera_suffix"]
            styled_path = Path(sample["styled_image"])
            styled_predictions = predictions.get((camera, "styled"), {}).get(
                str(styled_path.resolve())
            )
            missing_text = None
            if camera not in ready_views:
                missing_text = f"Cam{camera}: checkpoint missing"
            overlay = draw_overlay(
                styled_path,
                Path(sample["annotation"]),
                styled_predictions,
                pcutoff,
                missing_text,
            )
            overlay_path = overlay_dir / f"frame_{frame:06d}_cam{camera}.png"
            overlay.save(overlay_path)
            camera_rows.append(
                Image.new(
                    "RGB",
                    (1200, labeled_thumbnail(overlay, "").height),
                    (0, 0, 0),
                )
            )
            panels = [
                labeled_thumbnail(Image.open(sample["raw_image"]).convert("RGB"), f"Cam{camera} raw"),
                labeled_thumbnail(Image.open(styled_path).convert("RGB"), f"Cam{camera} styled"),
                labeled_thumbnail(overlay, f"Cam{camera} GT green / DLC red"),
            ]
            row_image = camera_rows[-1]
            for index, panel in enumerate(panels):
                row_image.paste(panel, (index * 400, 0))
        sheet = Image.new(
            "RGB",
            (1200, sum(row.height for row in camera_rows)),
            (0, 0, 0),
        )
        y = 0
        for row in camera_rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(contact_dir / f"frame_{frame:06d}.png")


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    configs = parse_configs(args.config)
    snapshot_indices = parse_snapshots(args.snapshot)
    reset_output(args.output, args.overwrite)
    manifest_rows = read_manifest(args.manifest)

    predictions = {}
    statuses = {}
    for camera in ("A", "B", "C"):
        camera_rows = [
            row for row in manifest_rows if row["camera_suffix"] == camera
        ]
        if not camera_rows:
            statuses[camera] = "no_samples"
            continue
        if camera not in configs:
            statuses[camera] = "missing_checkpoint"
            continue
        for condition in ("raw", "styled"):
            images = [Path(row[f"{condition}_image"]) for row in camera_rows]
            predictions[(camera, condition)] = run_condition(
                configs[camera],
                images,
                args.output / "predictions" / f"cam{camera}" / condition,
                args.device,
                args.pcutoff,
                args.seed,
                snapshot_indices.get(camera, -1),
            )
        statuses[camera] = "ready"

    metrics = metric_rows(manifest_rows, predictions)
    summaries = summarize(metrics, args.pcutoff)
    metrics_path = args.output / "per_keypoint_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    triangulation_metrics = triangulation_rows(
        manifest_rows, predictions, args.pcutoff
    )
    triangulation_summaries = summarize_triangulation(triangulation_metrics)
    triangulation_metrics_path = args.output / "triangulation_metrics.csv"
    triangulation_summary_path = args.output / "triangulation_summary.csv"
    if triangulation_metrics:
        with triangulation_metrics_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(triangulation_metrics[0])
            )
            writer.writeheader()
            writer.writerows(triangulation_metrics)
    if triangulation_summaries:
        with triangulation_summary_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(triangulation_summaries[0])
            )
            writer.writeheader()
            writer.writerows(triangulation_summaries)
    outlier_2d = sorted(
        (
            row
            for row in metrics
            if math.isfinite(row["error_px"])
            and row["likelihood"] >= args.pcutoff
            and row["error_px"] >= 100.0
        ),
        key=lambda row: row["error_px"],
        reverse=True,
    )
    outlier_3d = sorted(
        (
            row
            for row in triangulation_metrics
            if row["error_3d_world_units"] >= 5.0
            or row["prediction_reprojection_rmse_px"] >= 20.0
        ),
        key=lambda row: row["error_3d_world_units"],
        reverse=True,
    )
    outlier_2d_path = args.output / "outliers_2d.csv"
    outlier_3d_path = args.output / "outliers_3d.csv"
    if outlier_2d:
        with outlier_2d_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(outlier_2d[0]))
            writer.writeheader()
            writer.writerows(outlier_2d)
    if outlier_3d:
        with outlier_3d_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(outlier_3d[0]))
            writer.writeheader()
            writer.writerows(outlier_3d)
    write_contact_sheets(
        manifest_rows,
        predictions,
        {view for view, status in statuses.items() if status == "ready"},
        args.output,
        args.pcutoff,
    )
    report = {
        "manifest": str(args.manifest),
        "configs": {view: str(path) for view, path in configs.items()},
        "snapshot_indices": {
            view: snapshot_indices.get(view, -1) for view in configs
        },
        "statuses": statuses,
        "pcutoff": args.pcutoff,
        "seed": args.seed,
        "mapping": DLC_TO_BLENDER,
        "summary": summaries,
        "metrics": str(metrics_path),
        "triangulation_summary": triangulation_summaries,
        "triangulation_metrics": (
            str(triangulation_metrics_path) if triangulation_metrics else None
        ),
        "outliers_2d": str(outlier_2d_path) if outlier_2d else None,
        "outliers_3d": str(outlier_3d_path) if outlier_3d else None,
        "contact_sheets": str(args.output / "contact_sheets"),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

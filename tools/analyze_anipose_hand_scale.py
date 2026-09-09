#!/usr/bin/env python3
"""Summarize aligned Anipose hand dimensions and compare them with Blender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIDES = ("l", "r")
FINGERS = ("1", "2", "3", "4")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=project_root / "BlenderPyMouseProjects" / "PseudoSynthetic",
    )
    parser.add_argument("--model-measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def point(row: pd.Series, name: str) -> np.ndarray:
    return row[[f"{name}_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float64)


def trial_observations(trial_dir: Path) -> list[dict[str, object]]:
    aligned = pd.read_csv(
        trial_dir / "alignment" / "aligned_keypoints_blender.csv"
    )
    descriptors = pd.read_csv(trial_dir / "alignment" / "hand_descriptors.csv")
    if "fnum" in aligned:
        by_frame = {
            int(frame): index for index, frame in enumerate(aligned["fnum"].to_numpy())
        }
    else:
        by_frame = {index: index for index in range(len(aligned))}

    output = []
    for _, descriptor in descriptors[descriptors["valid"] == True].iterrows():
        frame = int(descriptor["frame"])
        side = str(descriptor["side"])
        if frame not in by_frame or side not in SIDES:
            continue
        row = aligned.iloc[by_frame[frame]]
        palm = point(row, f"{side}_mid_A")
        joints = [point(row, f"{side}{finger}_joint_A") for finger in FINGERS]
        middles = [point(row, f"{side}{finger}_mid_A") for finger in FINGERS]
        tips = [point(row, f"{side}{finger}_tip_A") for finger in FINGERS]
        cloud = np.stack([palm, *joints, *middles, *tips])
        pairwise = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=2)
        cloud_radii = np.linalg.norm(cloud - cloud.mean(axis=0), axis=1)
        observation: dict[str, object] = {
            "trial": trial_dir.name,
            "side": side,
            "frame": frame,
            "root_width": np.linalg.norm(joints[-1] - joints[0]),
            "tip_width": np.linalg.norm(tips[-1] - tips[0]),
            "mean_palm_to_root": np.mean(
                [np.linalg.norm(value - palm) for value in joints]
            ),
            "mean_palm_to_tip": np.mean(
                [np.linalg.norm(value - palm) for value in tips]
            ),
            "max_palm_to_tip": np.max(
                [np.linalg.norm(value - palm) for value in tips]
            ),
            "cloud_radius_p95": np.quantile(cloud_radii, 0.95),
            "cloud_radius_max": cloud_radii.max(),
            "cloud_diameter": pairwise.max(),
        }
        for index, (joint, middle, tip) in enumerate(
            zip(joints, middles, tips), start=1
        ):
            proximal = np.linalg.norm(middle - joint)
            distal = np.linalg.norm(tip - middle)
            observation[f"finger_{index}_proximal"] = proximal
            observation[f"finger_{index}_distal"] = distal
            observation[f"finger_{index}_total"] = proximal + distal
        output.append(observation)
    return output


def distribution(values: pd.Series) -> dict[str, float | int]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "count": int(len(finite)),
        "median": float(finite.median()),
        "p10": float(finite.quantile(0.10)),
        "p25": float(finite.quantile(0.25)),
        "p75": float(finite.quantile(0.75)),
        "p90": float(finite.quantile(0.90)),
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def model_summary(measurements: dict) -> dict[str, float]:
    sides = measurements["sides"]
    root_width = median(
        [sides[side]["finger_root_width_index_to_pinky"] for side in ("L", "R")]
    )
    palm_length = median(
        [sides[side]["palm_bone_length"] for side in ("L", "R")]
    )
    mean_palm_to_root = median(
        [
            sides[side]["fingers"][finger]["root_from_palm_head"]
            for side in ("L", "R")
            for finger in ("Index", "Middle", "Ring", "Pinky")
        ]
    )
    finger_totals = {
        finger: median(
            [
                sides[side]["fingers"][finger]["total_length"]
                for side in ("L", "R")
            ]
        )
        for finger in ("Index", "Middle", "Ring", "Pinky")
    }
    palm_to_tip = median(
        [
            sides[side]["fingers"][finger]["tip_from_palm_head"]
            for side in ("L", "R")
            for finger in ("Index", "Middle", "Ring", "Pinky")
        ]
    )
    mesh_radius_p95 = median(
        [sides[side]["weighted_mesh"]["radius_p95"] for side in ("L", "R")]
    )
    mesh_axis_p05_p95 = median(
        [
            axis["p05_p95_extent"]
            for side in ("L", "R")
            for axis in sides[side]["weighted_mesh"]["world_axis_extents"]
        ]
    )
    return {
        "root_width": root_width,
        "palm_bone_length": palm_length,
        "mean_palm_to_root": mean_palm_to_root,
        "mean_finger_total": median(list(finger_totals.values())),
        "palm_to_tip": palm_to_tip,
        "weighted_mesh_radius_p95": mesh_radius_p95,
        "weighted_mesh_typical_axis_extent": mesh_axis_p05_p95,
        **{f"{finger.lower()}_total": value for finger, value in finger_totals.items()},
    }


def main() -> None:
    args = parse_args()
    observations = []
    for trial_dir in sorted(args.projects_root.iterdir()):
        if (trial_dir / "alignment" / "aligned_keypoints_blender.csv").is_file():
            observations.extend(trial_observations(trial_dir))
    if not observations:
        raise RuntimeError("No aligned hand observations found")
    table = pd.DataFrame(observations)
    numeric_columns = [
        column
        for column in table.columns
        if column not in {"trial", "side", "frame"}
    ]
    trial_side_medians = table.groupby(["trial", "side"])[numeric_columns].median()
    anipose = {
        column: {
            "pooled": distribution(table[column]),
            "trial_side_median_distribution": distribution(
                trial_side_medians[column]
            ),
        }
        for column in numeric_columns
    }

    model_measurements = json.loads(
        args.model_measurements.read_text(encoding="utf-8")
    )
    model = model_summary(model_measurements)
    comparisons = {
        "root_width_anipose_over_model": (
            anipose["root_width"]["trial_side_median_distribution"]["median"]
            / model["root_width"]
        ),
        "mean_finger_total_anipose_over_model": (
            median(
                [
                    anipose[f"finger_{index}_total"][
                        "trial_side_median_distribution"
                    ]["median"]
                    for index in range(1, 5)
                ]
            )
            / model["mean_finger_total"]
        ),
        "mean_palm_to_tip_anipose_over_model": (
            anipose["mean_palm_to_tip"]["trial_side_median_distribution"]["median"]
            / model["palm_to_tip"]
        ),
        "mean_palm_to_root_anipose_over_model": (
            anipose["mean_palm_to_root"]["trial_side_median_distribution"]["median"]
            / model["mean_palm_to_root"]
        ),
        "cloud_radius_p95_anipose_over_weighted_mesh": (
            anipose["cloud_radius_p95"]["trial_side_median_distribution"]["median"]
            / model["weighted_mesh_radius_p95"]
        ),
    }
    report = {
        "projects_root": str(args.projects_root.resolve()),
        "model_measurements": str(args.model_measurements.resolve()),
        "valid_hand_observations": len(table),
        "trial_side_groups": len(trial_side_medians),
        "anipose": anipose,
        "model": model,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Overlay GT, original, and finetuned fixed camera rigs in one coordinate system."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
WATER_EVAL_PATH = PROJECT_ROOT / "water-reaching-dataset" / "evaluate_vggt_omega_keypoints.py"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_vggt_omega_blendedmvs import align_predicted_to_gt_world  # noqa: E402
from export_real_vggt_omega_frame_pointcloud import read_anipose_extrinsics  # noqa: E402
from export_vggt_gt_original_finetuned_comparison import camera_center, camera_metrics  # noqa: E402


CAMERAS = ("A", "B", "C")
RIG_STYLES = {
    "Ground truth": {"color": "#202124", "marker": "o", "line": "-", "short": "GT"},
    "Original": {"color": "#2878b5", "marker": "^", "line": "--", "short": "O"},
    "Finetuned": {"color": "#2a9d6f", "marker": "s", "line": "-.", "short": "FT"},
}


def load_water_eval():
    spec = importlib.util.spec_from_file_location("water_eval_fixedrig_overlay", WATER_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {WATER_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


water_eval = load_water_eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--original-fixedrel", type=Path, required=True)
    parser.add_argument("--finetuned-fixedrel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    return parser.parse_args()


def load_fixed_relative_extrinsics(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["fixed_relative_extrinsics"]
    return np.stack([np.asarray(values[cam], dtype=np.float64) for cam in CAMERAS])


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    return np.stack([camera_center(item) for item in extrinsics])


def set_equal_axes(ax, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low) / 2.0), 1e-6) * 1.18
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def draw_rig(ax, label: str, extrinsics: np.ndarray, arrow_length: float) -> None:
    style = RIG_STYLES[label]
    centers = camera_centers(extrinsics)
    ax.plot(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        color=style["color"],
        linestyle=style["line"],
        linewidth=1.8,
        alpha=0.9,
        label=label,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=style["color"],
        marker=style["marker"],
        s=62,
        edgecolors="white",
        linewidths=0.7,
        depthshade=False,
    )
    for extrinsic, center in zip(extrinsics, centers):
        forward = extrinsic[:3, :3].T[:, 2]
        ax.quiver(
            *center,
            *forward,
            length=arrow_length,
            color=style["color"],
            linewidth=1.5,
            arrow_length_ratio=0.28,
        )


def draw_correspondences(
    ax,
    gt: np.ndarray,
    original: np.ndarray,
    finetuned: np.ndarray,
) -> None:
    gt_centers = camera_centers(gt)
    for predicted, color in ((original, "#2878b5"), (finetuned, "#2a9d6f")):
        predicted_centers = camera_centers(predicted)
        for target, estimate in zip(gt_centers, predicted_centers):
            ax.plot(
                [target[0], estimate[0]],
                [target[1], estimate[1]],
                [target[2], estimate[2]],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )


def draw_projection(
    ax,
    rigs: tuple[tuple[str, np.ndarray], ...],
    axes: tuple[int, int],
    title: str,
) -> None:
    gt_centers = camera_centers(rigs[0][1])
    for label, extrinsics in rigs:
        style = RIG_STYLES[label]
        centers = camera_centers(extrinsics)
        ax.plot(
            centers[:, axes[0]],
            centers[:, axes[1]],
            color=style["color"],
            linestyle=style["line"],
            linewidth=1.8,
            label=label,
        )
        ax.scatter(
            centers[:, axes[0]],
            centers[:, axes[1]],
            c=style["color"],
            marker=style["marker"],
            s=58,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        if label != "Ground truth":
            for target, estimate in zip(gt_centers, centers):
                ax.plot(
                    [target[axes[0]], estimate[axes[0]]],
                    [target[axes[1]], estimate[axes[1]]],
                    color=style["color"],
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.75,
                )
    span = np.maximum(np.ptp(gt_centers[:, list(axes)], axis=0), 1.0)
    for cam, center in zip(CAMERAS, gt_centers):
        ax.annotate(
            f"cam{cam}",
            (center[axes[0]], center[axes[1]]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#202124",
        )
    ax.set_xlabel(("x", "y", "z")[axes[0]])
    ax.set_ylabel(("x", "y", "z")[axes[1]])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.12, y=0.12)
    ax.grid(alpha=0.24)
    ax.legend(loc="best", fontsize=8)


def main() -> None:
    args = parse_args()
    trial = args.trial.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output = (
        args.metrics_output.resolve()
        if args.metrics_output is not None
        else output.with_suffix(".json")
    )

    calibration = read_anipose_extrinsics(trial / "calibration" / "calibration.toml")
    gt = water_eval.relative_extrinsics_anchor_first(calibration)
    original_raw = load_fixed_relative_extrinsics(args.original_fixedrel.resolve())
    finetuned_raw = load_fixed_relative_extrinsics(args.finetuned_fixedrel.resolve())
    original, original_alignment = align_predicted_to_gt_world(original_raw, gt)
    finetuned, finetuned_alignment = align_predicted_to_gt_world(finetuned_raw, gt)

    all_centers = np.concatenate(
        [camera_centers(rig) for rig in (gt, original, finetuned)],
        axis=0,
    )
    arrow_length = max(float(np.max(np.ptp(all_centers, axis=0))), 1e-6) * 0.11
    rigs = (
        ("Ground truth", gt),
        ("Original", original),
        ("Finetuned", finetuned),
    )
    fig = plt.figure(figsize=(17.0, 5.8), dpi=190)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for label, rig in rigs:
        draw_rig(ax, label, rig, arrow_length)
    draw_correspondences(ax, gt, original, finetuned)
    for cam, center in zip(CAMERAS, camera_centers(gt)):
        ax.text(center[0], center[1], center[2], f" cam{cam}", color="#202124", fontsize=9)
    set_equal_axes(ax, all_centers)
    ax.view_init(elev=24, azim=-58)
    ax.set_title("Perspective")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8)

    draw_projection(fig.add_subplot(1, 3, 2), rigs, (0, 1), "XY projection")
    draw_projection(fig.add_subplot(1, 3, 3), rigs, (0, 2), "XZ projection")
    fig.suptitle(
        "Fixed relative camera rigs in the shared calibration coordinate system",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "trial": str(trial),
        "alignment_note": (
            "Original and finetuned fixed-relative rigs are independently aligned to the "
            "calibration rig using one Sim(3) over all three camera centers."
        ),
        "original": {
            "alignment": original_alignment,
            **camera_metrics(original, gt),
        },
        "finetuned": {
            "alignment": finetuned_alignment,
            **camera_metrics(finetuned, gt),
        },
        "output": str(output),
    }
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

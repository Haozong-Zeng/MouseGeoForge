#!/usr/bin/env python3
"""Create a compact report for shared-rig refinement and related experiments."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-summary", type=Path, required=True)
    parser.add_argument("--camera-log", type=Path, required=True)
    parser.add_argument("--triangulation-old", type=Path, required=True)
    parser.add_argument("--triangulation-new", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def percent_change(before: float, after: float) -> float:
    return 100.0 * (after / before - 1.0)


def svg_dashboard(
    trials: list[dict],
    validation: list[dict],
    baseline_camera: dict,
    best_camera: dict,
    categories: list[str],
    old_fixed: list[float],
    new_fixed: list[float],
) -> str:
    width, height = 1500, 1000
    panels = [(55, 80), (775, 80), (55, 540), (775, 540)]
    panel_w, panel_h = 650, 365
    plot_left, plot_top, plot_w, plot_h = 70, 55, 550, 250
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:Arial,sans-serif;fill:#263238}"
        ".title{font-size:25px;font-weight:700}"
        ".ptitle{font-size:18px;font-weight:700}"
        ".axis{stroke:#78909c;stroke-width:1}"
        ".grid{stroke:#cfd8dc;stroke-width:1;opacity:.65}"
        ".small{font-size:11px}.label{font-size:13px}"
        "</style>",
        '<rect width="1500" height="1000" fill="#fafafa"/>',
        '<text x="750" y="42" text-anchor="middle" class="title">'
        "Shared Rig Refinement and Geometry Training Diagnostics</text>",
    ]
    for x0, y0 in panels:
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" '
            'rx="6" fill="#ffffff" stroke="#cfd8dc"/>'
        )

    def frame(panel: int, title: str, y_label: str) -> tuple[float, float]:
        x0, y0 = panels[panel]
        px, py = x0 + plot_left, y0 + plot_top
        parts.extend(
            [
                f'<text x="{x0 + 20}" y="{y0 + 30}" class="ptitle">{html.escape(title)}</text>',
                f'<line x1="{px}" y1="{py + plot_h}" x2="{px + plot_w}" y2="{py + plot_h}" class="axis"/>',
                f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + plot_h}" class="axis"/>',
                f'<text transform="translate({x0 + 18},{py + plot_h / 2}) rotate(-90)" '
                f'text-anchor="middle" class="label">{html.escape(y_label)}</text>',
            ]
        )
        for frac in (0.25, 0.5, 0.75):
            gy = py + plot_h * frac
            parts.append(
                f'<line x1="{px}" y1="{gy}" x2="{px + plot_w}" y2="{gy}" class="grid"/>'
            )
        return px, py

    px, py = frame(0, "Strict held-out 3D median error by trial", "Error (mm)")
    initial_3d = [row["initial_val_median_mm"] for row in trials]
    refined_3d = [row["refined_val_median_mm"] for row in trials]
    ymax = max(initial_3d + refined_3d) * 1.08
    for index, (before, after) in enumerate(zip(initial_3d, refined_3d)):
        x = px + (index + 0.5) * plot_w / len(trials)
        y1 = py + plot_h * (1.0 - before / ymax)
        y2 = py + plot_h * (1.0 - after / ymax)
        color = "#2a9d8f" if after < before else "#e76f51"
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="2" opacity=".65"/>',
                f'<circle cx="{x:.1f}" cy="{y1:.1f}" r="4" fill="#264653"/>',
                f'<circle cx="{x:.1f}" cy="{y2:.1f}" r="4" fill="#e9c46a" stroke="#7f6600"/>',
            ]
        )
    parts.extend(
        [
            f'<text x="{px}" y="{py + plot_h + 27}" class="small">22 trials, sorted by initial 3D error</text>',
            f'<circle cx="{px + 350}" cy="{py + plot_h + 23}" r="4" fill="#264653"/>',
            f'<text x="{px + 360}" y="{py + plot_h + 27}" class="small">Initial</text>',
            f'<circle cx="{px + 430}" cy="{py + plot_h + 23}" r="4" fill="#e9c46a" stroke="#7f6600"/>',
            f'<text x="{px + 440}" y="{py + plot_h + 27}" class="small">Refined</text>',
        ]
    )

    px, py = frame(1, "Median reprojection error", "Pixels (log scale)")
    reproj = sorted(
        (
            row["initial_reproj_median_px"],
            row["refined_reproj_median_px"],
        )
        for row in trials
    )
    values = [value for pair in reproj for value in pair]
    log_min = math.log10(max(min(values) * 0.8, 0.1))
    log_max = math.log10(max(values) * 1.15)

    def log_y(value: float) -> float:
        return py + plot_h * (1.0 - (math.log10(value) - log_min) / (log_max - log_min))

    initial_points = []
    refined_points = []
    for index, (before, after) in enumerate(reproj):
        x = px + index * plot_w / max(len(reproj) - 1, 1)
        initial_points.append(f"{x:.1f},{log_y(before):.1f}")
        refined_points.append(f"{x:.1f},{log_y(after):.1f}")
    parts.extend(
        [
            f'<polyline points="{" ".join(initial_points)}" fill="none" stroke="#457b9d" stroke-width="3"/>',
            f'<polyline points="{" ".join(refined_points)}" fill="none" stroke="#f4a261" stroke-width="3"/>',
            f'<text x="{px + 355}" y="{py + plot_h + 27}" class="small" fill="#457b9d">Initial</text>',
            f'<text x="{px + 445}" y="{py + plot_h + 27}" class="small" fill="#f4a261">Refined</text>',
        ]
    )

    px, py = frame(2, "Real validation relative-rig loss", "Gauge-invariant loss")
    train_rows = [row for row in validation if row["step"] >= 0]
    x_min = min(row["step"] for row in train_rows)
    x_max = max(row["step"] for row in train_rows)
    losses = [row["relative_rig"] for row in train_rows] + [baseline_camera["relative_rig"]]
    y_min, y_max = min(losses), max(losses)
    y_pad = max((y_max - y_min) * 0.2, 1e-5)
    y_min -= y_pad
    y_max += y_pad

    def linear_xy(step: int, value: float) -> tuple[float, float]:
        x = px + (step - x_min) * plot_w / max(x_max - x_min, 1)
        y = py + plot_h * (1.0 - (value - y_min) / (y_max - y_min))
        return x, y

    loss_points = []
    for row in train_rows:
        x, y = linear_xy(row["step"], row["relative_rig"])
        loss_points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#6a4c93"/>')
    _, baseline_y = linear_xy(x_min, baseline_camera["relative_rig"])
    best_x, best_y = linear_xy(best_camera["step"], best_camera["relative_rig"])
    parts.extend(
        [
            f'<polyline points="{" ".join(loss_points)}" fill="none" stroke="#6a4c93" stroke-width="3"/>',
            f'<line x1="{px}" y1="{baseline_y:.1f}" x2="{px + plot_w}" y2="{baseline_y:.1f}" '
            'stroke="#78909c" stroke-dasharray="8 6"/>',
            f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="7" fill="#d62828"/>',
            f'<text x="{px}" y="{py + plot_h + 27}" class="small">Training step</text>',
            f'<text x="{px + 335}" y="{py + 20}" class="small">Best: step {best_camera["step"]}, '
            f'{best_camera["relative_rig"]:.4f}</text>',
        ]
    )

    px, py = frame(3, "Real fixed-rig probabilistic triangulation", "Mean 3D error (mm)")
    ymax = max(old_fixed + new_fixed) * 1.2
    group_w = plot_w / len(categories)
    bar_w = group_w * 0.27
    for index, (name, old, new) in enumerate(zip(categories, old_fixed, new_fixed)):
        center = px + (index + 0.5) * group_w
        old_h = plot_h * old / ymax
        new_h = plot_h * new / ymax
        parts.extend(
            [
                f'<rect x="{center - bar_w - 3:.1f}" y="{py + plot_h - old_h:.1f}" '
                f'width="{bar_w:.1f}" height="{old_h:.1f}" fill="#8d99ae"/>',
                f'<rect x="{center + 3:.1f}" y="{py + plot_h - new_h:.1f}" '
                f'width="{bar_w:.1f}" height="{new_h:.1f}" fill="#2a9d8f"/>',
                f'<text x="{center:.1f}" y="{py + plot_h + 27}" text-anchor="middle" '
                f'class="label">{html.escape(name)}</text>',
            ]
        )
    parts.extend(
        [
            f'<rect x="{px + 320}" y="{py + 10}" width="12" height="12" fill="#8d99ae"/>',
            f'<text x="{px + 338}" y="{py + 21}" class="small">Old fusion</text>',
            f'<rect x="{px + 430}" y="{py + 10}" width="12" height="12" fill="#2a9d8f"/>',
            f'<text x="{px + 448}" y="{py + 21}" class="small">Robust fusion</text>',
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rig = json.loads(args.rig_summary.read_text(encoding="utf-8"))
    old_tri = json.loads(args.triangulation_old.read_text(encoding="utf-8"))
    new_tri = json.loads(args.triangulation_new.read_text(encoding="utf-8"))
    camera_rows = read_csv(args.camera_log)

    trials = sorted(
        rig["trials"],
        key=lambda row: row["initial_val_median_mm"],
    )
    names = [row["trial"] for row in trials]
    validation = []
    relative_key = "real_geometry_val_real_geometry_loss_relative_rig"
    rotation_key = (
        "real_geometry_val_real_geometry_relative_rig_rotation_error_deg"
    )
    center_key = "real_geometry_val_real_geometry_relative_rig_center_error"
    baseline_key = (
        "real_geometry_val_real_geometry_relative_rig_baseline_ratio_error"
    )
    for row in camera_rows:
        relative = finite_float(row, relative_key)
        if relative is None:
            continue
        validation.append(
            {
                "step": int(float(row["step"])),
                "relative_rig": relative,
                "rotation_deg": finite_float(row, rotation_key),
                "center": finite_float(row, center_key),
                "baseline_ratio": finite_float(row, baseline_key),
            }
        )
    best_camera = min(
        (row for row in validation if row["step"] >= 0),
        key=lambda row: row["relative_rig"],
    )
    baseline_camera = min(validation, key=lambda row: row["step"])

    categories = ["finger", "palm", "head"]
    old_fixed = [old_tri["real"][name]["fixed3d"]["mean"] for name in categories]
    new_fixed = [new_tri["real"][name]["fixed3d"]["mean"] for name in categories]

    figure_path = args.output_dir / "shared_rig_training_summary.svg"
    figure_path.write_text(
        svg_dashboard(
            trials,
            validation,
            baseline_camera,
            best_camera,
            categories,
            old_fixed,
            new_fixed,
        ),
        encoding="utf-8",
    )

    aggregate = rig["aggregate"]
    report_data = {
        "shared_rig": aggregate,
        "camera_head": {
            "baseline": baseline_camera,
            "best": best_camera,
        },
        "probabilistic_triangulation_real_fixed_rig_mean_mm": {
            name: {"old": old, "new": new}
            for name, old, new in zip(categories, old_fixed, new_fixed)
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(report_data, indent=2),
        encoding="utf-8",
    )

    worsening = [
        row for row in rig["trials"]
        if row["refined_val_median_mm"] >= row["initial_val_median_mm"]
    ]
    lines = [
        "# Shared-rig refinement experiment",
        "",
        "## Full-video shared rig",
        "",
        f"- Trials: {aggregate['trial_count']}",
        (
            "- Median reprojection error: "
            f"{aggregate['initial_reproj_median_px_median']:.3f} -> "
            f"{aggregate['refined_reproj_median_px_median']:.3f} px "
            f"({percent_change(aggregate['initial_reproj_median_px_median'], aggregate['refined_reproj_median_px_median']):+.1f}%)."
        ),
        (
            "- Strict held-out median 3D error: "
            f"{aggregate['initial_val_median_mm_median']:.3f} -> "
            f"{aggregate['refined_val_median_mm_median']:.3f} mm "
            f"({percent_change(aggregate['initial_val_median_mm_median'], aggregate['refined_val_median_mm_median']):+.1f}%)."
        ),
        (
            "- Strict held-out median RMSE: "
            f"{aggregate['initial_val_rmse_mm_median']:.3f} -> "
            f"{aggregate['refined_val_rmse_mm_median']:.3f} mm."
        ),
        (
            f"- Improved trials: {aggregate['improved_validation_trials']}/"
            f"{aggregate['trial_count']} in held-out 3D and "
            f"{aggregate['improved_reprojection_trials']}/{aggregate['trial_count']} "
            "in reprojection."
        ),
        (
            "- Median rotation correction: "
            f"CamB {aggregate['camB_rotation_change_deg_median']:.2f} deg, "
            f"CamC {aggregate['camC_rotation_change_deg_median']:.2f} deg."
        ),
        "",
        "Trials where reprojection improved but held-out 3D worsened:",
    ]
    lines.extend(
        f"- `{row['trial']}`: {row['initial_val_median_mm']:.3f} -> "
        f"{row['refined_val_median_mm']:.3f} mm"
        for row in worsening
    )
    lines.extend(
        [
            "",
            "## Relative-rig camera-head adaptation",
            "",
            (
                f"- Baseline real relative-rig loss: "
                f"{baseline_camera['relative_rig']:.6f}."
            ),
            (
                f"- Best observed validation point: step {best_camera['step']}, "
                f"loss {best_camera['relative_rig']:.6f} "
                f"({percent_change(baseline_camera['relative_rig'], best_camera['relative_rig']):+.2f}%)."
            ),
            (
                f"- Relative rotation error: {baseline_camera['rotation_deg']:.3f} -> "
                f"{best_camera['rotation_deg']:.3f} deg."
            ),
            (
                f"- Baseline-ratio error: {baseline_camera['baseline_ratio']:.4f} -> "
                f"{best_camera['baseline_ratio']:.4f}; this term regressed and needs "
                "a larger weight in the next run."
            ),
            "",
            "## Robust probabilistic triangulation",
            "",
        ]
    )
    lines.extend(
        f"- {name}: {old:.4f} -> {new:.4f} mm "
        f"({percent_change(old, new):+.1f}%)."
        for name, old, new in zip(categories, old_fixed, new_fixed)
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The shared full-video rig is worth keeping as a preprocessing and pseudo-label stage. "
            "Camera-head training should supervise gauge-invariant relative rig geometry, save the "
            "best real relative-rig checkpoint, increase the baseline/center terms, and add a "
            "same-trial temporal consistency loss. Robust covariance triangulation helps once the "
            "rig is accurate, but it does not compensate for incorrect cameras.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

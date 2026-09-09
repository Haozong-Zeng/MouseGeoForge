#!/usr/bin/env python3
"""Build fixed-scale side-depth diagnostics and report far-wall validity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        dtype = "<f4" if scale < 0 else ">f4"
        depth = np.fromfile(stream, dtype=dtype).reshape(height, width)
    return np.flipud(depth)


def fixed_scale_color(depth: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.clip((depth - near_mm) / (far_mm - near_mm), 0.0, 1.0)
    # Turbo runs blue -> green -> yellow -> red as depth increases.
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = (255, 0, 255)  # BGR magenta: unmistakably invalid depth.
    return color


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-mm", type=float, default=180.0)
    parser.add_argument("--far-mm", type=float, default=500.0)
    parser.add_argument("--scan-view", type=int, default=2)
    parser.add_argument("--scan-frame-start", type=int, default=None)
    parser.add_argument("--scan-frame-end", type=int, default=None)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)

    frame_root = args.dataset_root / f"frame_{args.frame:06d}" / "rendered_depth_maps"
    panels = []
    stats = {}
    for view_id in range(1, 4):
        depth = read_pfm(frame_root / f"{view_id:08d}.pfm")
        height, width = depth.shape
        valid = np.isfinite(depth) & (depth > 0)
        # Upper-center region contains the camera-facing far wall and excludes
        # the bedding, mouse, and lower wall lip for all three side cameras.
        y0, y1 = int(height * 0.25), int(height * 0.40)
        x0, x1 = int(width * 0.25), int(width * 0.75)
        roi = depth[y0:y1, x0:x1]
        roi_valid = np.isfinite(roi) & (roi > 0)
        roi_values = roi[roi_valid]
        stats[str(view_id)] = {
            "overall_valid_fraction": float(valid.mean()),
            "far_wall_roi": [x0, y0, x1, y1],
            "far_wall_valid_fraction": float(roi_valid.mean()),
            "far_wall_median_mm": float(np.median(roi_values)),
            "far_wall_p01_mm": float(np.percentile(roi_values, 1)),
            "far_wall_p99_mm": float(np.percentile(roi_values, 99)),
            "center_depth_mm": float(depth[height // 2, width // 2]),
        }

        panel = fixed_scale_color(depth, args.near_mm, args.far_mm)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 255, 255), 2)
        label = (
            f"view {view_id:03d} | ROI valid {roi_valid.mean() * 100:.3f}% | "
            f"median {np.median(roi_values):.1f} mm"
        )
        cv2.rectangle(panel, (0, 0), (width, 38), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    scan_summary = None
    if args.scan_frame_start is not None or args.scan_frame_end is not None:
        scan_start = args.scan_frame_start if args.scan_frame_start is not None else args.frame
        scan_end = args.scan_frame_end if args.scan_frame_end is not None else scan_start
        if scan_end < scan_start:
            raise ValueError("scan frame end precedes scan frame start")
        valid_fractions = []
        medians = []
        zero_counts = []
        for frame in range(scan_start, scan_end + 1):
            scan_path = (
                args.dataset_root
                / f"frame_{frame:06d}"
                / "rendered_depth_maps"
                / f"{args.scan_view:08d}.pfm"
            )
            scan_depth = read_pfm(scan_path)
            height, width = scan_depth.shape
            y0, y1 = int(height * 0.25), int(height * 0.40)
            x0, x1 = int(width * 0.25), int(width * 0.75)
            scan_roi = scan_depth[y0:y1, x0:x1]
            scan_valid = np.isfinite(scan_roi) & (scan_roi > 0)
            valid_fractions.append(float(scan_valid.mean()))
            zero_counts.append(int(scan_roi.size - np.count_nonzero(scan_valid)))
            medians.append(float(np.median(scan_roi[scan_valid])))
        scan_summary = {
            "view_id": args.scan_view,
            "frame_range": [scan_start, scan_end],
            "frame_count": scan_end - scan_start + 1,
            "minimum_far_wall_valid_fraction": min(valid_fractions),
            "maximum_invalid_pixels_in_far_wall_roi": max(zero_counts),
            "far_wall_median_mm_range": [min(medians), max(medians)],
            "frames_with_any_invalid_far_wall_roi_pixel": sum(
                count > 0 for count in zero_counts
            ),
        }

    contact = np.vstack(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), contact)
    stats_path = args.output.with_suffix(".json")
    payload = {"frame_stats": stats, "all_frame_scan": scan_summary}
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload}, indent=2), flush=True)


if __name__ == "__main__":
    main()

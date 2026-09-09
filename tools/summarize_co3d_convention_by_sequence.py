#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


RATE_SUFFIXES = ("_rate", "_cos")


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def z(value: float) -> float:
    return 0.0 if not np.isfinite(value) else value


def large(value: float) -> float:
    return 5.0 if not np.isfinite(value) else value


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def score(row: dict[str, float]) -> float:
    depth_support = z(row.get("depth_mask_hit_rate_all", float("nan")))
    ply_support = z(row.get("ply_mask_hit_rate_all", float("nan")))
    ply_visible_support = z(row.get("ply_depth_visible_rate_all", float("nan")))
    silhouette_iou = z(row.get("ply_silhouette_iou", float("nan")))
    silhouette_target = z(row.get("ply_silhouette_target_coverage_rate", float("nan")))
    silhouette_precision = z(row.get("ply_silhouette_projection_precision_rate", float("nan")))
    depth_quality = depth_support * z(row.get("depth_agree_10pct_rate", float("nan")))
    rel_penalty = depth_support * min(large(row.get("depth_median_relative_error", float("nan"))), 2.0)
    color_penalty = ply_visible_support * min(large(row.get("ply_visible_color_mae", float("nan"))), 1.0)
    return (
        5.0 * ply_support
        + 6.0 * ply_visible_support
        + 4.0 * silhouette_iou
        + 1.5 * silhouette_target
        + 1.5 * silhouette_precision
        + 5.0 * depth_support
        + 2.0 * depth_quality
        + 0.8 * z(row.get("depth_in_image_rate", float("nan")))
        + 0.6 * z(row.get("ply_in_image_rate", float("nan")))
        + 0.8 * z(row.get("ply_camera_look_cos", float("nan")))
        - rel_penalty
        - color_penalty
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output")
    parser.add_argument(
        "--focus",
        nargs="*",
        default=None,
        help="Candidate names to include in the per-sequence comparison. Defaults to all candidates.",
    )
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    numeric_fields = sorted(
        key
        for key in {k for row in rows for k in row.keys()}
        if key not in {"category", "sequence", "candidate", "description", "diagnostic_kind"}
    )
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["sequence"], row["candidate"])].append(row)

    summary_rows = []
    for (category, sequence, candidate), group_rows in grouped.items():
        out: dict[str, float | str] = {"category": category, "sequence": sequence, "candidate": candidate}
        for field in numeric_fields:
            values = [to_float(row.get(field, "")) for row in group_rows]
            if field.endswith(RATE_SUFFIXES):
                out[field] = finite_mean(values)
            else:
                out[field] = finite_median(values)
        out["diagnostic_score"] = score(out)  # type: ignore[arg-type]
        summary_rows.append(out)

    by_sequence: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in summary_rows:
        by_sequence[(str(row["category"]), str(row["sequence"]))].append(row)

    comparison_rows = []
    candidate_names = sorted({str(row["candidate"]) for row in summary_rows})
    focus_names = args.focus if args.focus is not None else candidate_names
    for (category, sequence), candidates in by_sequence.items():
        focus = [row for row in candidates if row["candidate"] in focus_names]
        if not focus:
            continue
        focus.sort(key=lambda row: float(row["diagnostic_score"]), reverse=True)
        best = focus[0]
        runner_up = focus[1] if len(focus) > 1 else None
        row_out: dict[str, float | str] = {
            "category": category,
            "sequence": sequence,
            "winner": best["candidate"],
            "winner_score": best["diagnostic_score"],
            "runner_up": runner_up["candidate"] if runner_up else "",
            "runner_up_score": runner_up["diagnostic_score"] if runner_up else float("nan"),
            "score_gap": (
                float(best["diagnostic_score"]) - float(runner_up["diagnostic_score"])
                if runner_up
                else float("nan")
            ),
        }
        by_candidate_name = {str(row["candidate"]): row for row in focus}
        for name in focus_names:
            candidate_row = by_candidate_name.get(name)
            prefix = safe_name(name)
            for metric in (
                "diagnostic_score",
                "ply_visible_color_mae",
                "ply_depth_visible_rate_all",
                "depth_mask_hit_rate_all",
                "depth_valid_target_rate",
                "depth_median_relative_error",
                "depth_agree_10pct_rate",
                "depth_self_in_image_rate",
                "depth_self_mask_hit_rate_all",
                "depth_self_median_relative_error",
                "depth_self_agree_10pct_rate",
                "ply_mask_hit_rate_all",
                "ply_silhouette_iou",
                "ply_silhouette_target_coverage_rate",
                "ply_silhouette_projection_precision_rate",
            ):
                row_out[f"{prefix}_{metric}"] = (
                    candidate_row.get(metric, float("nan")) if candidate_row else float("nan")
                )
        comparison_rows.append(row_out)

    comparison_rows.sort(key=lambda row: (row["category"], row["sequence"]))
    out_path = Path(args.output) if args.output else Path(args.csv_path).with_name("sequence_candidate_comparison.csv")
    fields = list(comparison_rows[0].keys()) if comparison_rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in comparison_rows:
        counts[str(row["category"])][str(row["winner"])] += 1

    print(f"wrote {out_path}")
    for category in sorted(counts):
        print(category)
        for winner, count in sorted(counts[category].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {winner}: {count}")


if __name__ == "__main__":
    main()

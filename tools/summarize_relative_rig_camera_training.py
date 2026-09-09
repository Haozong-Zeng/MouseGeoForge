#!/usr/bin/env python3
"""Compare two relative-rig camera-head training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = {
    "rotation_loss": "real_geometry_val_real_geometry_loss_relative_rig_rotation",
    "center_loss": "real_geometry_val_real_geometry_loss_relative_rig_center",
    "baseline_loss": "real_geometry_val_real_geometry_loss_relative_rig_baseline",
    "focal_loss": "real_geometry_val_real_geometry_loss_relative_rig_focal",
    "rotation_error_deg": (
        "real_geometry_val_real_geometry_relative_rig_rotation_error_deg"
    ),
    "center_error": "real_geometry_val_real_geometry_relative_rig_center_error",
    "baseline_ratio_error": (
        "real_geometry_val_real_geometry_relative_rig_baseline_ratio_error"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for source_row in source:
        if not source_row.get(FIELDS["rotation_loss"]):
            continue
        row: dict[str, float | int] = {"step": int(float(source_row["step"]))}
        row.update({name: float(source_row[field]) for name, field in FIELDS.items()})
        row["old_weight_objective"] = (
            row["rotation_loss"]
            + row["center_loss"]
            + 0.5 * row["baseline_loss"]
            + 0.25 * row["focal_loss"]
        )
        row["new_weight_objective"] = (
            row["rotation_loss"]
            + 1.5 * row["center_loss"]
            + 2.0 * row["baseline_loss"]
            + 0.25 * row["focal_loss"]
        )
        rows.append(row)
    return rows


def percent_change(before: float, after: float) -> float:
    return 100.0 * (after / before - 1.0)


def main() -> None:
    args = parse_args()
    old_rows = load_rows(args.previous_run / "train_log.csv")
    new_rows = load_rows(args.new_run / "train_log.csv")
    old_best = min(
        (row for row in old_rows if row["step"] >= 0),
        key=lambda row: row["old_weight_objective"],
    )
    new_best = min(
        (row for row in new_rows if row["step"] >= 0),
        key=lambda row: row["new_weight_objective"],
    )
    new_baseline = next(row for row in new_rows if row["step"] == -1)

    summary = {
        "configuration": {
            "max_steps": 600,
            "lr": 1e-6,
            "center_weight": 1.5,
            "baseline_weight": 2.0,
        },
        "untrained_baseline": new_baseline,
        "previous_best": old_best,
        "new_best": new_best,
        "new_best_checkpoint": str(
            args.new_run
            / "checkpoints"
            / "vggt_omega_relative_rig_camera_best_real_relative_rig.pt"
        ),
    }
    (args.new_run / "training_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Relative-rig camera-head training result",
        "",
        f"- Best checkpoint: step {new_best['step']}.",
        (
            f"- New-weight objective: {new_baseline['new_weight_objective']:.6f} -> "
            f"{new_best['new_weight_objective']:.6f} "
            f"({percent_change(new_baseline['new_weight_objective'], new_best['new_weight_objective']):+.2f}%)."
        ),
        (
            f"- Rotation error: {new_baseline['rotation_error_deg']:.3f} -> "
            f"{new_best['rotation_error_deg']:.3f} deg."
        ),
        (
            f"- Center error: {new_baseline['center_error']:.4f} -> "
            f"{new_best['center_error']:.4f}."
        ),
        (
            f"- Baseline-ratio error: {new_baseline['baseline_ratio_error']:.4f} -> "
            f"{new_best['baseline_ratio_error']:.4f}."
        ),
        "",
        "Compared with the previous run:",
        (
            f"- Rotation: {old_best['rotation_error_deg']:.3f} -> "
            f"{new_best['rotation_error_deg']:.3f} deg."
        ),
        (
            f"- Center: {old_best['center_error']:.4f} -> "
            f"{new_best['center_error']:.4f}."
        ),
        (
            f"- Baseline ratio: {old_best['baseline_ratio_error']:.4f} -> "
            f"{new_best['baseline_ratio_error']:.4f}."
        ),
        "",
        (
            "The stronger baseline term reduced baseline drift relative to the "
            "previous adaptation, but neither run beats the untrained baseline-ratio "
            "error. The next change should be trial-level temporal/shared-rig "
            "consistency rather than a further scalar weight increase."
        ),
        "",
    ]
    report = "\n".join(lines)
    (args.new_run / "TRAINING_RESULT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

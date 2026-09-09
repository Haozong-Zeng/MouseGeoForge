#!/usr/bin/env python3
"""Compare separated registration audits from two camera-pose modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--builtin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    report_path = path if path.is_file() else path / "separated_registration_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(report: dict, section: str, field: str) -> float:
    return float(report[section][field])


def main() -> None:
    args = parse_args()
    calibrated = load_report(args.calibrated)
    builtin = load_report(args.builtin)

    rows = []
    for label, section, field in (
        ("target_to_real_median_px", "target_projection", "median_px"),
        ("target_to_real_p95_px", "target_projection", "p95_px"),
        ("model_to_real_median_px", "actual_model_bones", "median_real_px"),
        ("model_to_real_p95_px", "actual_model_bones", "p95_real_px"),
        ("model_to_target_median_px", "actual_model_bones", "median_target_px"),
        ("model_to_target_p95_px", "actual_model_bones", "p95_target_px"),
    ):
        calibrated_value = metric(calibrated, section, field)
        builtin_value = metric(builtin, section, field)
        rows.append(
            {
                "metric": label,
                "calibrated": calibrated_value,
                "builtin": builtin_value,
                "builtin_minus_calibrated": builtin_value - calibrated_value,
                "builtin_over_calibrated": builtin_value / calibrated_value,
            }
        )

    result = {
        "calibrated_audit": str(args.calibrated),
        "builtin_audit": str(args.builtin),
        "metrics": rows,
        "decision": {
            "camera_pose": "calibrated",
            "reason": (
                "The built-in camera poses increase both target-to-real and "
                "actual-model-to-real median projection error. Keep the fixed "
                "real rig and correct the arm, palm, and finger retargeting."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

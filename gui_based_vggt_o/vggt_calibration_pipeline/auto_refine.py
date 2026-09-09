#!/usr/bin/env python3
"""Run a grid of metric-refinement configurations and select a winner.

For each candidate configuration this driver runs
refine_metric_calibration.optimize_refinement with leave-one-distance-out
cross-validation enabled, writes a comparison table, and copies the winning
calibration into OUTPUT_DIR/best/.

Selection caveat (validated on videos_6yca frame_000051 against Anipose
ground-truth pose shape): LOO distance RMSE is a good overfit detector
*within* the loose-prior family (it flags fxfy / f_shared / loose-k1
overfit), but it is NOT a reliable proxy for 3D pose shape *across*
families. Configurations with tight VGGT priors minimize LOO by suppressing
click influence while inheriting VGGT's rig-shape bias. The grid therefore
marks the shape-validated loose-prior family as selectable and runs the
tight-prior relative-mode family for the comparison table only. Use
--selection loo to rank every configuration purely by LOO (appropriate when
you only care about local metric distances, not pose shape).

Because LOO is computed from few distances, configurations within
--selection-tolerance-mm of the best LOO RMSE are treated as ties and the
earliest grid entry (the simplest/most trusted) wins.

Typical use with an aggregated VGGT prior:

  python vggt_calibration_pipeline/aggregate_vggt_priors.py DATASET_DIR
  python -m vggt_calibration_pipeline.auto_refine FRAME_DIR [FRAME_DIR ...] \
    --calibration DATASET_DIR/vggt_prior/calibration_prior.toml \
    --prior-stats DATASET_DIR/vggt_prior/prior_stats.json \
    --output-dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from vggt_calibration_pipeline.refine_metric_calibration import build_arg_parser, optimize_refinement


LOOSE = ["--rotation-prior-deg", "3", "--translation-prior-mm", "25"]
RELATIVE = ["--prior-mode", "relative", "--free-scale", "--triangulation", "normalized"]

# (label, selectable, extra refine_metric_calibration args)
CONFIG_GRID: list[tuple[str, bool, list[str]]] = [
    # Shape-validated loose-prior family (best pose shape on frame_000051).
    ("agg_loose_f", True, LOOSE + ["--refine-intrinsics", "--intrinsics-mode", "f", "--sigma-refit-iters", "2", "--drop-outlier-obs-z", "4.0"]),
    ("agg_loose_f_noirls", True, LOOSE + ["--refine-intrinsics", "--intrinsics-mode", "f"]),
    ("agg_loose", True, LOOSE),
    ("agg_loose_irls_drop", True, LOOSE + ["--sigma-refit-iters", "2", "--drop-outlier-obs-z", "4.0"]),
    # Overfit canaries: LOO reliably rejects these when they overfit.
    ("agg_loose_fxfy", True, LOOSE + ["--refine-intrinsics", "--intrinsics-mode", "fxfy", "--sigma-refit-iters", "2", "--drop-outlier-obs-z", "4.0"]),
    ("agg_loose_f_k1", True, LOOSE + ["--refine-intrinsics", "--intrinsics-mode", "f", "--refine-distortion", "--distortion-mode", "k1", "--distortion-prior", "0.3", "--sigma-refit-iters", "2", "--drop-outlier-obs-z", "4.0"]),
    # Tight-prior relative-mode family: best LOO / local metric distances,
    # but inherits VGGT rig-shape bias. Comparison only by default.
    ("relative_fs", False, RELATIVE),
    ("relative_fs_irls_drop", False, RELATIVE + ["--sigma-refit-iters", "2", "--drop-outlier-obs-z", "4.0"]),
    ("relative_fs_f_irls_drop", False, RELATIVE + ["--refine-intrinsics", "--intrinsics-mode", "f", "--sigma-refit-iters", "2", "--drop-outlier-obs-z", "4.0"]),
    ("legacy_default", False, []),
]


BEST_FILES = [
    "calibration_metric.toml",
    "summary.json",
    "points_3d.csv",
    "distance_report.csv",
    "validation_report.csv",
    "reprojection_report.csv",
    "intrinsics_report.csv",
    "loo_report.csv",
]


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refine_parser = build_arg_parser()

    base_argv: list[str] = [str(frame_dir) for frame_dir in args.frame_dirs]
    if args.calibration is not None:
        base_argv += ["--calibration", str(args.calibration)]
    if args.prior_stats is not None:
        base_argv += ["--prior-stats", str(args.prior_stats)]
    if args.origin is not None:
        base_argv += ["--origin", args.origin]

    rows: list[dict[str, Any]] = []
    for label, selectable, extra in CONFIG_GRID:
        config_dir = output_dir / "configs" / label
        argv = base_argv + ["--output-dir", str(config_dir), "--loo"] + extra
        print(f"[{label}] running...", flush=True)
        try:
            summary = optimize_refinement(refine_parser.parse_args(argv))
        except Exception as error:  # noqa: BLE001 - keep the grid running
            print(f"[{label}] FAILED: {error}", flush=True)
            rows.append({"config": label, "selectable": selectable, "error": str(error)})
            continue
        loo = summary.get("loo_mm") or {}
        rows.append(
            {
                "config": label,
                "selectable": selectable,
                "loo_rmse_mm": loo.get("rmse"),
                "loo_mean_abs_mm": loo.get("mean_abs"),
                "loo_max_abs_mm": loo.get("max"),
                "fit_rmse_mm": summary["distance_residual_mm"]["refined_fit"]["rmse"],
                "holdout_rmse_mm": summary["distance_residual_mm"]["refined_holdout"]["rmse"],
                "reproj_rmse_px": summary["reprojection_px"]["refined"]["rmse"],
                "prior_scale": summary.get("prior_scale"),
                "num_dropped_observations": summary.get("num_dropped_observations"),
                "fx_delta_pct_max": summary["intrinsics_delta"]["fx_percent"]["max"],
                "output_dir": summary["output_dir"],
            }
        )
        print(
            f"[{label}] loo_rmse={loo.get('rmse')!r} fit_rmse={rows[-1]['fit_rmse_mm']:.3f} "
            f"reproj={rows[-1]['reproj_rmse_px']:.3f}",
            flush=True,
        )

    scored = [row for row in rows if row.get("loo_rmse_mm") is not None]
    if args.selection == "recommended":
        scored = [row for row in scored if row["selectable"]]
    if not scored:
        raise RuntimeError("No configuration produced a LOO score; check the per-config errors above.")
    best_rmse = min(row["loo_rmse_mm"] for row in scored)
    candidates = [row for row in scored if row["loo_rmse_mm"] <= best_rmse + args.selection_tolerance_mm]
    grid_order = {label: i for i, (label, _, _) in enumerate(CONFIG_GRID)}
    best = min(candidates, key=lambda row: grid_order[row["config"]])

    rows.sort(key=lambda row: row.get("loo_rmse_mm") if row.get("loo_rmse_mm") is not None else float("inf"))
    comparison_path = output_dir / "comparison.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(best["output_dir"])
    for name in BEST_FILES:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, best_dir / name)

    selection = {
        "selected_config": best["config"],
        "selection_mode": args.selection,
        "selection_tolerance_mm": args.selection_tolerance_mm,
        "loo_rmse_mm": best["loo_rmse_mm"],
        "fit_rmse_mm": best["fit_rmse_mm"],
        "reproj_rmse_px": best["reproj_rmse_px"],
        "tied_candidates": [row["config"] for row in candidates],
        "comparison_csv": str(comparison_path),
        "best_dir": str(best_dir),
        "configs": {row["config"]: row.get("loo_rmse_mm") for row in rows},
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame_dirs", type=Path, nargs="+")
    parser.add_argument("--calibration", type=Path, help="Prior/init calibration (e.g. aggregated calibration_prior.toml)")
    parser.add_argument("--prior-stats", type=Path, help="prior_stats.json from aggregate_vggt_priors.py")
    parser.add_argument("--origin", help="Fixed origin camera name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", default="recommended", choices=["recommended", "loo"], help="recommended: select by LOO among the shape-validated family; loo: select by LOO across every configuration")
    parser.add_argument("--selection-tolerance-mm", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    selection = run_grid(parse_args())
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

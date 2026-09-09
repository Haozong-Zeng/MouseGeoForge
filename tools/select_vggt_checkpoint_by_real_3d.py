"""Screen VGGT-Omega checkpoints with fixed-rig real-data triangulation metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATER_ROOT = PROJECT_ROOT / "water-reaching-dataset"
VGGT_RUNNER = WATER_ROOT / "run_vggt_omega_dataset_eval.py"
ANIPOSE_EVAL = WATER_ROOT / "evaluate_vggt_calibration_anipose_triangulation.py"
DEFAULT_RUN_DIR = PROJECT_ROOT / "output/blenderpymouse_vggt_omega_all/run_20260716_base_lr5e6_e50"
DEFAULT_OUTPUT_ROOT = WATER_ROOT / "vggt-omega-checkpoint-screen-real-fixedrig"
DEFAULT_TRIALS = (
    "E_L2_2024-11-18_1_Trial_1",
    "FB_L2_2024-03-16_1_Trial_3",
    "FB_R2_2024-03-15_1_Trial_1",
    "FG_L2_2024-03-05_1_Trial_4",
    "FG_R3_2024-04-04_1_Trial_2",
    "FJ_L2_2024-06-24_1_Trial_7",
    "FS_L2_2024-06-25_1_Trial_2",
    "FU_R2L2_2024-06-27_1_Trial_1",
)


def parse_epoch_spec(spec: str) -> list[int]:
    epochs: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending epoch range: {part}")
            epochs.update(range(start, end + 1))
        else:
            epochs.add(int(part))
    if not epochs:
        raise ValueError("No epochs selected")
    return sorted(epochs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "vggt-omega/vggt_omega_1b_512.pt",
    )
    parser.add_argument("--epochs", default="1-16,20,25,30,40,50")
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        metavar="LABEL=CHECKPOINT",
        help="Add an arbitrary checkpoint candidate. Repeat as needed; use --epochs '' to skip epoch discovery.",
    )
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-top-fraction", type=float, default=0.5)
    parser.add_argument("--catastrophic-ratio", type=float, default=2.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_epoch_checkpoints(run_dir: Path, epochs: list[int]) -> list[tuple[str, Path, int | None]]:
    checkpoint_dir = run_dir / "checkpoints"
    by_epoch: dict[int, Path] = {}
    for path in checkpoint_dir.glob("*.pt"):
        match = re.search(r"_epoch(\d+)_step\d+\.pt$", path.name)
        if match:
            by_epoch[int(match.group(1))] = path.resolve()
    missing = [epoch for epoch in epochs if epoch not in by_epoch]
    if missing:
        raise FileNotFoundError(f"Missing epoch checkpoint(s): {missing}")
    return [(f"epoch{epoch:03d}", by_epoch[epoch], epoch) for epoch in epochs]


def parse_candidate_specs(specs: list[str] | None) -> list[tuple[str, Path, int | None]]:
    candidates = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=CHECKPOINT for --candidate, got {spec!r}")
        label, path_text = spec.split("=", maxsplit=1)
        label = label.strip()
        if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(f"Invalid candidate label: {label!r}")
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        candidates.append((label, path, None))
    return candidates


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(command)
    print(f"$ {command_text}", flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(command_text + "\n\n")
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; see {log_path}")


def evaluate_candidate(
    label: str,
    checkpoint: Path,
    output_root: Path,
    trials: list[str],
    args: argparse.Namespace,
) -> Path:
    candidate_root = output_root / "candidates" / label
    vggt_root = candidate_root / "vggt_eval"
    anipose_root = candidate_root / "anipose_fixedrel"
    final_csv = anipose_root / "all_trials_summary.csv"
    if args.resume and final_csv.exists():
        existing = pd.read_csv(final_csv)
        if len(existing) == len(trials) and set(existing["trial_name"]) == set(trials):
            print(f"[{label}] complete result already exists; reusing it", flush=True)
            return final_csv

    runner_command = [
        sys.executable,
        "-u",
        str(VGGT_RUNNER),
        "--dataset-root",
        str(WATER_ROOT),
        "--output-root",
        str(vggt_root),
        "--checkpoint",
        str(checkpoint),
        "--frame-step",
        str(args.frame_step),
        "--batch-size",
        str(args.batch_size),
        "--min-views",
        "2",
        "--alignment-mode",
        "global",
        "--camera-stabilization-mode",
        "avg-relative-extrinsic",
        "--score-top-fraction",
        str(args.score_top_fraction),
        "--camera-outlier-mode",
        "pairwise",
        "--camera-outlier-z-threshold",
        "3.5",
        "--resume",
    ]
    for trial in trials:
        runner_command.extend(["--trial-name", trial])
    run_logged(runner_command, candidate_root / "vggt_eval.log", args.dry_run)

    anipose_command = [
        sys.executable,
        "-u",
        str(ANIPOSE_EVAL),
        "--source-root",
        str(vggt_root),
        "--output-root",
        str(anipose_root),
        "--calibration-mode",
        "fixedrel",
        "--triangulation-backend",
        "aniposelib",
        "--min-views",
        "2",
        "--alignment-mode",
        "global",
    ]
    for trial in trials:
        anipose_command.extend(["--trial-name", trial])
    run_logged(anipose_command, candidate_root / "anipose_eval.log", args.dry_run)
    return final_csv


def finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def build_ranking(
    result_paths: list[tuple[str, Path, Path, int | None]],
    output_root: Path,
    trials: list[str],
    catastrophic_ratio: float,
) -> pd.DataFrame:
    metric_tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, tuple[Path, int | None]] = {}
    for label, csv_path, checkpoint, epoch in result_paths:
        table = pd.read_csv(csv_path)
        table = table.set_index("trial_name", drop=False)
        metric_tables[label] = table
        metadata[label] = (checkpoint, epoch)

    baseline = metric_tables["baseline"]
    detail_rows: list[dict] = []
    ranking_rows: list[dict] = []
    for label, table in metric_tables.items():
        ratios_3d: list[float] = []
        ratios_reproj: list[float] = []
        failed_count = 0
        catastrophic_count = 0
        improved_both_count = 0
        for trial in trials:
            base_row = baseline.loc[trial] if trial in baseline.index else None
            row = table.loc[trial] if trial in table.index else None
            base_rmse = finite_positive(base_row.get("rmse")) if base_row is not None else None
            base_reproj = finite_positive(base_row.get("triangulation_reproj_rmse_px")) if base_row is not None else None
            rmse = finite_positive(row.get("rmse")) if row is not None else None
            reproj = finite_positive(row.get("triangulation_reproj_rmse_px")) if row is not None else None
            status = str(row.get("status")) if row is not None else "missing"
            valid = status == "ok" and None not in (base_rmse, base_reproj, rmse, reproj)
            ratio_3d = rmse / base_rmse if valid else None
            ratio_reproj = reproj / base_reproj if valid else None
            if not valid:
                failed_count += 1
            else:
                ratios_3d.append(ratio_3d)
                ratios_reproj.append(ratio_reproj)
                if ratio_3d >= catastrophic_ratio or ratio_reproj >= catastrophic_ratio:
                    catastrophic_count += 1
                if ratio_3d < 1.0 and ratio_reproj < 1.0:
                    improved_both_count += 1
            detail_rows.append(
                {
                    "candidate": label,
                    "trial_name": trial,
                    "status": status,
                    "rmse_3d": rmse,
                    "reprojection_rmse_px": reproj,
                    "baseline_rmse_3d": base_rmse,
                    "baseline_reprojection_rmse_px": base_reproj,
                    "rmse_3d_ratio": ratio_3d,
                    "reprojection_ratio": ratio_reproj,
                }
            )

        median_3d = float(np.median(ratios_3d)) if ratios_3d else math.inf
        median_reproj = float(np.median(ratios_reproj)) if ratios_reproj else math.inf
        joint_ratio = math.sqrt(median_3d * median_reproj)
        checkpoint, epoch = metadata[label]
        ranking_rows.append(
            {
                "candidate": label,
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "trial_count": len(trials),
                "valid_trial_count": len(ratios_3d),
                "failed_trial_count": failed_count,
                "catastrophic_trial_count": catastrophic_count,
                "improved_both_trial_count": improved_both_count,
                "median_rmse_3d_ratio": median_3d,
                "median_reprojection_ratio": median_reproj,
                "joint_geomean_ratio": joint_ratio,
                "passes_joint_baseline": bool(
                    failed_count == 0
                    and catastrophic_count == 0
                    and median_3d < 1.0
                    and median_reproj < 1.0
                ),
            }
        )

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(output_root / "per_trial_metrics.csv", index=False)
    ranking = pd.DataFrame(ranking_rows).sort_values(
        [
            "failed_trial_count",
            "catastrophic_trial_count",
            "joint_geomean_ratio",
            "median_rmse_3d_ratio",
            "median_reprojection_ratio",
        ],
        kind="stable",
    )
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking.to_csv(output_root / "checkpoint_ranking.csv", index=False)
    return ranking


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trials = list(args.trial_name or DEFAULT_TRIALS)
    epochs = parse_epoch_spec(args.epochs) if args.epochs.strip() else []
    candidates = [("baseline", args.baseline_checkpoint.resolve(), None)]
    if epochs:
        candidates.extend(find_epoch_checkpoints(run_dir, epochs))
    candidates.extend(parse_candidate_specs(args.candidate))
    labels = [label for label, _, _ in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Candidate labels must be unique: {labels}")

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "epochs": epochs,
        "trials": trials,
        "frame_step_for_rig_estimation": args.frame_step,
        "triangulation_frames": "all frames with valid 2D/3D annotations",
        "camera_stabilization": "pairwise-outlier/top50 then one fixed relative rig per trial",
        "primary_metrics": ["globally_aligned_3d_rmse", "triangulation_reprojection_rmse_px"],
        "ranking_rule": (
            "failed trials, catastrophic trials, then geometric mean of median per-trial "
            "3D-RMSE and reprojection ratios relative to the unfinetuned baseline"
        ),
    }
    (output_root / "screening_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results: list[tuple[str, Path, Path, int | None]] = []
    for index, (label, checkpoint, epoch) in enumerate(candidates, start=1):
        print(f"\n[{index}/{len(candidates)} checkpoints] {label}: {checkpoint}", flush=True)
        csv_path = evaluate_candidate(label, checkpoint, output_root, trials, args)
        if not args.dry_run:
            results.append((label, csv_path, checkpoint, epoch))

    if args.dry_run:
        return
    ranking = build_ranking(results, output_root, trials, args.catastrophic_ratio)
    summary = {
        **config,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "ranking_csv": str(output_root / "checkpoint_ranking.csv"),
        "per_trial_csv": str(output_root / "per_trial_metrics.csv"),
        "best_candidate": ranking.iloc[0].replace({np.nan: None}).to_dict(),
        "retrain_recommended": not bool(
            ranking.loc[ranking["candidate"] != "baseline", "passes_joint_baseline"].any()
        ),
    }
    (output_root / "screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(ranking.head(10).to_string(index=False), flush=True)
    print(json.dumps(summary["best_candidate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

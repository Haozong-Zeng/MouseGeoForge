#!/usr/bin/env python3
"""Run the full VGGT-Omega pipeline over a list of dataset paths.

For each dataset the script runs:
  1. VGGT calibration (one calibration.toml per frame)
  2. Model intermediates (saved by default; use --no-save-intermediates to skip)
  3. Anipose 3D triangulation (one triangulation per frame's calibration)
  4. Reconstruction error comparison against ground-truth pose-3d

Rescue mode (--intermediates-only):
  Skip calibration and anipose entirely. For each dataset path in --paths-file,
  locate the matching output dir under --output-root and run only the intermediates
  step. Frames that already have patch_importance.npy are skipped automatically.

Expected dataset layout (same as temp_data/videos):
    <dataset>/
      videos-raw/         source videos
      pose-2d/            2D keypoint tracks
      pose-3d/            ground-truth 3D CSV (for evaluation)
    <dataset>/../config.toml    Anipose config

The paths file should have one dataset directory per line.
Blank lines and lines starting with # are ignored.

Example paths file (datasets.txt):
    /data/session_2026_01/videos
    /data/session_2026_02/videos
    # /data/session_broken/videos   ← skipped

Usage:
    python scripts/run_batch.py \\
        --paths-file datasets.txt \\
        --output-root /media/drive/vggt_outputs \\
        --checkpoint checkpoints/vggt_omega_1b_512.pt \\
        --frames all \\
        --batch-size 8 \\
        --anipose-env anipose \\
        --anipose-workers 2

    # Rescue: fill in missing intermediates for already-calibrated datasets
    python scripts/run_batch.py \\
        --paths-file datasets.txt \\
        --output-root /media/drive/vggt_outputs \\
        --checkpoint checkpoints/vggt_omega_1b_512.pt \\
        --intermediates-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Path to anipose_eval.py — lives in the analysis folder alongside the output data
ANIPOSE_EVAL_SCRIPT = Path("/media/tony/backup_inte/vggt_paper/analysis /anipose_eval.py")


def read_paths(paths_file: Path) -> list[Path]:
    paths = []
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    if not paths:
        raise ValueError(f"No dataset paths found in {paths_file}")
    return paths


def find_gt_csv(dataset: Path) -> Path | None:
    pose3d_dir = dataset / "pose-3d"
    if not pose3d_dir.exists():
        return None
    csvs = sorted(pose3d_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def output_dir_for(dataset: Path, args: argparse.Namespace) -> Path:
    """Replicate run_pipeline's folder-naming logic to find the output dir."""
    folder_name = args.output_name or dataset.name
    candidate = args.output_root / folder_name
    if candidate.exists():
        src_file = candidate / "source.txt"
        if src_file.exists():
            for line in src_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("dataset:"):
                    existing = line.split(":", 1)[1].strip()
                    if Path(existing) == dataset.resolve():
                        return candidate
        # Different source — find the suffixed dir that was created
        for d in sorted(args.output_root.glob(f"{folder_name}_*")):
            src_file = d / "source.txt"
            if src_file.exists():
                for line in src_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("dataset:") and Path(line.split(":", 1)[1].strip()) == dataset.resolve():
                        return d
    return candidate


def run_calibration(dataset: Path, args: argparse.Namespace) -> tuple[bool, float]:
    cmd = [
        sys.executable, "vggt_calibration_pipeline/run_pipeline.py",
        "--dataset", str(dataset),
        "--output-root", str(args.output_root),
        "--checkpoint", str(args.checkpoint),
        "--frames", args.frames,
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.output_name:
        cmd += ["--output-name", args.output_name]
    if args.save_intermediates:
        cmd += ["--save-intermediates"]
    if args.intermediates_no_grads:
        cmd += ["--intermediates-no-grads"]
    if args.skip_existing:
        cmd += ["--skip-existing"]
    if args.camera_names:
        cmd += ["--camera-names"] + args.camera_names

    t0 = time.time()
    result = subprocess.run(cmd)
    return result.returncode == 0, time.time() - t0


def run_anipose(dataset: Path, output_dir: Path, args: argparse.Namespace) -> tuple[bool, float]:
    config = args.config
    if config is None or not config.exists():
        print(f"  SKIP anipose — config.toml not found at {config}")
        return False, 0.0

    gt_csv = find_gt_csv(dataset)
    staging = output_dir / ".anipose_staging"
    t0 = time.time()

    # 1. Prepare one anipose project per frame calibration into a staging dir
    print("  [anipose] preparing projects...")
    cmd_prepare = [
        sys.executable, str(ANIPOSE_EVAL_SCRIPT), "prepare",
        "--calibrations", str(output_dir),
        "--config", str(config),
        "--source-videos-dir", str(dataset),
        "--output-root", str(staging),
    ]
    if subprocess.run(cmd_prepare).returncode != 0:
        return False, time.time() - t0

    # 2. Triangulate in parallel from staging dir
    print("  [anipose] triangulating...")
    cmd_run = [
        sys.executable, str(ANIPOSE_EVAL_SCRIPT), "run",
        "--output-root", str(staging),
        "--commands", "triangulate",
        "--conda-env", args.anipose_env,
        "--continue-on-error",
        "--workers", str(args.anipose_workers),
    ]
    if args.anipose_cpu_lists:
        cmd_run += ["--cpu-lists"] + args.anipose_cpu_lists
    if subprocess.run(cmd_run).returncode != 0:
        return False, time.time() - t0

    # 3. Compare against ground truth (reads from staging, writes to comparisons/)
    if gt_csv:
        print(f"  [anipose] comparing against {gt_csv.name}...")
        comparison_csv = output_dir / "comparisons" / "anipose_pose3d_comparison.csv"
        cmd_compare = [
            sys.executable, str(ANIPOSE_EVAL_SCRIPT), "compare",
            "--output-root", str(staging),
            "--gt-pose3d", str(gt_csv),
            "--output-csv", str(comparison_csv),
        ]
        if subprocess.run(cmd_compare).returncode != 0:
            return False, time.time() - t0
        print(f"  [anipose] comparison saved → {comparison_csv}")
    else:
        print(f"  [anipose] no pose-3d CSV found in {dataset / 'pose-3d'} — skipping comparison")

    # 4. Move each anipose project into its frame folder, clean up staging
    _move_anipose_into_frames(staging, output_dir)

    return True, time.time() - t0


def _move_anipose_into_frames(staging: Path, output_dir: Path) -> None:
    """Move staging/calibration_XXXXXX/ → frame_XXXXXX/anipose/ for each frame."""
    for project in sorted(staging.iterdir()):
        if not project.is_dir():
            continue
        # calibration_000000 → frame_000000
        num = project.name.replace("calibration_", "")
        frame_dir = output_dir / f"frame_{num}"
        if frame_dir.is_dir():
            dst = frame_dir / "anipose"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(project), str(dst))
    # Remove any leftover files in staging (e.g. anipose_command_results.csv)
    # then the staging dir itself
    shutil.rmtree(staging, ignore_errors=True)


def run_intermediates_only(dataset: Path, output_dir: Path, args: argparse.Namespace) -> tuple[bool, float]:
    """Re-extract frames and run only the intermediates step."""
    cmd = [
        sys.executable, "vggt_calibration_pipeline/run_pipeline.py",
        "--dataset", str(dataset),
        "--output-root", str(output_dir.parent),
        "--output-name", output_dir.name,
        "--checkpoint", str(args.checkpoint),
        "--frames", args.frames,
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--save-intermediates",
        "--skip-existing",  # skip calibration (already done)
    ]
    if args.intermediates_skip_done:
        cmd += ["--intermediates-skip-done"]
    if args.intermediates_no_grads:
        cmd += ["--intermediates-no-grads"]
    if args.camera_names:
        cmd += ["--camera-names"] + args.camera_names
    t0 = time.time()
    result = subprocess.run(cmd)
    return result.returncode == 0, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- paths ---
    parser.add_argument("--paths-file", type=Path, required=True,
                        help="Text file with one dataset path per line.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root directory for all outputs.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Anipose config.toml — shared across all datasets. Not needed with --intermediates-only.")

    # --- calibration ---
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--frames", default="all",
                        help="Frame spec: 0,10,20 or 0:405:5 or all (default: all).")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-name", default=None,
                        help="Override output subfolder name (only useful for single-path runs).")
    parser.add_argument("--save-intermediates", action=argparse.BooleanOptionalAction, default=True,
                        help="Save model intermediates for every frame (default: True).")
    parser.add_argument("--intermediates-no-grads", action="store_true",
                        help="Skip backward pass when saving intermediates.")
    parser.add_argument("--intermediates-only", action="store_true",
                        help="Skip calibration and anipose; only run the intermediates step on existing output dirs.")
    parser.add_argument("--intermediates-skip-done", action="store_true",
                        help="With --intermediates-only: skip frames that already have patch_importance.npy.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames whose calibration.toml already exists.")
    parser.add_argument("--camera-names", nargs="+", default=None,
                        help="Camera names (default: inferred from filenames).")

    # --- anipose ---
    parser.add_argument("--anipose-env", default="anipose",
                        help="Conda environment name for Anipose (default: anipose).")
    parser.add_argument("--anipose-workers", type=int, default=3,
                        help="Parallel triangulation workers (default: 2).")
    parser.add_argument("--anipose-cpu-lists", nargs="+", default=["0-3", "4-7","8-11"],
                        help="CPU affinity lists per worker, e.g. 0-5 6-11.")
    parser.add_argument("--skip-anipose", action="store_true",
                        help="Run calibration only, skip triangulation and comparison.")

    args = parser.parse_args()

    datasets = read_paths(args.paths_file)
    total = len(datasets)
    mode = "intermediates-only" if args.intermediates_only else "full pipeline"
    print(f"Batch run ({mode}): {total} dataset(s) → {args.output_root}\n")

    results: list[tuple[Path, str, float]] = []

    for i, dataset in enumerate(datasets, start=1):
        print(f"[{i}/{total}] {dataset}")

        if args.intermediates_only:
            output_dir = output_dir_for(dataset, args)
            if not output_dir.exists():
                print(f"  SKIP — no existing output dir found for this dataset\n")
                results.append((dataset, "missing", 0.0))
                continue
            print(f"  [intermediates] running on {output_dir} ...")
            ok, t = run_intermediates_only(dataset, output_dir, args)
            if not ok:
                print(f"  FAILED intermediates ({t:.0f}s)\n")
                results.append((dataset, "intermediates_failed", t))
                continue
            print(f"  OK ({t:.0f}s)\n")
            results.append((dataset, "ok", t))
            continue

        if not dataset.exists():
            print(f"  SKIP — path does not exist\n")
            results.append((dataset, "missing", 0.0))
            continue

        # Step 1: calibration
        print("  [calibration] running...")
        ok, t = run_calibration(dataset, args)
        if not ok:
            print(f"  FAILED calibration ({t:.0f}s)\n")
            results.append((dataset, "calib_failed", t))
            continue
        print(f"  [calibration] done ({t:.0f}s)")
        total_t = t

        # Step 2: anipose
        if not args.skip_anipose:
            output_dir = output_dir_for(dataset, args)
            ok, t = run_anipose(dataset, output_dir, args)
            total_t += t
            if not ok:
                print(f"  FAILED anipose ({t:.0f}s)\n")
                results.append((dataset, "anipose_failed", total_t))
                continue
            print(f"  [anipose] done ({t:.0f}s)")

        print(f"  OK ({total_t:.0f}s)\n")
        results.append((dataset, "ok", total_t))

    # Summary
    ok_runs     = [r for r in results if r[1] == "ok"]
    failed_runs = [r for r in results if r[1] != "ok"]
    total_time  = sum(r[2] for r in results)

    print("=" * 60)
    print(f"Completed {len(ok_runs)}/{total} datasets in {total_time:.0f}s")
    if failed_runs:
        print(f"\nFailed ({len(failed_runs)}):")
        for path, reason, _ in failed_runs:
            print(f"  {path}  [{reason}]")
    print("=" * 60)

    return 0 if not failed_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())

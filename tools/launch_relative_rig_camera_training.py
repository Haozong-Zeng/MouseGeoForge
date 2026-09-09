#!/usr/bin/env python3
"""Relaunch camera-head training from a recorded experiment configuration."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-checkpoint",
        type=Path,
        required=True,
        help="Adapter checkpoint whose saved args define datasets and validation splits.",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--center-weight", type=float, default=1.5)
    parser.add_argument("--baseline-weight", type=float, default=2.0)
    parser.add_argument("--synthetic-relative-rig-weight", type=float, default=1.0)
    parser.add_argument("--real-relative-rig-weight", type=float, default=2.0)
    parser.add_argument("--real-geometry-weight", type=float, default=0.0)
    parser.add_argument("--real-geometry-predicted-rig-weight", type=float, default=1.0)
    parser.add_argument("--real-geometry-fixed-rig-weight", type=float, default=1.0)
    parser.add_argument("--real-geometry-pairwise-weight", type=float, default=0.5)
    parser.add_argument("--real-geometry-reprojection-weight", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--temporal-frames", type=int, default=1)
    parser.add_argument("--shared-rig-weight", type=float, default=0.0)
    parser.add_argument("--rig-min-confidence", type=float, default=0.8)
    parser.add_argument("--rig-min-coverage", type=float, default=0.35)
    parser.add_argument("--rig-keep-fraction", type=float, default=1.0)
    parser.add_argument("--rig-max-rotation-error-deg", type=float, default=30.0)
    parser.add_argument("--rig-max-center-error", type=float, default=0.4)
    parser.add_argument("--rig-outlier-rotation-deg", type=float, default=15.0)
    parser.add_argument("--rig-outlier-center", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    repo = Path(__file__).resolve().parents[1] / "vggt-omega"
    sys.path.insert(0, str(repo))

    from training.finetune_blendedmvs import train

    payload = torch.load(cli.template_checkpoint, map_location="cpu", weights_only=False)
    saved_args = dict(payload["args"])
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cli.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    saved_args.update(
        {
            "checkpoint": str(cli.base_checkpoint),
            "adapter_checkpoint": "",
            "output": str(cli.output_dir / "vggt_omega_relative_rig_camera.pt"),
            "checkpoint_dir": str(checkpoint_dir),
            "log_csv": str(cli.output_dir / "train_log.csv"),
            "max_steps": cli.max_steps,
            "epochs": max(int(saved_args.get("epochs", 1)), 1),
            "lr": cli.lr,
            "relative_rig_center_weight": cli.center_weight,
            "relative_rig_baseline_weight": cli.baseline_weight,
            "relative_rig_weight": cli.synthetic_relative_rig_weight,
            "real_geometry_relative_rig_weight": cli.real_relative_rig_weight,
            "real_geometry_weight": cli.real_geometry_weight,
            "real_geometry_predicted_rig_weight": (
                cli.real_geometry_predicted_rig_weight
            ),
            "real_geometry_fixed_rig_weight": cli.real_geometry_fixed_rig_weight,
            "real_geometry_pairwise_weight": cli.real_geometry_pairwise_weight,
            "real_geometry_reprojection_weight": (
                cli.real_geometry_reprojection_weight
            ),
            "val_every": cli.validation_every,
            "checkpoint_every": cli.checkpoint_every,
            "log_every": cli.validation_every,
            "real_geometry_temporal_frames": cli.temporal_frames,
            "real_geometry_shared_rig_weight": cli.shared_rig_weight,
            "real_geometry_rig_min_confidence": cli.rig_min_confidence,
            "real_geometry_rig_min_coverage": cli.rig_min_coverage,
            "real_geometry_rig_keep_fraction": cli.rig_keep_fraction,
            "real_geometry_rig_max_rotation_error_deg": (
                cli.rig_max_rotation_error_deg
            ),
            "real_geometry_rig_max_center_error": cli.rig_max_center_error,
            "real_geometry_shared_rig_outlier_rotation_deg": (
                cli.rig_outlier_rotation_deg
            ),
            "real_geometry_shared_rig_outlier_center": cli.rig_outlier_center,
            "real_geometry_shared_rig_rotation_weight": 1.0,
            "real_geometry_shared_rig_center_weight": 1.0,
            "real_geometry_shared_rig_baseline_weight": 1.0,
            "real_geometry_shared_rig_focal_weight": 0.1,
        }
    )

    train(Namespace(**saved_args))


if __name__ == "__main__":
    main()

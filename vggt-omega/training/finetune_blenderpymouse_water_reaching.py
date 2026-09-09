#!/usr/bin/env python3
"""Convenience entrypoint for BlenderPyMouse water-reaching VGGT-Omega finetuning."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from finetune_blendedmvs import parse_args, train  # noqa: E402


def main() -> None:
    args = parse_args()
    default_dataset = PROJECT_ROOT / "dataset" / "BlenderPyMouse"
    if args.blendedmvs_root == str(PROJECT_ROOT / "dataset" / "blendedmvs_sample"):
        args.blendedmvs_root = str(default_dataset)
    if not args.real_val_roots:
        args.real_val_roots = str(PROJECT_ROOT / "water-reaching-dataset" / "FJ_L2_2024-06-24_1_Trial_7")
    if args.output.endswith("vggt_omega_blendedmvs_light_heads_smoke.pt"):
        args.output = str(PROJECT_ROOT / "output" / "blenderpymouse_vggt_omega_all" / "vggt_omega_blenderpymouse_all_light_heads_latest.pt")
    if args.log_csv.endswith("vggt_omega_blendedmvs_light_heads_smoke_log.csv"):
        args.log_csv = str(PROJECT_ROOT / "output" / "blenderpymouse_vggt_omega_all" / "train_log.csv")
    if not args.checkpoint_dir:
        args.checkpoint_dir = str(PROJECT_ROOT / "output" / "blenderpymouse_vggt_omega_all" / "checkpoints")
    train(args)


if __name__ == "__main__":
    main()

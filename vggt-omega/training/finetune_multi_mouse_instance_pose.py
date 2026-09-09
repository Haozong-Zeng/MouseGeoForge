#!/usr/bin/env python3
"""Phase-0 entrypoint for the long-term multi-mouse instance-pose head."""

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
    args.enable_multi_instance_pose = True
    args.enable_keypoint_head = False
    args.forward_camera_head = False
    args.forward_depth_head = False
    args.camera_weight = 0.0
    args.depth_weight = 0.0
    args.keypoint_head_weight = 0.0
    if args.instance_pose_weight == 0.0:
        args.instance_pose_weight = 1.0
    if args.train_mode == "light_heads":
        args.train_mode = "instance_pose_head"
    if args.output.endswith("vggt_omega_blendedmvs_light_heads_smoke.pt"):
        args.output = str(
            PROJECT_ROOT
            / "output"
            / "multi_mouse_instance_pose"
            / "instance_pose_latest.pt"
        )
    if args.log_csv.endswith("vggt_omega_blendedmvs_light_heads_smoke_log.csv"):
        args.log_csv = str(
            PROJECT_ROOT / "output" / "multi_mouse_instance_pose" / "train_log.csv"
        )
    if not args.checkpoint_dir:
        args.checkpoint_dir = str(
            PROJECT_ROOT / "output" / "multi_mouse_instance_pose" / "checkpoints"
        )
    train(args)


if __name__ == "__main__":
    main()

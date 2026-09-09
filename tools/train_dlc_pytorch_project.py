#!/usr/bin/env python3
"""Train a prepared DeepLabCut PyTorch project."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--trainingset-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from deeplabcut.pose_estimation_pytorch.apis import train_network

    train_network(
        config=args.config.resolve(),
        shuffle=args.shuffle,
        trainingsetindex=args.trainingset_index,
        device=args.device,
        snapshot_path=args.resume.resolve() if args.resume else None,
    )


if __name__ == "__main__":
    main()

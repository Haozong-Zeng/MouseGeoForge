#!/usr/bin/env python3
"""Print selected metadata fields from a VGGT adapter checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Saved training argument to print. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", {})
    selected = args.arg or sorted(saved_args)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "args": {key: saved_args.get(key) for key in selected},
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

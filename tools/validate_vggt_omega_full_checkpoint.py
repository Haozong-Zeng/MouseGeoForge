#!/usr/bin/env python3
"""Strictly load a full VGGT-Omega checkpoint and optionally run inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt-omega"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--images", type=Path, nargs=3)
    parser.add_argument("--image-resolution", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("Checkpoint is not an official raw tensor state dict")

    if args.adapter_checkpoint:
        payload = torch.load(
            args.adapter_checkpoint.resolve(),
            map_location="cpu",
            weights_only=False,
        )
        adapter = payload["model"]
        mismatched = [
            name
            for name, tensor in adapter.items()
            if name not in state or not torch.equal(state[name], tensor)
        ]
        if mismatched:
            raise ValueError(f"Merged checkpoint differs for adapter keys: {mismatched[:10]}")
        print(f"adapter_overlay_ok tensors={len(adapter)}", flush=True)
        del payload, adapter

    model = VGGTOmega()
    result = model.load_state_dict(state, strict=True)
    print(
        f"strict_load_ok tensors={len(state)} "
        f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
        flush=True,
    )
    del state

    if not args.images:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested forward validation")
    paths = [str(path.resolve()) for path in args.images]
    images = load_and_preprocess_images(
        paths,
        image_resolution=args.image_resolution,
    ).cuda()
    model = model.eval().cuda()
    with torch.inference_mode():
        output = model(images)
    print(
        f"forward_ok keys={','.join(sorted(output))} "
        f"depth_shape={tuple(output['depth'].shape)} "
        f"pose_shape={tuple(output['pose_enc'].shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

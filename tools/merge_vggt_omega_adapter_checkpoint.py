#!/usr/bin/env python3
"""Merge a VGGT-Omega adapter payload into an official full state dict."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
        metadata = payload
    else:
        state = payload
        metadata = None
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"Checkpoint does not contain a tensor state dict: {path}")
    return state, metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    base_path = args.base_checkpoint.resolve()
    adapter_path = args.adapter_checkpoint.resolve()
    output_path = args.output.resolve()
    manifest_path = (args.manifest or output_path.with_suffix(output_path.suffix + ".json")).resolve()

    base_state, base_metadata = load_state_dict(base_path)
    if base_metadata is not None:
        raise ValueError(f"Base checkpoint must be an official full state dict: {base_path}")
    adapter_state, adapter_metadata = load_state_dict(adapter_path)
    if adapter_metadata is None or adapter_metadata.get("save_full") is not False:
        raise ValueError(f"Adapter checkpoint is not a light-weight training payload: {adapter_path}")

    missing = sorted(set(adapter_state) - set(base_state))
    if missing:
        raise KeyError(f"Adapter contains {len(missing)} keys absent from base: {missing[:10]}")
    mismatched = [
        name
        for name, tensor in adapter_state.items()
        if tensor.shape != base_state[name].shape or tensor.dtype != base_state[name].dtype
    ]
    if mismatched:
        raise ValueError(f"Adapter/base tensor mismatch for {len(mismatched)} keys: {mismatched[:10]}")

    merged = dict(base_state)
    merged.update(adapter_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)

    manifest = {
        "format": "vggt_omega_official_full_state_dict",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": str(base_path),
        "adapter_checkpoint": str(adapter_path),
        "output_checkpoint": str(output_path),
        "base_tensor_count": len(base_state),
        "adapter_tensor_count": len(adapter_state),
        "output_tensor_count": len(merged),
        "adapter_epoch": adapter_metadata.get("epoch"),
        "adapter_step": adapter_metadata.get("step"),
        "output_size_bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared helpers for compact Core23 temporal sequence datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "mousegeoforge_blenderpymouse_core23_temporal_v1"

CORE23 = (
    "nose",
    "head_center",
    "left_ear",
    "right_ear",
    "neck",
    "spine_center",
    "lumbar_spine",
    "tail_base",
    "tail_q1",
    "tail_q2",
    "tail_q3",
    "tail_q4",
    "tail_q5",
    "tail_tip",
    "left_forepaw_center",
    "right_forepaw_center",
    "left_hindpaw_center",
    "right_hindpaw_center",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "abdomen_center",
)


def map_to_core23(
    source_xyz: np.ndarray,
    source_names: tuple[str, ...] | list[str],
    source_to_core23: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``[T,M,K,3]`` source coordinates without inventing landmarks."""
    source_xyz = np.asarray(source_xyz)
    if source_xyz.ndim != 4 or source_xyz.shape[-1] != 3:
        raise ValueError(f"Expected [T,M,K,3], got {source_xyz.shape}")
    if source_xyz.shape[2] != len(source_names):
        raise ValueError(
            f"Source landmark mismatch: {source_xyz.shape[2]} != {len(source_names)}"
        )
    output = np.zeros((*source_xyz.shape[:2], len(CORE23), 3), dtype=np.float32)
    valid = np.zeros((*source_xyz.shape[:2], len(CORE23)), dtype=bool)
    source_index = {name: index for index, name in enumerate(source_names)}
    target_index = {name: index for index, name in enumerate(CORE23)}
    for source_name, target_name in source_to_core23.items():
        values = source_xyz[:, :, source_index[source_name]]
        finite = np.isfinite(values).all(axis=-1)
        target = target_index[target_name]
        output[:, :, target] = np.nan_to_num(values, nan=0.0).astype(np.float32)
        valid[:, :, target] = finite
    return output, valid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_split(name: str) -> str:
    bucket = hashlib.sha256(name.encode("utf-8")).digest()[0] % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)

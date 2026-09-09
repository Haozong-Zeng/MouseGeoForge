#!/usr/bin/env python3
"""Report numeric differences between corresponding PFM files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        dtype = "<f4" if scale < 0 else ">f4"
        image = np.fromfile(stream, dtype=dtype).reshape(height, width)
    return np.flipud(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)
    left = read_pfm(args.left)
    right = read_pfm(args.right)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    both = (left > 0) & (right > 0)
    differences = np.abs(left[both] - right[both])
    print(
        {
            "shape": list(left.shape),
            "valid_disagreement_pixels": int(
                np.count_nonzero((left > 0) != (right > 0))
            ),
            "valid_union_pixels": int(np.count_nonzero((left > 0) | (right > 0))),
            "median_absolute_mm": float(np.median(differences)),
            "p99_absolute_mm": float(np.percentile(differences, 99)),
            "max_absolute_mm": float(differences.max()),
        }
    )


if __name__ == "__main__":
    main()

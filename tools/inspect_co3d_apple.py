#!/usr/bin/env python3
import gzip
import json
from pathlib import Path


def load_jgz(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main() -> None:
    root = Path("dataset/co3d_apple/apple")
    for name in ["frame_annotations.jgz", "sequence_annotations.jgz"]:
        data = load_jgz(root / name)
        print(f"{name}: {type(data).__name__} len={len(data)}")
        first = data[0] if isinstance(data, list) else next(iter(data.values()))
        print(json.dumps(first, indent=2)[:4000])
    set_lists = json.loads((root / "set_lists.json").read_text())
    print(f"set_lists keys: {list(set_lists.keys())}")
    for key, value in set_lists.items():
        print(key, type(value).__name__, len(value))
        print(value[:3] if isinstance(value, list) else value)


if __name__ == "__main__":
    main()

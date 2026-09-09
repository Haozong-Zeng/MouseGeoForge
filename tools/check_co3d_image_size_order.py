#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("category_root", nargs="+")
    args = parser.parse_args()
    for root_str in args.category_root:
        root = Path(root_str)
        data_root = root.parent
        frames = json.load(gzip.open(root / "frame_annotations.jgz", "rt"))
        print(root)
        for frame in frames[:5]:
            image_path = data_root / frame["image"]["path"]
            print(" annotation", frame["image"]["size"], "PIL", Image.open(image_path).size, image_path)


if __name__ == "__main__":
    main()

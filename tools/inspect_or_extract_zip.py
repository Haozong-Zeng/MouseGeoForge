#!/usr/bin/env python3
import argparse
from collections import defaultdict
import os
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--grep", action="append", default=[])
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--group-top", action="store_true")
    parser.add_argument("--group-sort", choices=["name", "size"], default="name")
    parser.add_argument("--extract-prefix", action="append", default=[])
    parser.add_argument("--extract-to")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as zf:
        names = zf.namelist()
        print(f"zip: {args.zip_path}")
        print(f"members: {len(names)}")
        if args.list:
            for name in names[: args.limit]:
                print(name)
        if args.group_top:
            groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for name in names:
                parts = name.split("/", 1)
                if len(parts) < 2 or not parts[0]:
                    continue
                info = zf.getinfo(name)
                groups[parts[0]][0] += 1
                groups[parts[0]][1] += info.file_size
            if args.group_sort == "size":
                sorted_groups = sorted(groups.items(), key=lambda item: (item[1][1], item[0]))
            else:
                sorted_groups = sorted(groups.items())
            for group, (count, size) in sorted_groups[: args.limit]:
                print(f"{group}\t{count}\t{size / 1024 / 1024:.1f}MB")
        for pattern in args.grep:
            print(f"--- grep: {pattern} ---")
            shown = 0
            for name in names:
                if pattern in name:
                    print(name)
                    shown += 1
                    if shown >= args.limit:
                        break
        if args.extract_to:
            os.makedirs(args.extract_to, exist_ok=True)
            if args.extract_prefix:
                total = 0
                for raw_prefix in args.extract_prefix:
                    prefix = raw_prefix.rstrip("/") + "/"
                    selected = [name for name in names if name.startswith(prefix)]
                    for name in selected:
                        target = Path(args.extract_to) / name
                        target.resolve().relative_to(Path(args.extract_to).resolve())
                        zf.extract(name, args.extract_to)
                    total += len(selected)
                    print(f"extracted_prefix: {raw_prefix}")
                    print(f"member_count: {len(selected)}")
                print(f"total_members: {total}")
            else:
                zf.extractall(args.extract_to)
            print(f"extracted_to: {args.extract_to}")


if __name__ == "__main__":
    main()

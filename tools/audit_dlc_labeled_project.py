#!/usr/bin/env python3
"""Audit DeepLabCut labeled-data folders without modifying the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument(
        "--compare-project",
        type=Path,
        action="append",
        default=[],
        help="Report exact image matches against another DLC project.",
    )
    return parser.parse_args()


def index_to_path(project: Path, value: object) -> Path:
    if isinstance(value, tuple):
        parts = [str(part) for part in value]
    else:
        raw = str(value).replace("\\", "/")
        parts = [part for part in raw.split("/") if part and part != "."]
    if "labeled-data" in parts:
        parts = parts[parts.index("labeled-data") :]
    return project.joinpath(*parts)


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output = (args.output or project / "labeled-data-audit.json").resolve()
    config_path = project / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_bodyparts = list(config.get("bodyparts", []))
    configured_paths = {
        Path(str(path).replace("\\", "/")).stem: str(path)
        for path in config.get("video_sets", {})
    }
    configured_stems = set(configured_paths)

    folder_records: list[dict] = []
    image_hashes: dict[str, list[str]] = {}
    all_bodypart_signatures: Counter[tuple[str, ...]] = Counter()

    for folder in sorted((project / "labeled-data").iterdir()):
        if not folder.is_dir():
            continue
        h5_files = sorted(folder.glob("CollectedData_*.h5"))
        csv_files = sorted(folder.glob("CollectedData_*.csv"))
        image_files = sorted(
            path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        record = {
            "folder": folder.name,
            "configured": folder.name in configured_stems,
            "configured_video_path": configured_paths.get(folder.name),
            "h5_files": [path.name for path in h5_files],
            "csv_files": [path.name for path in csv_files],
            "image_file_count": len(image_files),
            "rows": 0,
            "columns": 0,
            "scorers": [],
            "bodyparts": [],
            "coordinate_levels": [],
            "missing_index_images": [],
            "unindexed_images": [],
            "image_sizes": {},
            "read_error": None,
        }

        referenced: set[Path] = set()
        if h5_files:
            try:
                table = pd.read_hdf(h5_files[0])
                record["rows"] = len(table)
                record["columns"] = len(table.columns)
                if isinstance(table.columns, pd.MultiIndex):
                    names = list(table.columns.names)
                    if "scorer" in names:
                        record["scorers"] = sorted(
                            {str(value) for value in table.columns.get_level_values("scorer")}
                        )
                    if "bodyparts" in names:
                        record["bodyparts"] = list(
                            dict.fromkeys(
                                str(value)
                                for value in table.columns.get_level_values("bodyparts")
                            )
                        )
                    elif table.columns.nlevels >= 2:
                        record["bodyparts"] = list(
                            dict.fromkeys(
                                str(value) for value in table.columns.get_level_values(-2)
                            )
                        )
                    record["coordinate_levels"] = sorted(
                        {str(value) for value in table.columns.get_level_values(-1)}
                    )
                signature = tuple(record["bodyparts"])
                all_bodypart_signatures[signature] += 1
                for index in table.index:
                    referenced.add(index_to_path(project, index).resolve())
                record["missing_index_images"] = [
                    str(path.relative_to(project))
                    for path in sorted(referenced)
                    if not path.is_file()
                ]
            except Exception as exc:  # noqa: BLE001
                record["read_error"] = f"{type(exc).__name__}: {exc}"

        local_images = {path.resolve() for path in image_files}
        record["unindexed_images"] = [
            str(path.relative_to(project)) for path in sorted(local_images - referenced)
        ]
        size_counts: Counter[str] = Counter()
        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    size_counts[f"{image.width}x{image.height}"] += 1
            except Exception:  # noqa: BLE001
                size_counts["unreadable"] += 1
            if args.hash_images:
                digest = image_digest(image_path)
                image_hashes.setdefault(digest, []).append(
                    str(image_path.relative_to(project))
                )
        record["image_sizes"] = dict(sorted(size_counts.items()))
        folder_records.append(record)

    duplicate_hash_groups = [
        paths for paths in image_hashes.values() if len(paths) > 1
    ]
    cross_project_matches: dict[str, list[dict]] = {}
    if args.compare_project:
        primary_hashes = set(image_hashes)
        if not primary_hashes:
            raise ValueError("--compare-project requires --hash-images")
        for compare_project_arg in args.compare_project:
            compare_project = compare_project_arg.resolve()
            matches: list[dict] = []
            for image_path in sorted((compare_project / "labeled-data").glob("*/*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                digest = image_digest(image_path)
                if digest in primary_hashes:
                    matches.append(
                        {
                            "sha256": digest,
                            "primary": image_hashes[digest],
                            "comparison": str(image_path.relative_to(compare_project)),
                        }
                    )
            cross_project_matches[str(compare_project)] = matches
    report = {
        "project": str(project),
        "config": str(config_path),
        "expected_bodyparts": expected_bodyparts,
        "configured_video_stems": len(configured_stems),
        "summary": {
            "folders": len(folder_records),
            "configured_folders": sum(item["configured"] for item in folder_records),
            "folders_with_h5": sum(bool(item["h5_files"]) for item in folder_records),
            "folders_with_csv": sum(bool(item["csv_files"]) for item in folder_records),
            "folders_with_read_errors": sum(bool(item["read_error"]) for item in folder_records),
            "total_label_rows": sum(item["rows"] for item in folder_records),
            "total_images": sum(item["image_file_count"] for item in folder_records),
            "missing_index_images": sum(
                len(item["missing_index_images"]) for item in folder_records
            ),
            "unindexed_images": sum(
                len(item["unindexed_images"]) for item in folder_records
            ),
            "bodypart_signature_count": len(all_bodypart_signatures),
            "folders_matching_config_bodyparts": sum(
                item["bodyparts"] == expected_bodyparts for item in folder_records
            ),
            "duplicate_image_hash_groups": len(duplicate_hash_groups),
            "cross_project_exact_matches": {
                project_path: len(matches)
                for project_path, matches in cross_project_matches.items()
            },
        },
        "bodypart_signatures": [
            {"bodyparts": list(signature), "folder_count": count}
            for signature, count in all_bodypart_signatures.most_common()
        ],
        "duplicate_image_hash_groups": duplicate_hash_groups,
        "cross_project_exact_matches": cross_project_matches,
        "folders": folder_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()

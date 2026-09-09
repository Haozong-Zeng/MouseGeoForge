#!/usr/bin/env python3
"""Safely merge independently downloaded DeepLabCut ZIP archives."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="wr_cam*.zip")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def safe_destination(output_dir: Path, info: ZipInfo) -> Path:
    archive_path = PurePosixPath(info.filename)
    if archive_path.is_absolute() or ".." in archive_path.parts:
        raise ValueError(f"Unsafe archive path: {info.filename!r}")
    destination = output_dir.joinpath(*archive_path.parts).resolve()
    if not destination.is_relative_to(output_dir):
        raise ValueError(f"Archive path escapes output directory: {info.filename!r}")
    return destination


def file_crc32(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def existing_file_matches(path: Path, info: ZipInfo) -> bool:
    return path.stat().st_size == info.file_size and file_crc32(path) == info.CRC


def extract_member(archive: ZipFile, info: ZipInfo, destination: Path) -> str:
    if info.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        return "directories"
    if destination.exists():
        if not destination.is_file() or not existing_file_matches(destination, info):
            raise FileExistsError(
                f"Existing file conflicts with archive member: {destination}"
            )
        return "verified_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        temporary.unlink()
    try:
        with archive.open(info, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise OSError(f"Size mismatch after extraction: {info.filename}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "extracted"


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(archive_dir.glob(args.glob))
    if not archives:
        raise FileNotFoundError(
            f"No archives matched {args.glob!r} under {archive_dir}"
        )

    started = time.time()
    counts = {
        "archives": len(archives),
        "members": 0,
        "directories": 0,
        "extracted": 0,
        "verified_existing": 0,
        "uncompressed_bytes": 0,
    }
    archive_rows = []
    for archive_path in archives:
        row = {
            "archive": str(archive_path),
            "members": 0,
            "extracted": 0,
            "verified_existing": 0,
            "uncompressed_bytes": 0,
        }
        print(f"extracting {archive_path.name}", flush=True)
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                destination = safe_destination(output_dir, info)
                status = extract_member(archive, info, destination)
                counts["members"] += 1
                counts[status] += 1
                counts["uncompressed_bytes"] += info.file_size
                row["members"] += 1
                row["uncompressed_bytes"] += info.file_size
                if status in ("extracted", "verified_existing"):
                    row[status] += 1
        archive_rows.append(row)

    report = {
        "archive_dir": str(archive_dir),
        "output_dir": str(output_dir),
        "glob": args.glob,
        "elapsed_seconds": round(time.time() - started, 3),
        "counts": counts,
        "archives": archive_rows,
    }
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_dir / "dlc_archive_merge_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()

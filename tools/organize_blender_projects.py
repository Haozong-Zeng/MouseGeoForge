#!/usr/bin/env python3
"""Separate BlenderPyMouse production projects from recoverable Blender archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ARCHIVE_NAME = "BlenderProjects_Archive_20260901"


def resolved_under(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_reference(value: str) -> str | None:
    portable = value.replace("\\", "/")
    marker = "/home/haozong/projects/MouseGeoForge/"
    index = portable.find(marker)
    if index < 0:
        return None
    return portable[index:]


def replacement_for(old_value: str, new_path: Path) -> str:
    if "\\" in old_value:
        return "\\\\wsl.localhost\\Ubuntu" + str(new_path).replace("/", "\\")
    return str(new_path)


def replace_json_strings(value, replacements: dict[str, Path]) -> tuple[object, int]:
    if isinstance(value, dict):
        result = {}
        changes = 0
        for key, child in value.items():
            result[key], child_changes = replace_json_strings(child, replacements)
            changes += child_changes
        return result, changes
    if isinstance(value, list):
        result = []
        changes = 0
        for child in value:
            updated, child_changes = replace_json_strings(child, replacements)
            result.append(updated)
            changes += child_changes
        return result, changes
    if isinstance(value, str):
        normalized = normalized_reference(value)
        if normalized in replacements:
            return replacement_for(value, replacements[normalized]), 1
    return value, 0


def atomic_json_write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def move_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.rename(destination)
    except OSError:
        shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--archive-name", default=DEFAULT_ARCHIVE_NAME)
    parser.add_argument(
        "--defer-production",
        action="append",
        default=[],
        help="Production source path relative to project root that must remain in place.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    configuration_path = root / "configs" / "blenderpymouse_production_projects.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    production_root = resolved_under(root, root / configuration["destination_root"])
    archive_root = resolved_under(root, root / "backup" / args.archive_name)
    archive_files_root = archive_root / "files"
    existing_manifest_path = archive_root / "archive_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.is_file()
        else {}
    )
    deferred = {
        resolved_under(root, root / relative) for relative in args.defer_production
    }

    production_moves: list[tuple[Path, Path, list[str]]] = []
    support_moves: list[tuple[Path, Path]] = []
    replacements: dict[str, Path] = {}
    production_sources: set[Path] = set()
    deferred_backups: set[Path] = set()

    for project in configuration["projects"]:
        source = resolved_under(root, root / project["source"])
        destination = resolved_under(production_root, production_root / project["destination"])
        production_sources.add(source)
        if source in deferred:
            deferred_backups.add(source.with_suffix(source.suffix + "1"))
            continue
        if source.exists() and destination.exists():
            raise FileExistsError(f"Both production paths exist: {source} and {destination}")
        if not source.exists() and not destination.exists():
            raise FileNotFoundError(f"Production project missing: {source}")
        if source.exists():
            production_moves.append((source, destination, list(project["datasets"])))
        replacements[str(source)] = destination

    for support in configuration["support_files"]:
        source = resolved_under(root, root / support["source"])
        destination = resolved_under(production_root, production_root / support["destination"])
        if source.exists() and destination.exists():
            raise FileExistsError(f"Both support paths exist: {source} and {destination}")
        if not source.exists() and not destination.exists():
            raise FileNotFoundError(f"Production support file missing: {source}")
        if source.exists():
            support_moves.append((source, destination))
        replacements[str(source)] = destination

    excluded_roots = {production_root, root / "backup"}
    archive_candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".blend", ".blend1"}:
            continue
        resolved = path.resolve()
        if any(base == resolved or base in resolved.parents for base in excluded_roots):
            continue
        if resolved in deferred or resolved in deferred_backups:
            continue
        if resolved in production_sources and any(resolved == row[0] for row in production_moves):
            continue
        destination = resolved_under(
            archive_files_root,
            archive_files_root / resolved.relative_to(root),
        )
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        archive_candidates.append((resolved, destination))

    summary = {
        "mode": "execute" if args.execute else "dry-run",
        "production_move_count": len(production_moves),
        "support_move_count": len(support_moves),
        "deferred_production_count": len(deferred),
        "archive_file_count": len(archive_candidates),
        "production_bytes": sum(source.stat().st_size for source, _, _ in production_moves),
        "archive_bytes": sum(source.stat().st_size for source, _ in archive_candidates),
        "production_root": str(production_root),
        "archive_root": str(archive_root),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return

    production_rows = []
    for index, (source, destination, datasets) in enumerate(production_moves, start=1):
        size = source.stat().st_size
        move_file(source, destination)
        production_rows.append(
            {
                "old_path": str(source.relative_to(root)),
                "path": str(destination.relative_to(root)),
                "size": size,
                "datasets": datasets,
            }
        )
        print(f"production_move {index}/{len(production_moves)} {destination}", flush=True)

    for source, destination in support_moves:
        move_file(source, destination)

    archive_rows = []
    for index, (source, destination) in enumerate(archive_candidates, start=1):
        size = source.stat().st_size
        mtime_ns = source.stat().st_mtime_ns
        move_file(source, destination)
        os.utime(destination, ns=(destination.stat().st_atime_ns, mtime_ns))
        archive_rows.append(
            {
                "original_path": str(source.relative_to(root)),
                "archive_path": str(destination.relative_to(root)),
                "size": size,
            }
        )
        if index == 1 or index == len(archive_candidates) or index % 25 == 0:
            print(f"archive_move {index}/{len(archive_candidates)}", flush=True)

    changed_json_files = 0
    changed_json_values = 0
    dataset_root = root / "dataset" / "BlenderPyMouse"
    for path in dataset_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        updated, changes = replace_json_strings(payload, replacements)
        if changes:
            atomic_json_write(path, updated)
            changed_json_files += 1
            changed_json_values += changes

    if not args.skip_hashes:
        hash_total = len(production_rows) + len(archive_rows)
        hash_index = 0
        for row in production_rows:
            row["sha256"] = sha256(root / row["path"])
            hash_index += 1
            print(f"hash_progress {hash_index}/{hash_total}", flush=True)
        for row in archive_rows:
            row["sha256"] = sha256(root / row["archive_path"])
            hash_index += 1
            if hash_index == hash_total or hash_index % 10 == 0:
                print(f"hash_progress {hash_index}/{hash_total}", flush=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    all_production_rows = [
        *existing_manifest.get("production_projects", []),
        *production_rows,
    ]
    all_archive_rows = [
        *existing_manifest.get("archived_files", []),
        *archive_rows,
    ]
    manifest = {
        "format": "mousegeoforge_blender_project_archive_v1",
        "generated_at_utc": generated_at,
        **summary,
        "production_move_count": len(all_production_rows),
        "archive_file_count": len(all_archive_rows),
        "production_bytes": sum(row["size"] for row in all_production_rows),
        "archive_bytes": sum(row["size"] for row in all_archive_rows),
        "dataset_json_files_updated": changed_json_files,
        "dataset_json_values_updated": changed_json_values,
        "deferred_production": [str(path.relative_to(root)) for path in sorted(deferred)],
        "production_projects": all_production_rows,
        "archived_files": all_archive_rows,
    }
    archive_root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(archive_root / "archive_manifest.json", manifest)
    production_root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(production_root / "production_manifest.json", manifest)
    (archive_root / "README.md").write_text(
        "# Blender project archive\n\n"
        "Files under `files/` retain their original MouseGeoForge-relative paths. "
        "Use `archive_manifest.json` for sizes, SHA-256 checksums, and restoration paths.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset_json_files_updated": changed_json_files,
                "dataset_json_values_updated": changed_json_values,
                "manifest": str(archive_root / "archive_manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

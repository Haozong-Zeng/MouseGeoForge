#!/usr/bin/env python3
"""Print external file dependencies of the Blender project loaded by Blender."""

from __future__ import annotations

import json
from pathlib import Path

import bpy


def record(kind: str, name: str, filepath: str, packed: bool = False) -> dict:
    resolved = Path(bpy.path.abspath(filepath)).resolve() if filepath else None
    return {
        "kind": kind,
        "name": name,
        "filepath": filepath,
        "resolved": str(resolved) if resolved is not None else None,
        "exists": bool(resolved and resolved.exists()),
        "packed": packed,
    }


dependencies: list[dict] = []
for image in bpy.data.images:
    if image.source == "FILE" and image.filepath:
        dependencies.append(
            record("image", image.name, image.filepath, packed=bool(image.packed_file))
        )
for library in bpy.data.libraries:
    dependencies.append(record("library", library.name, library.filepath))
for collection_name, collection in (
    ("movieclip", bpy.data.movieclips),
    ("sound", bpy.data.sounds),
    ("font", bpy.data.fonts),
    ("cache_file", bpy.data.cache_files),
    ("volume", bpy.data.volumes),
):
    for block in collection:
        filepath = getattr(block, "filepath", "")
        if filepath and filepath != "<builtin>":
            packed_file = getattr(block, "packed_file", None)
            dependencies.append(
                record(collection_name, block.name, filepath, packed=bool(packed_file))
            )

payload = {
    "blend": str(Path(bpy.data.filepath).resolve()),
    "dependency_count": len(dependencies),
    "unpacked_dependency_count": sum(not item["packed"] for item in dependencies),
    "missing_unpacked_count": sum(
        not item["packed"] and not item["exists"] for item in dependencies
    ),
    "dependencies": dependencies,
}
print("BLEND_DEPENDENCY_AUDIT=" + json.dumps(payload, ensure_ascii=False), flush=True)

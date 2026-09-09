#!/usr/bin/env python3
"""Audit the render-critical objects in the dual-mouse Homecage scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


CAMERAS = (
    "HomecageCam_Top",
    "HomecageCam_SideNegX",
    "HomecageCam_SideNegY",
    "HomecageCam_SidePosY",
)
INSTANCES = {
    "black_mouse": ("LP_Black_Armature", "LP_Black_Mesh"),
    "brown_mouse": ("LP_Brown_Armature", "LP_Brown_Mesh"),
}
SOURCE_SHELL = "MGF_MARS_HomecageShell"
PROXIES = {
    "HomecageCam_SideNegX": ("MGF_DepthProxy_SideNegX", "MGF_DepthShellProxy_SideNegX", 0),
    "HomecageCam_SideNegY": ("MGF_DepthProxy_SideNegY", "MGF_DepthShellProxy_SideNegY", 12),
    "HomecageCam_SidePosY": ("MGF_DepthProxy_SidePosY", "MGF_DepthShellProxy_SidePosY", 1),
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def object_record(obj: bpy.types.Object | None) -> dict:
    if obj is None:
        return {"exists": False}
    return {
        "exists": True,
        "name": obj.name,
        "type": obj.type,
        "hide_render": bool(obj.hide_render),
        "hide_viewport": bool(obj.hide_viewport),
        "parent": obj.parent.name if obj.parent else None,
        "collections": [collection.name for collection in obj.users_collection],
        "vertex_count": len(obj.data.vertices) if obj.type == "MESH" else None,
        "polygon_count": len(obj.data.polygons) if obj.type == "MESH" else None,
    }


def main() -> None:
    args = args_after_dash()
    if len(args) != 1:
        raise SystemExit("Usage: blender -b scene.blend -P audit_homecage_dual_dataset_scene.py -- report.json")
    scene = bpy.context.scene
    payload = {
        "blend": str(Path(bpy.data.filepath).resolve()),
        "frame_range": [scene.frame_start, scene.frame_end],
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "scene_properties": {
            key: scene.get(key)
            for key in (
                "MGF_side_depth_wall_policy",
                "MGF_side_depth_proxy_version",
                "MGF_source_frame_start",
                "MGF_source_stride",
                "MGF_sampled_fps",
            )
        },
        "cameras": {name: object_record(bpy.data.objects.get(name)) for name in CAMERAS},
        "instances": {
            label: {
                "armature": object_record(bpy.data.objects.get(names[0])),
                "mesh": object_record(bpy.data.objects.get(names[1])),
                "related_render_meshes": [
                    obj.name
                    for obj in bpy.data.objects
                    if obj.type == "MESH"
                    and obj.name.startswith("LP_Black_" if label == "black_mouse" else "LP_Brown_")
                    and not obj.hide_render
                ],
            }
            for label, names in INSTANCES.items()
        },
        "source_shell": object_record(bpy.data.objects.get(SOURCE_SHELL)),
        "depth_proxies": {},
        "capture_view_layer": "MGF_Capture_Clean" if scene.view_layers.get("MGF_Capture_Clean") else scene.view_layers[0].name,
        "hair_curve_count": sum(obj.type == "CURVES" for obj in bpy.data.objects),
    }
    for camera, (collection_name, object_name, removed_face) in PROXIES.items():
        collection = bpy.data.collections.get(collection_name)
        proxy = bpy.data.objects.get(object_name)
        payload["depth_proxies"][camera] = {
            "collection": collection_name,
            "collection_exists": collection is not None,
            "collection_hide_render": bool(collection.hide_render) if collection else None,
            "object": object_record(proxy),
            "expected_removed_face": removed_face,
            "stored_removed_face": proxy.get("mgf_removed_source_face_index") if proxy else None,
            "stored_camera": proxy.get("mgf_camera") if proxy else None,
            "stored_policy": proxy.get("mgf_policy") if proxy else None,
        }

    errors = []
    for name, record in payload["cameras"].items():
        if not record["exists"] or record.get("type") != "CAMERA":
            errors.append(f"missing camera {name}")
    for label, record in payload["instances"].items():
        if not record["armature"]["exists"] or record["armature"].get("type") != "ARMATURE":
            errors.append(f"missing armature for {label}")
        if not record["mesh"]["exists"] or record["mesh"].get("type") != "MESH":
            errors.append(f"missing mesh for {label}")
    if not payload["source_shell"]["exists"]:
        errors.append(f"missing source shell {SOURCE_SHELL}")
    for camera, record in payload["depth_proxies"].items():
        if not record["collection_exists"] or not record["object"]["exists"]:
            errors.append(f"missing selective depth proxy for {camera}")
        if record["stored_removed_face"] != record["expected_removed_face"]:
            errors.append(f"wrong removed face for {camera}")
    payload["errors"] = errors
    payload["ready"] = not errors
    output = Path(args[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ready": payload["ready"], "errors": errors}, indent=2), flush=True)
    if errors:
        raise RuntimeError("Dual Homecage dataset scene audit failed")


if __name__ == "__main__":
    main()

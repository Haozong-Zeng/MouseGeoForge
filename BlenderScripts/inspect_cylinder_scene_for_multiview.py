#!/usr/bin/env python3
"""Print a compact JSON audit for cylinder mouse scenes."""

from __future__ import annotations

import json

import bpy


def rounded(values, digits=6):
    return [round(float(value), digits) for value in values]


def material_summary(material):
    nodes = []
    if material and material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type in {"BSDF_PRINCIPLED", "BSDF_GLASS", "OUTPUT_MATERIAL", "MIX_SHADER"}:
                nodes.append({"name": node.name, "type": node.type})
    return {
        "name": material.name if material else None,
        "use_nodes": bool(material and material.use_nodes),
        "surface_render_method": getattr(material, "surface_render_method", None) if material else None,
        "blend_method": getattr(material, "blend_method", None) if material else None,
        "nodes": nodes,
    }


def object_record(obj):
    record = {
        "name": obj.name,
        "type": obj.type,
        "collections": [collection.name for collection in obj.users_collection],
        "location": rounded(obj.location),
        "rotation_euler": rounded(obj.rotation_euler),
        "scale": rounded(obj.scale),
        "parent": obj.parent.name if obj.parent else None,
        "hide_render": bool(obj.hide_render),
    }
    if obj.type in {"MESH", "CURVE"}:
        record["dimensions"] = rounded(obj.dimensions)
        record["materials"] = [material_summary(slot.material) for slot in obj.material_slots]
    if obj.type == "CAMERA":
        record["camera"] = {
            "lens": float(obj.data.lens),
            "sensor_width": float(obj.data.sensor_width),
            "sensor_height": float(obj.data.sensor_height),
            "sensor_fit": obj.data.sensor_fit,
            "clip_start": float(obj.data.clip_start),
            "clip_end": float(obj.data.clip_end),
            "matrix_world": [rounded(row) for row in obj.matrix_world],
        }
    return record


def interesting(obj):
    name = obj.name.lower()
    environment = {
        "building_floor",
        "part_cylinder",
        "part_glasscylinder160mm",
        "part_whitehexagonbackground",
    }
    return obj.type == "CAMERA" or name in environment or any(
        token in name
        for token in ("mouse", "armature", "deformation", "controller")
    )


scene = bpy.context.scene
compositor_tree = getattr(scene, "node_tree", None) or getattr(scene, "compositing_node_group", None)
payload = {
    "blend": bpy.data.filepath,
    "scene": scene.name,
    "frame_range": [scene.frame_start, scene.frame_end],
    "render_engine": scene.render.engine,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
    "active_camera": scene.camera.name if scene.camera else None,
    "object_count": len(bpy.data.objects),
    "camera_count": sum(obj.type == "CAMERA" for obj in bpy.data.objects),
    "lights": [
        {
            "name": obj.name,
            "type": obj.data.type,
            "energy": float(obj.data.energy),
            "hide_render": bool(obj.hide_render),
            "collections": [collection.name for collection in obj.users_collection],
        }
        for obj in bpy.data.objects
        if obj.type == "LIGHT"
    ],
    "collection_render_visibility": {
        collection.name: not collection.hide_render
        for collection in bpy.data.collections
        if collection.name in {"Collection", "Mouse_White", "Mouse_Brown", "moseq", "MGF_8View_Cameras"}
    },
    "world": {
        "name": scene.world.name if scene.world else None,
        "color": rounded(scene.world.color) if scene.world else None,
        "use_nodes": bool(scene.world and scene.world.use_nodes),
        "nodes": [node.type for node in scene.world.node_tree.nodes]
        if scene.world and scene.world.use_nodes and scene.world.node_tree
        else [],
    },
    "compositor": {
        "use_nodes": compositor_tree is not None,
        "use_compositing": bool(getattr(scene.render, "use_compositing", True)),
        "nodes": [
            {"name": node.name, "type": node.type, "mute": bool(node.mute)}
            for node in compositor_tree.nodes
        ]
        if compositor_tree is not None
        else [],
    },
    "objects": [object_record(obj) for obj in bpy.data.objects if interesting(obj)],
}
print("MGF_SCENE_AUDIT_BEGIN")
print(json.dumps(payload, ensure_ascii=False, indent=2))
print("MGF_SCENE_AUDIT_END")

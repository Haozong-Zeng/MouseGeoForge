#!/usr/bin/env python3
"""Dump appearance-related Blender data blocks for project comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def socket_value(socket):
    value = getattr(socket, "default_value", None)
    if hasattr(value, "to_list"):
        return value.to_list()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return list(value)
    except TypeError:
        return repr(value)


def node_dump(node):
    payload = {
        "name": node.name,
        "label": node.label,
        "type": node.bl_idname,
        "inputs": {
            socket.name: socket_value(socket)
            for socket in node.inputs
            if not socket.is_linked
        },
    }
    if node.type == "VALTORGB":
        payload["color_ramp"] = {
            "interpolation": node.color_ramp.interpolation,
            "elements": [
                {"position": element.position, "color": list(element.color)}
                for element in node.color_ramp.elements
            ],
        }
    return payload


def material_dump(material):
    return {
        "name": material.name,
        "surface_render_method": getattr(material, "surface_render_method", None),
        "use_nodes": material.use_nodes,
        "nodes": [node_dump(node) for node in material.node_tree.nodes]
        if material.use_nodes and material.node_tree
        else [],
        "links": [
            {
                "from_node": link.from_node.name,
                "from_socket": link.from_socket.name,
                "to_node": link.to_node.name,
                "to_socket": link.to_socket.name,
            }
            for link in material.node_tree.links
        ]
        if material.use_nodes and material.node_tree
        else [],
    }


def main() -> None:
    args = args_after_dash()
    if len(args) != 1:
        raise SystemExit("Usage: blender -b file.blend -P inspect_blend_appearance.py -- out.json")
    scene = bpy.context.scene
    view = scene.view_settings
    payload = {
        "blend": bpy.data.filepath,
        "scene": {
            "engine": scene.render.engine,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "film_transparent": scene.render.film_transparent,
            "cycles": {
                "device": scene.cycles.device,
                "samples": scene.cycles.samples,
                "use_denoising": scene.cycles.use_denoising,
                "preview_samples": scene.cycles.preview_samples,
                "transparent_max_bounces": scene.cycles.transparent_max_bounces,
                "transmission_bounces": scene.cycles.transmission_bounces,
            },
            "view_transform": view.look,
            "view_settings": {
                "look": view.look,
                "exposure": view.exposure,
                "gamma": view.gamma,
                "view_transform": view.view_transform,
            },
        },
        "world": {
            "color": list(scene.world.color) if scene.world else None,
            "nodes": [node_dump(node) for node in scene.world.node_tree.nodes]
            if scene.world and scene.world.use_nodes
            else [],
        },
        "lights": [
            {
                "object": obj.name,
                "data": obj.data.name,
                "type": obj.data.type,
                "energy": obj.data.energy,
                "color": list(obj.data.color),
                "size": getattr(obj.data, "size", None),
                "location": list(obj.location),
                "rotation_euler": list(obj.rotation_euler),
                "hide_render": obj.hide_render,
            }
            for obj in bpy.data.objects
            if obj.type == "LIGHT"
        ],
        "curve_objects": [
            {
                "name": obj.name,
                "data": obj.data.name,
                "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
                "hide_render": obj.hide_render,
                "parent": obj.parent.name if obj.parent else None,
                "surface": getattr(getattr(obj, "data", None), "surface", None).name
                if getattr(getattr(obj, "data", None), "surface", None)
                else None,
                "curve_count": len(obj.data.curves) if obj.type == "CURVES" else None,
                "point_count": len(obj.data.points) if obj.type == "CURVES" else None,
                "modifiers": [
                    {
                        "name": modifier.name,
                        "type": modifier.type,
                        "node_group": modifier.node_group.name
                        if modifier.type == "NODES" and modifier.node_group
                        else None,
                    }
                    for modifier in obj.modifiers
                ],
            }
            for obj in bpy.data.objects
            if obj.type in {"CURVES", "CURVE"}
        ],
        "selected_objects": [],
        "materials": [material_dump(material) for material in bpy.data.materials],
        "mouse_meshes": [
            {
                "name": obj.name,
                "vertex_count": len(obj.data.vertices),
                "edge_count": len(obj.data.edges),
                "polygon_count": len(obj.data.polygons),
                "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            }
            for obj in bpy.data.objects
            if obj.type == "MESH" and obj.name.startswith("MS_Mouse")
        ],
    }
    keywords = ("glass", "cylinder", "metal", "steel", "alum", "fur", "hair", "mouse")
    for obj in bpy.data.objects:
        material_names = [slot.material.name if slot.material else None for slot in obj.material_slots]
        haystack = " ".join([obj.name, *(name or "" for name in material_names)]).lower()
        if any(keyword in haystack for keyword in keywords):
            payload["selected_objects"].append(
                {
                    "name": obj.name,
                    "type": obj.type,
                    "materials": material_names,
                    "hide_render": obj.hide_render,
                    "collections": [collection.name for collection in obj.users_collection],
                }
            )
    Path(args[0]).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare the sampled Keypoint-MoSeq scene for brown-fur Core-23 rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


ACTIVE_MOUSE = "MS_Mouse.001"
ACTIVE_COLLECTION = "Mouse_White"
INACTIVE_COLLECTION = "Mouse_Brown"
FUR_SOURCE_MATERIAL = "Black_Whiskers"
FUR_MATERIAL = "MGF_BrownFur_KeypointMoseq"
DEPTH_IGNORE_COLLECTION = "MGF_DepthIgnoreTransparentGlass"
TRANSPARENT_OBJECTS = (
    "Part_GlassCylinder160mm",
    "Part_AntiReflectiveGlass",
    "Part_AntiReflectiveGlass_Circle",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brown-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args_after_dash())


def append_brown_material(source: Path) -> bpy.types.Material:
    old = bpy.data.materials.get(FUR_MATERIAL)
    if old is not None:
        bpy.data.materials.remove(old)
    with bpy.data.libraries.load(str(source), link=False) as (available, requested):
        if FUR_SOURCE_MATERIAL not in available.materials:
            raise KeyError(f"{FUR_SOURCE_MATERIAL!r} not found in {source}")
        requested.materials = [FUR_SOURCE_MATERIAL]
    material = requested.materials[0]
    if material is None:
        raise RuntimeError(f"Failed to append {FUR_SOURCE_MATERIAL!r}")
    material.name = FUR_MATERIAL
    color_ramp = next(
        (node for node in material.node_tree.nodes if node.type == "VALTORGB"),
        None,
    )
    if color_ramp is not None and len(color_ramp.color_ramp.elements) >= 2:
        color_ramp.color_ramp.elements[0].color = (0.014, 0.0065, 0.003, 1.0)
        color_ramp.color_ramp.elements[-1].color = (0.14, 0.055, 0.022, 1.0)
    hair = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED"),
        None,
    )
    if hair is not None:
        hair.inputs["Roughness"].default_value = 0.72
        if hair.inputs.get("Random Color") is not None:
            hair.inputs["Random Color"].default_value = 0.08
    return material


def assign_active_fur(material: bpy.types.Material) -> list[str]:
    assigned = []
    for obj in bpy.data.objects:
        if obj.type != "CURVES" or getattr(obj.data, "surface", None) is None:
            continue
        if obj.data.surface.name != ACTIVE_MOUSE:
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.hide_render = False
        assigned.append(obj.name)
    if not assigned:
        raise RuntimeError(f"No hair curves are bound to {ACTIVE_MOUSE!r}")
    return assigned


def configure_mouse_collections() -> None:
    active = bpy.data.collections.get(ACTIVE_COLLECTION)
    inactive = bpy.data.collections.get(INACTIVE_COLLECTION)
    if active is None or inactive is None:
        raise KeyError(f"Expected {ACTIVE_COLLECTION!r} and {INACTIVE_COLLECTION!r}")
    active.hide_render = False
    inactive.hide_render = True


def configure_lighting() -> dict[str, float]:
    energies = {
        "MGF_BottomFill": 4.0,
        "MGF_FullTrajectory_SideFill": 2.2,
        "日光": 2.8,
    }
    for name, energy in energies.items():
        light = bpy.data.objects.get(name)
        if light is not None and light.type == "LIGHT":
            light.data.energy = energy
    world_strength = 1.1
    world = bpy.context.scene.world
    if world is not None and world.use_nodes:
        background = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
        if background is not None:
            background.inputs["Color"].default_value = (0.14, 0.145, 0.15, 1.0)
            background.inputs["Strength"].default_value = world_strength
    bpy.context.scene.view_settings.exposure = 0.5
    bpy.context.scene.view_settings.gamma = 1.0
    return {**energies, "world_strength": world_strength, "exposure_ev": 0.5}


def configure_depth_ignore_collection() -> list[str]:
    collection = bpy.data.collections.get(DEPTH_IGNORE_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(DEPTH_IGNORE_COLLECTION)
        bpy.context.scene.collection.children.link(collection)
    linked = []
    for name in TRANSPARENT_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise KeyError(name)
        if collection.objects.get(obj.name) is None:
            collection.objects.link(obj)
        linked.append(obj.name)
    collection.hide_render = False
    collection.hide_viewport = False
    return linked


def main() -> None:
    args = parse_args()
    brown_material = append_brown_material(args.brown_source)
    assigned = assign_active_fur(brown_material)
    configure_mouse_collections()
    lighting = configure_lighting()
    transparent = configure_depth_ignore_collection()

    scene = bpy.context.scene
    scene["MGF_fur_style"] = "brown"
    scene["MGF_brown_fur_material"] = brown_material.name
    scene["MGF_depth_ignore_collection"] = DEPTH_IGNORE_COLLECTION
    scene["MGF_core23_armature"] = "DeformationSystem.001"
    scene["MGF_core23_mesh"] = ACTIVE_MOUSE
    scene.frame_set(1)
    bpy.context.view_layer.update()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    report = {
        "output": str(output),
        "fur_material": brown_material.name,
        "fur_objects": assigned,
        "lighting": lighting,
        "depth_ignore_collection": DEPTH_IGNORE_COLLECTION,
        "depth_ignore_objects": transparent,
    }
    output.with_suffix(".appearance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Transfer the validated ClimbWall appearance setup to the MoSeq scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


DONOR = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\BlenderPyMouseProjects\Cylinder\PyMouseStudio-BrownFur-ClimbWall.blend"
)
OUTPUT = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\PyMouseStudio-BrownFur-Controller-moseq-FullTrajectorySample10-First200-Core23-ClimbWallAppearance.blend"
)

FUR_MATERIALS = {
    "guard_long": "Black_Whiskers.006",
    "guard_short": "Black_Whiskers.007",
    "whisker": "Black_Whiskers.008",
    "dense": "Black_Whiskers.009",
}
BODY_MATERIALS = [
    "lambert2.004",
    "Black_Eyes.004",
    "Material.014",
    "Material.013",
    "Material.012",
]
GLASS_MATERIALS = ["color_cylinder.001", "color_anti-reflective_glass"]
METAL_MATERIAL = "color_AluminumSupport"


def append_materials(names: list[str]) -> dict[str, bpy.types.Material]:
    loaded = {}
    for source_name in names:
        with bpy.data.libraries.load(str(DONOR), link=False) as (data_from, data_to):
            if source_name not in data_from.materials:
                raise KeyError(f"Donor material not found: {source_name}")
            data_to.materials = [source_name]
        material = data_to.materials[0]
        if material is None:
            raise RuntimeError(f"Failed to append donor material: {source_name}")
        material.name = f"MGF_ClimbWall_{source_name}"
        loaded[source_name] = material
    return loaded


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def retarget_modifier_objects(
    obj: bpy.types.Object,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    for modifier in obj.modifiers:
        for key in modifier.keys():
            try:
                if modifier[key] == source_mesh:
                    modifier[key] = target_mesh
            except (ReferenceError, TypeError):
                continue


def duplicate_guard_hair(
    source_name: str,
    output_name: str,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
    material: bpy.types.Material,
    destination_collection: bpy.types.Collection,
) -> bpy.types.Object:
    existing = bpy.data.objects.get(output_name)
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)
    source = bpy.data.objects[source_name]
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = output_name
    duplicate.data.name = f"{output_name}_Curves"
    destination_collection.objects.link(duplicate)
    duplicate.parent = target_mesh
    duplicate.matrix_parent_inverse = target_mesh.matrix_world.inverted()
    duplicate.data.surface = target_mesh
    retarget_modifier_objects(duplicate, source_mesh, target_mesh)
    assign_material(duplicate, material)
    duplicate.hide_render = False
    duplicate.hide_viewport = False
    return duplicate


def configure_cycles(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.preview_samples = 64
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.max_bounces = 12
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def configure_reference_lighting(scene: bpy.types.Scene) -> None:
    if scene.world:
        scene.world.color = (0.0508761, 0.0508761, 0.0508761)
        background = scene.world.node_tree.nodes.get("Background") if scene.world.node_tree else None
        if background:
            background.inputs["Color"].default_value = (0.4701546, 0.4701546, 0.4701546, 1.0)
            background.inputs["Strength"].default_value = 0.45
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        obj.hide_render = obj.name != "日光"
    sunlight = bpy.data.objects.get("日光")
    if sunlight and sunlight.type == "LIGHT":
        sunlight.hide_render = False
        sunlight.data.energy = 5.0
        sunlight.data.color = (1.0, 1.0, 1.0)
        sunlight.data.angle = 0.174533
    for name in ("Building_Floor", "Part_TopCover", "Part_WhiteHexagonBackground"):
        enclosure = bpy.data.objects.get(name)
        if enclosure and hasattr(enclosure, "visible_shadow"):
            enclosure.visible_shadow = name != "Part_TopCover"


def create_procedural_aluminum() -> bpy.types.Material:
    old = bpy.data.materials.get("MGF_BrushedAluminum_Procedural")
    if old:
        bpy.data.materials.remove(old)
    material = bpy.data.materials.new("MGF_BrushedAluminum_Procedural")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    coordinates = nodes.new("ShaderNodeTexCoord")
    bump = nodes.new("ShaderNodeBump")
    shader.inputs["Metallic"].default_value = 0.95
    shader.inputs["Roughness"].default_value = 0.2
    shader.inputs["Anisotropic"].default_value = 0.35
    texture.noise_dimensions = "3D"
    texture.inputs["Scale"].default_value = 7.0
    texture.inputs["Detail"].default_value = 2.0
    texture.inputs["Roughness"].default_value = 0.35
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.12, 0.14, 0.16, 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.52, 0.58, 0.64, 1.0)
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.03
    links.new(coordinates.outputs["Generated"], texture.inputs["Vector"])
    links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(texture.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def main() -> None:
    active_mesh = bpy.data.objects["MS_Mouse.001"]
    alternate_mesh = bpy.data.objects["MS_Mouse.002"]
    if (len(active_mesh.data.vertices), len(active_mesh.data.polygons)) != (
        len(alternate_mesh.data.vertices),
        len(alternate_mesh.data.polygons),
    ):
        raise RuntimeError("Active and alternate mouse meshes do not share topology")

    requested = [
        *FUR_MATERIALS.values(),
        *BODY_MATERIALS,
        *GLASS_MATERIALS,
        METAL_MATERIAL,
    ]
    donor = append_materials(requested)

    for index, source_name in enumerate(BODY_MATERIALS):
        if index >= len(active_mesh.material_slots):
            active_mesh.data.materials.append(donor[source_name])
        else:
            active_mesh.material_slots[index].material = donor[source_name]

    dense = donor[FUR_MATERIALS["dense"]]
    for name in (
        "BaseFur.001",
        "EarFur.001",
        "EyesFur.001",
        "LegFur.001",
        "Mouth.001",
        "TailFur.001",
    ):
        assign_material(bpy.data.objects[name], dense)
    assign_material(bpy.data.objects["Whisker.001"], donor[FUR_MATERIALS["whisker"]])

    destination_collection = bpy.data.objects["BaseFur.001"].users_collection[0]
    guard_short = duplicate_guard_hair(
        "BaseFur.003",
        "MGF_BaseFur_GuardShort",
        alternate_mesh,
        active_mesh,
        donor[FUR_MATERIALS["guard_short"]],
        destination_collection,
    )
    guard_long = duplicate_guard_hair(
        "BaseFur.004",
        "MGF_BaseFur_GuardLong",
        alternate_mesh,
        active_mesh,
        donor[FUR_MATERIALS["guard_long"]],
        destination_collection,
    )

    cylinder = bpy.data.objects["Part_GlassCylinder160mm"]
    platform = bpy.data.objects["Part_AntiReflectiveGlass"]
    platform_circle = bpy.data.objects.get("Part_AntiReflectiveGlass_Circle")
    assign_material(cylinder, donor["color_cylinder.001"])
    assign_material(platform, donor["color_anti-reflective_glass"])
    if platform_circle:
        assign_material(platform_circle, donor["color_anti-reflective_glass"])

    donor_metal = donor[METAL_MATERIAL]
    procedural_metal = create_procedural_aluminum()
    metal_objects = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        uses_aluminum = obj.name.startswith("MGF_SupportRing_") or any(
            slot.material and slot.material.name.startswith("color_AluminumSupport")
            for slot in obj.material_slots
        )
        if not uses_aluminum:
            continue
        target_metal = procedural_metal if obj.name.startswith("MGF_") else donor_metal
        for slot in obj.material_slots:
            if obj.name.startswith("MGF_SupportRing_") or (
                slot.material and slot.material.name.startswith("color_AluminumSupport")
            ):
                slot.material = target_metal
        metal_objects.append(obj.name)

    scene = bpy.context.scene
    configure_cycles(scene)
    configure_reference_lighting(scene)
    scene["MGF_appearance_reference"] = str(DONOR)
    scene["MGF_appearance_revision"] = "climbwall_exact_materials_guard_hair_cycles_v1"
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT))

    audit = {
        "output": str(OUTPUT),
        "donor": str(DONOR),
        "body_materials": [donor[name].name for name in BODY_MATERIALS],
        "fur_materials": {role: donor[name].name for role, name in FUR_MATERIALS.items()},
        "guard_hair": {
            guard_short.name: [len(guard_short.data.curves), len(guard_short.data.points)],
            guard_long.name: [len(guard_long.data.curves), len(guard_long.data.points)],
        },
        "glass": {
            cylinder.name: cylinder.active_material.name,
            platform.name: platform.active_material.name,
            platform_circle.name if platform_circle else "": platform_circle.active_material.name
            if platform_circle
            else None,
        },
        "metal_object_count": len(metal_objects),
        "metal_objects": metal_objects,
        "cycles_samples": scene.cycles.samples,
    }
    OUTPUT.with_suffix(".appearance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

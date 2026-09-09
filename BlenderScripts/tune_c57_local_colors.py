"""Create a topology-stable C57 local-colour material variant.

The script assigns semantic skin materials from rig weights in the undeformed
mesh. Material indices then follow the skinned topology at every animation
frame, avoiding frame-dependent world-space masks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


MOUSE_NAME = "MS_Mouse.001"
FUR_MATERIALS = {
    "BaseFur.001": ("MGF_C57_LocalV2_Coat", (0.0011, 0.0009, 0.0008, 1.0), 0.52, 0.40),
    "EyesFur.001": ("MGF_C57_LocalV2_Coat", (0.0011, 0.0009, 0.0008, 1.0), 0.52, 0.40),
    "EarFur.001": ("MGF_C57_LocalV2_EarFur", (0.0220, 0.0100, 0.0070, 1.0), 0.58, 0.45),
    "LegFur.001": ("MGF_C57_LocalV2_LegFur", (0.0015, 0.0010, 0.0009, 1.0), 0.55, 0.43),
    "TailFur.001": ("MGF_C57_LocalV2_TailFur", (0.0035, 0.0021, 0.0018, 1.0), 0.62, 0.49),
}


def set_input(node: bpy.types.Node, name: str, value: object) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def copy_material(source: bpy.types.Material, name: str) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = source.copy()
        material.name = name
    material.use_fake_user = True
    material["MGF_C57_local_colour_v2"] = True
    material["MGF_source_material"] = source.name
    return material


def tune_hair_material(
    source: bpy.types.Material,
    name: str,
    colour: tuple[float, float, float, float],
    roughness: float,
    radial_roughness: float,
) -> bpy.types.Material:
    material = copy_material(source, name)
    hair = next((node for node in material.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED"), None)
    if hair is None:
        raise RuntimeError(f"{source.name} has no Principled Hair BSDF")
    hair.parametrization = "COLOR"
    set_input(hair, "Color", colour)
    set_input(hair, "Roughness", roughness)
    set_input(hair, "Radial Roughness", radial_roughness)
    set_input(hair, "Coat", 0.025)
    set_input(hair, "Random Color", 0.07)
    set_input(hair, "Random Roughness", 0.12)
    return material


def tune_skin_ramps(material: bpy.types.Material, colours: tuple[tuple[float, ...], tuple[float, ...]]) -> None:
    ramps = [node for node in material.node_tree.nodes if node.type == "VALTORGB"]
    if not ramps:
        raise RuntimeError(f"{material.name} has no skin colour ramp")
    for ramp in ramps:
        ramp.color_ramp.elements[0].color = colours[0]
        ramp.color_ramp.elements[-1].color = colours[1]
    for node in material.node_tree.nodes:
        if node.type == "SUBSURFACE_SCATTERING":
            set_input(node, "Color", colours[0])
            set_input(node, "Roughness", 0.78)


def make_skin_material(
    name: str,
    colour: tuple[float, float, float, float],
    roughness: float,
    coat_weight: float,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    material.use_fake_user = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(shader, "Base Color", colour)
    set_input(shader, "Roughness", roughness)
    set_input(shader, "IOR", 1.45)
    set_input(shader, "Specular IOR Level", 0.28)
    set_input(shader, "Coat Weight", coat_weight)
    set_input(shader, "Subsurface Weight", 0.025)
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material["MGF_C57_local_colour_v2"] = True
    return material


def ensure_material_slot(obj: bpy.types.Object, material: bpy.types.Material) -> int:
    for index, slot in enumerate(obj.material_slots):
        if slot.material == material:
            return index
    obj.data.materials.append(material)
    return len(obj.material_slots) - 1


def vertex_weight_sum(obj: bpy.types.Object, vertex_index: int, group_indices: set[int]) -> float:
    return sum(
        element.weight
        for element in obj.data.vertices[vertex_index].groups
        if element.group in group_indices
    )


def assign_weighted_region(
    obj: bpy.types.Object,
    group_names: list[str],
    material_index: int,
    threshold: float = 0.65,
) -> int:
    group_indices = {obj.vertex_groups[name].index for name in group_names if name in obj.vertex_groups}
    if not group_indices:
        raise RuntimeError(f"None of the requested groups exist: {group_names}")
    assigned = 0
    for polygon in obj.data.polygons:
        # Only split the original body material. Eyes, ears and paws retain their semantic slots.
        if polygon.material_index != 0:
            continue
        weight = sum(
            vertex_weight_sum(obj, vertex_index, group_indices)
            for vertex_index in polygon.vertices
        ) / len(polygon.vertices)
        if weight >= threshold:
            polygon.material_index = material_index
            assigned += 1
    return assigned


def apply_local_colours() -> dict[str, object]:
    mouse = bpy.data.objects.get(MOUSE_NAME)
    if mouse is None or mouse.type != "MESH":
        raise RuntimeError(f"Missing skinned mouse mesh: {MOUSE_NAME}")
    if len(mouse.material_slots) < 5:
        raise RuntimeError("Mouse mesh does not have the expected body/eye/paw/ear material slots")

    # Preserve the existing shader topology while moving the exposed skin into
    # the muted grey-pink range seen in visible-light C57 references.
    hind_paw = copy_material(mouse.material_slots[2].material, "MGF_C57_LocalV2_HindPaw")
    front_paw = copy_material(mouse.material_slots[3].material, "MGF_C57_LocalV2_FrontPaw")
    ear_skin = copy_material(mouse.material_slots[4].material, "MGF_C57_LocalV2_EarSkin")
    tune_skin_ramps(hind_paw, ((0.080, 0.036, 0.033, 1.0), (0.205, 0.095, 0.082, 1.0)))
    tune_skin_ramps(front_paw, ((0.068, 0.030, 0.028, 1.0), (0.180, 0.078, 0.068, 1.0)))
    tune_skin_ramps(ear_skin, ((0.380, 0.170, 0.140, 1.0), (0.100, 0.040, 0.034, 1.0)))
    for node in ear_skin.node_tree.nodes:
        if node.type == "SUBSURFACE_SCATTERING" and any(
            link.from_node == node for link in ear_skin.node_tree.links
        ):
            set_input(node, "Scale", 0.25)
    mouse.material_slots[2].material = hind_paw
    mouse.material_slots[3].material = front_paw
    mouse.material_slots[4].material = ear_skin

    nose = make_skin_material("MGF_C57_LocalV2_Nose", (0.021, 0.0080, 0.0085, 1.0), 0.47, 0.075)
    tail = make_skin_material("MGF_C57_LocalV2_TailSkin", (0.066, 0.036, 0.033, 1.0), 0.74, 0.01)
    nose_index = ensure_material_slot(mouse, nose)
    tail_index = ensure_material_slot(mouse, tail)
    tail_groups = [name for name in mouse.vertex_groups.keys() if name.startswith("Tail")]
    tail_faces = assign_weighted_region(mouse, tail_groups, tail_index)
    nose_faces = assign_weighted_region(mouse, ["HeadEnd_M"], nose_index)

    coat_source = bpy.data.materials.get("MGF_C57_First200_Coat")
    if coat_source is None:
        raise RuntimeError("Missing source C57 coat material")
    created: dict[str, bpy.types.Material] = {}
    for object_name, (material_name, colour, roughness, radial) in FUR_MATERIALS.items():
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "CURVES":
            raise RuntimeError(f"Missing fur curves object: {object_name}")
        material = created.get(material_name)
        if material is None:
            material = tune_hair_material(coat_source, material_name, colour, roughness, radial)
            created[material_name] = material
        obj.data.materials.clear()
        obj.data.materials.append(material)

    mouse["MGF_C57_local_colour_v2"] = True
    mouse["MGF_C57_tail_faces"] = tail_faces
    mouse["MGF_C57_nose_faces"] = nose_faces
    return {
        "mouse": mouse.name,
        "tail_faces": tail_faces,
        "nose_faces": nose_faces,
        "fur_materials": sorted(created),
    }


def blender_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-blend", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = blender_args()
    result = apply_local_colours()
    if args.output_blend:
        args.output_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
        result["saved"] = str(args.output_blend)
    print(result)


if __name__ == "__main__":
    main()

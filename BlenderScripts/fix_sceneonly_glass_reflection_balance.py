"""Restore ceiling emission and reduce glass reflection without clouding transmission."""

import bpy


CEILING_MATERIAL = "Material.022"
CEILING_EMISSION = 11.8
GLASS_SETTINGS = {
    "color_cylinder.003": {"reflection_scale": 0.16, "roughness": 0.012},
    "color_anti-reflective_glass.002": {"reflection_scale": 0.06, "roughness": 0.006},
}
REFLECTION_NODE_NAME = "MGF_FresnelReflectionScale"


def restore_ceiling_emission():
    material = bpy.data.materials.get(CEILING_MATERIAL)
    if material is None or material.node_tree is None:
        raise RuntimeError(f"Missing node material: {CEILING_MATERIAL}")
    emission_nodes = [node for node in material.node_tree.nodes if node.type == "EMISSION"]
    if len(emission_nodes) != 1:
        raise RuntimeError(
            f"Expected one Emission node in {CEILING_MATERIAL}, found {len(emission_nodes)}"
        )
    emission_nodes[0].inputs["Strength"].default_value = CEILING_EMISSION


def configure_glass(material_name, reflection_scale, roughness):
    material = bpy.data.materials.get(material_name)
    if material is None or material.node_tree is None:
        raise RuntimeError(f"Missing glass node material: {material_name}")

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    fresnel_nodes = [node for node in nodes if node.type == "FRESNEL"]
    mix_nodes = [node for node in nodes if node.type == "MIX_SHADER"]
    glass_nodes = [node for node in nodes if node.type == "BSDF_GLASS"]
    if len(fresnel_nodes) != 1 or len(mix_nodes) != 1 or len(glass_nodes) != 1:
        raise RuntimeError(
            f"Unexpected glass graph in {material_name}: "
            f"Fresnel={len(fresnel_nodes)}, Mix={len(mix_nodes)}, Glass={len(glass_nodes)}"
        )

    fresnel = fresnel_nodes[0]
    mix = mix_nodes[0]
    glass = glass_nodes[0]
    glass.inputs["Roughness"].default_value = roughness

    scale = nodes.get(REFLECTION_NODE_NAME)
    if scale is None:
        scale = nodes.new("ShaderNodeMath")
        scale.name = REFLECTION_NODE_NAME
        scale.label = "MGF Fresnel reflection scale"
        scale.operation = "MULTIPLY"
        scale.location = ((fresnel.location.x + mix.location.x) / 2.0, fresnel.location.y)
    scale.inputs[1].default_value = reflection_scale

    for link in list(mix.inputs[0].links):
        links.remove(link)
    if not any(link.from_node == fresnel and link.to_node == scale for link in links):
        links.new(fresnel.outputs["Fac"], scale.inputs[0])
    links.new(scale.outputs[0], mix.inputs[0])


def main():
    restore_ceiling_emission()
    for material_name, settings in GLASS_SETTINGS.items():
        configure_glass(material_name, **settings)

    scene = bpy.context.scene
    scene["mgf_ceiling_tube_emission"] = CEILING_EMISSION
    scene["mgf_glass_reflection_revision"] = 2
    scene["mgf_glass_reflection_notes"] = (
        "Fresnel reflection scaled to 0.16 cylinder / 0.06 platform; "
        "Glass roughness 0.012 / 0.006"
    )
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print(f"Restored {CEILING_MATERIAL} emission to {CEILING_EMISSION}")
    for material_name, settings in GLASS_SETTINGS.items():
        print(f"Updated {material_name}: {settings}")
    print(f"Saved: {bpy.data.filepath}")


if __name__ == "__main__":
    main()

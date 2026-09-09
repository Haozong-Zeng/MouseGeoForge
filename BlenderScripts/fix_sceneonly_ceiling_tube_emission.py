"""Reduce the modeled classroom tube emission without changing scene lights."""

import bpy


MATERIAL_NAME = "Material.022"
TARGET_STRENGTH = 3.5


def main():
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is None or material.node_tree is None:
        raise RuntimeError(f"Missing node material: {MATERIAL_NAME}")

    emission_nodes = [node for node in material.node_tree.nodes if node.type == "EMISSION"]
    if len(emission_nodes) != 1:
        raise RuntimeError(
            f"Expected one Emission node in {MATERIAL_NAME}, found {len(emission_nodes)}"
        )

    emission = emission_nodes[0]
    previous = float(emission.inputs["Strength"].default_value)
    emission.inputs["Strength"].default_value = TARGET_STRENGTH

    scene = bpy.context.scene
    scene["mgf_ceiling_tube_emission_previous"] = previous
    scene["mgf_ceiling_tube_emission"] = TARGET_STRENGTH
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print(
        f"Updated {MATERIAL_NAME} Emission Strength: "
        f"{previous:.4f} -> {TARGET_STRENGTH:.4f}"
    )
    print(f"Saved: {bpy.data.filepath}")


if __name__ == "__main__":
    main()

"""Apply the visually selected warm-grey C57 ear material and save the blend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-blend", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    skin = bpy.data.materials["MGF_C57_LocalV2_EarSkin"]
    fur = bpy.data.materials["MGF_C57_LocalV2_EarFur"]
    hair = next(node for node in fur.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED")
    hair.inputs["Color"].default_value = (0.0220, 0.0100, 0.0070, 1.0)
    for node in skin.node_tree.nodes:
        if node.type == "VALTORGB":
            node.color_ramp.elements[0].color = (0.380, 0.170, 0.140, 1.0)
            node.color_ramp.elements[-1].color = (0.100, 0.040, 0.034, 1.0)
        elif node.type == "SUBSURFACE_SCATTERING" and any(
            link.from_node == node for link in skin.node_tree.links
        ):
            node.inputs["Scale"].default_value = 0.25
    scene = bpy.context.scene
    scene["MGF_C57_ear_material_profile"] = "warm-grey-visible-light-v1"
    scene["MGF_C57_ear_material_visual_reference"] = "C57BL/6J visible-light supplier references"
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()), check_existing=False)
    print("Applied C57 warm-grey ear material")


if __name__ == "__main__":
    main()

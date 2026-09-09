#!/usr/bin/env python3
"""Replace the dual-mouse brown coat with the validated Controller-MoSeq coat."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


TARGET_MATERIAL = "MGF_BrownFur_ControllerMoseq"
HOMECAGE_RAMP_COLORS = (
    (0.0100, 0.0042, 0.0018, 1.0),
    (0.0750, 0.0250, 0.0085, 1.0),
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def ramp_summary(material: bpy.types.Material) -> list[dict]:
    ramp = next((node for node in material.node_tree.nodes if node.type == "VALTORGB"), None)
    if ramp is None:
        raise RuntimeError(f"{material.name} has no Color Ramp")
    return [
        {"position": element.position, "color": list(element.color)}
        for element in ramp.color_ramp.elements
    ]


def append_material(source: Path, source_name: str) -> bpy.types.Material:
    existing = bpy.data.materials.get(TARGET_MATERIAL)
    if existing is not None:
        bpy.data.materials.remove(existing)
    with bpy.data.libraries.load(str(source), link=False) as (available, requested):
        if source_name not in available.materials:
            raise KeyError(f"{source_name!r} missing from {source}")
        requested.materials = [source_name]
    material = requested.materials[0]
    if material is None:
        raise RuntimeError("Brown material append failed")
    material.name = TARGET_MATERIAL
    # Accessing the ramp here is also a structural guard: the older SceneOnly
    # material had a single unlinked orange Color and caused the mismatch.
    ramp = next(node for node in material.node_tree.nodes if node.type == "VALTORGB")
    if len(ramp.color_ramp.elements) != 2:
        raise RuntimeError("Expected the two-stop Controller-MoSeq brown ramp")
    for element, color in zip(ramp.color_ramp.elements, HOMECAGE_RAMP_COLORS):
        element.color = color
    # The donor drove its ramp from Generated-Z, which creates broad bands once
    # the coat is cloned and arbitrarily rotated.  Preserve the validated brown
    # range but vary it per strand for rotation-invariant natural heterogeneity.
    for link in list(material.node_tree.links):
        if link.to_node == ramp and link.to_socket == ramp.inputs["Factor"]:
            material.node_tree.links.remove(link)
    hair_info = material.node_tree.nodes.get("MGF_Brown_StrandRandom")
    if hair_info is None:
        hair_info = material.node_tree.nodes.new("ShaderNodeHairInfo")
        hair_info.name = "MGF_Brown_StrandRandom"
        hair_info.label = "Rotation-invariant strand variation"
    material.node_tree.links.new(hair_info.outputs["Random"], ramp.inputs["Factor"])
    return material


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-material", default="MGF_BrownFur_KeypointMoseq")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(args_after_dash())

    brown_mesh = bpy.data.objects.get("LP_Brown_Mesh")
    if brown_mesh is None:
        raise KeyError("LP_Brown_Mesh")
    material = append_material(args.donor.resolve(), args.donor_material)
    assigned = []
    for obj in bpy.data.objects:
        if not obj.name.startswith("LP_Brown_") or obj.type != "CURVES":
            continue
        if getattr(obj.data, "surface", None) != brown_mesh:
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.hide_render = False
        assigned.append({
            "object": obj.name,
            "curve_count": len(obj.data.curves),
            "point_count": len(obj.data.points),
        })
    if not assigned:
        raise RuntimeError("No LP_Brown fur curves were assigned")

    old = bpy.data.materials.get("MGF_BrownFur_LightningPosePilot")
    if old is not None and old.users == 0:
        bpy.data.materials.remove(old)

    scene = bpy.context.scene
    scene["MGF_brown_fur_profile"] = "Controller-MoSeq validated dark brown Color Ramp"
    scene["MGF_brown_fur_material"] = material.name
    scene["MGF_brown_fur_donor"] = str(args.donor.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), check_existing=False)

    report = {
        "schema": "mousegeoforge.dual_mouse_brownfur_controller_match.v1",
        "output_blend": str(args.output),
        "donor_blend": str(args.donor.resolve()),
        "donor_material": args.donor_material,
        "target_material": material.name,
        "color_driver": "Hair Info.Random (rotation invariant per-strand variation)",
        "color_ramp": ramp_summary(material),
        "assigned_curve_objects": assigned,
        "assigned_curve_object_count": len(assigned),
        "assigned_curve_count": sum(row["curve_count"] for row in assigned),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "assigned_curve_objects"}, indent=2))


if __name__ == "__main__":
    main()

"""Build a black-fur WashFace scene without changing the source animation.

Run with Blender 5.1, for example:
  blender -b SOURCE.blend --python build_blackfur_washface.py

The script is intentionally idempotent: it replaces only MGF-owned materials
and the forelimb compensation modifiers before saving a new .blend file.
"""

from __future__ import annotations

import json
import os

import bpy


BLENDER_DIR = os.path.dirname(os.path.abspath(bpy.data.filepath))
ROOT = os.path.dirname(BLENDER_DIR)
SOURCE_BLEND = os.path.join(
    BLENDER_DIR,
    "PyMouseStudio-WhiteFur-WaterReaching-FrameToFrame-WashFace.blend",
)
DONOR_BLEND = os.path.join(
    BLENDER_DIR,
    "PyMouseStudio-BrowmFur-WaterReaching-IK-BilateralHand-Control-HairGN-Experiment-V19-PalmToMouth-CompactFingers.blend",
)
OUTPUT_BLEND = os.path.join(
    BLENDER_DIR,
    "PyMouseStudio-BlackFur-WaterReaching-FrameToFrame-WashFace-ForelimbFurFix.blend",
)
REPORT_PATH = os.path.join(ROOT, "output", "blackfur_washface", "build_report.json")

COMPENSATION_GROUP = "GN_Forelimb_Fur_Stretch_Compensation"
COMPENSATION_OBJECTS = ("BaseFur.003", "LegFur.001")
BODY_FUR_OBJECTS = (
    "BaseFur.003",
    "EyesFur.001",
    "LegFur.001",
    "Mouth.001",
    "TailFur.001",
)
EAR_FUR_OBJECTS = ("EarFur.001",)
WHISKER_OBJECTS = ("Whisker.001",)


def ensure_source_file() -> None:
    current = os.path.normcase(os.path.abspath(bpy.data.filepath))
    expected = os.path.normcase(os.path.abspath(SOURCE_BLEND))
    if current != expected:
        raise RuntimeError(f"Open the WashFace source first. Got: {bpy.data.filepath}")


def replace_material(name: str) -> bpy.types.Material:
    old = bpy.data.materials.get(name)
    if old:
        bpy.data.materials.remove(old, do_unlink=True)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (0.018, 0.022, 0.028, 1.0)
    return material


def make_black_hair_material(
    name: str,
    colors: tuple[tuple[float, float, float, float], ...],
    roughness: float,
    radial_roughness: float,
) -> bpy.types.Material:
    material = replace_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (500, 0)

    hair = nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.location = (220, 0)
    hair.name = "MGF Principled Hair"
    if hasattr(hair, "parametrization"):
        hair.parametrization = "COLOR"
    hair.inputs["Roughness"].default_value = roughness
    hair.inputs["Radial Roughness"].default_value = radial_roughness
    hair.inputs["IOR"].default_value = 1.45

    hair_info = nodes.new("ShaderNodeHairInfo")
    hair_info.location = (-500, 20)
    hair_info.name = "MGF Per-strand Variation"

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-210, 20)
    ramp.name = "MGF Charcoal Fur Range"
    ramp.color_ramp.interpolation = "EASE"
    while len(ramp.color_ramp.elements) > 2:
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])
    ramp.color_ramp.elements[0].position = 0.08
    ramp.color_ramp.elements[0].color = colors[0]
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = colors[-1]
    if len(colors) == 3:
        middle = ramp.color_ramp.elements.new(0.52)
        middle.color = colors[1]

    links.new(hair_info.outputs["Random"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], hair.inputs["Color"])
    links.new(hair.outputs[0], output.inputs["Surface"])
    return material


def assign_single_material(object_names: tuple[str, ...], material: bpy.types.Material) -> list[str]:
    assigned = []
    for object_name in object_names:
        obj = bpy.data.objects.get(object_name)
        if not obj or obj.type != "CURVES":
            continue
        obj.hide_viewport = False
        obj.hide_render = False
        obj.hide_set(False)
        obj.data.materials.clear()
        obj.data.materials.append(material)
        assigned.append(object_name)
    return assigned


def set_duplicate_density(
    object_name: str,
    count: int,
    radius: float | None = None,
) -> dict[str, object]:
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise RuntimeError(f"Missing fur object: {object_name}")
    modifier = obj.modifiers.get("Duplicate Hair Curves")
    if not modifier:
        raise RuntimeError(f"Duplicate Hair Curves modifier missing on {object_name}")
    old_count = int(modifier.get("Input_2", 0))
    old_radius = float(modifier.get("Input_5", 0.0))
    modifier["Input_2"] = int(count)
    if radius is not None:
        modifier["Input_5"] = float(radius)
    return {
        "object": object_name,
        "old_count": old_count,
        "new_count": int(count),
        "old_radius": old_radius,
        "new_radius": float(modifier.get("Input_5", 0.0)),
    }


def append_compensation_group() -> bpy.types.NodeTree:
    existing = bpy.data.node_groups.get(COMPENSATION_GROUP)
    if existing:
        return existing
    with bpy.data.libraries.load(DONOR_BLEND, link=False) as (data_from, data_to):
        if COMPENSATION_GROUP not in data_from.node_groups:
            raise RuntimeError(f"Missing node group in donor: {COMPENSATION_GROUP}")
        data_to.node_groups = [COMPENSATION_GROUP]
    group = bpy.data.node_groups.get(COMPENSATION_GROUP)
    if not group:
        raise RuntimeError(f"Failed to append node group: {COMPENSATION_GROUP}")
    return group


def retarget_group_object_info(group: bpy.types.NodeTree, body: bpy.types.Object) -> int:
    changed = 0
    for node in group.nodes:
        object_input = node.inputs.get("Object") if hasattr(node, "inputs") else None
        if object_input is not None and hasattr(object_input, "default_value"):
            value = object_input.default_value
            if value is not None or node.bl_idname == "GeometryNodeObjectInfo":
                object_input.default_value = body
                changed += 1
    return changed


def add_compensation_modifier(
    obj: bpy.types.Object,
    group: bpy.types.NodeTree,
) -> dict[str, object]:
    for modifier in list(obj.modifiers):
        if modifier.name == "Forelimb Fur Stretch Compensation":
            obj.modifiers.remove(modifier)

    modifier = obj.modifiers.new("Forelimb Fur Stretch Compensation", "NODES")
    modifier.node_group = group
    modifier.show_viewport = True
    modifier.show_render = True

    values = {"Socket_2": 0.32, "Socket_3": 0.05, "Socket_4": 0.65}
    for key, value in values.items():
        try:
            modifier[key] = value
        except Exception:
            pass

    surface_index = next(
        (i for i, item in enumerate(obj.modifiers) if item.name == "Surface Deform"),
        -1,
    )
    target_index = surface_index + 1 if surface_index >= 0 else 0
    obj.modifiers.move(len(obj.modifiers) - 1, target_index)
    return {
        "object": obj.name,
        "index": target_index,
        "settings": {key: modifier.get(key) for key in values},
    }


def animation_audit() -> dict[str, object]:
    armature = bpy.data.objects.get("DeformationSystem")
    if not armature or armature.type != "ARMATURE":
        raise RuntimeError("DeformationSystem armature not found")
    action = armature.animation_data.action if armature.animation_data else None
    ik_constraints = []
    for bone in armature.pose.bones:
        for constraint in bone.constraints:
            if constraint.type in {"IK", "SPLINE_IK"}:
                ik_constraints.append(f"{bone.name}:{constraint.name}:{constraint.type}")
    return {
        "armature": armature.name,
        "bones": len(armature.data.bones),
        "action": action.name if action else None,
        "action_range": list(action.frame_range) if action else None,
        "ik_constraints": ik_constraints,
    }


def main() -> None:
    ensure_source_file()
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    body_material = make_black_hair_material(
        "MGF_BlackFur_Body",
        (
            (0.0045, 0.0060, 0.0090, 1.0),
            (0.0160, 0.0210, 0.0280, 1.0),
            (0.0460, 0.0410, 0.0370, 1.0),
        ),
        roughness=0.56,
        radial_roughness=0.34,
    )
    ear_material = make_black_hair_material(
        "MGF_BlackFur_Ear",
        (
            (0.0070, 0.0080, 0.0110, 1.0),
            (0.0260, 0.0240, 0.0250, 1.0),
            (0.0640, 0.0490, 0.0450, 1.0),
        ),
        roughness=0.60,
        radial_roughness=0.36,
    )
    whisker_material = make_black_hair_material(
        "MGF_BlackFur_Whisker",
        (
            (0.0015, 0.0020, 0.0030, 1.0),
            (0.0120, 0.0150, 0.0190, 1.0),
        ),
        roughness=0.42,
        radial_roughness=0.28,
    )

    assigned = {
        "body": assign_single_material(BODY_FUR_OBJECTS, body_material),
        "ear": assign_single_material(EAR_FUR_OBJECTS, ear_material),
        "whisker": assign_single_material(WHISKER_OBJECTS, whisker_material),
    }
    limb_density = set_duplicate_density("LegFur.001", 5, radius=0.02)

    body = bpy.data.objects.get("MS_Mouse")
    if not body:
        raise RuntimeError("MS_Mouse surface not found")
    group = append_compensation_group()
    retarget_count = retarget_group_object_info(group, body)
    compensation = []
    for object_name in COMPENSATION_OBJECTS:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise RuntimeError(f"Missing fur object: {object_name}")
        compensation.append(add_compensation_modifier(obj, group))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 181
    scene.frame_set(1)

    audit = animation_audit()
    report = {
        "source": SOURCE_BLEND,
        "donor": DONOR_BLEND,
        "output": OUTPUT_BLEND,
        "materials": assigned,
        "limb_density": limb_density,
        "compensation": compensation,
        "object_info_nodes_retargeted": retarget_count,
        "animation": audit,
        "body_materials_preserved": [slot.material.name for slot in body.material_slots if slot.material],
    }

    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    print("MGF_BLACKFUR_BUILD_REPORT=" + json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()

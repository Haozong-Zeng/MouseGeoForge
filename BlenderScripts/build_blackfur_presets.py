"""Build the FG and FS non-IK presets with the calibrated black-fur setup."""

from __future__ import annotations

import json
import os
import sys

import bpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_blackfur_washface import (  # noqa: E402
    COMPENSATION_GROUP,
    DONOR_BLEND,
    add_compensation_modifier,
    animation_audit,
    append_compensation_group,
    assign_single_material,
    make_black_hair_material,
    retarget_group_object_info,
    set_duplicate_density,
)


PRESETS = {
    "PyMouseStudio-BrowmFur-WaterReaching-FrameToFrame-FGcamB.blend": {
        "label": "FG",
        "output": "PyMouseStudio-BlackFur-WaterReaching-FrameToFrame-FGcamB-ForelimbFurFix.blend",
    },
    "PyMouseStudio-BrowmFur-WaterReaching-FrameToFrame-FScamC.blend": {
        "label": "FS",
        "output": "PyMouseStudio-BlackFur-WaterReaching-FrameToFrame-FScamC-ForelimbFurFix.blend",
    },
}

BODY_FUR_OBJECTS = (
    "BaseFur.003",
    "BaseFur.004",
    "BaseFur.005",
    "EyesFur.001",
    "LegFur.001",
    "Mouth.001",
    "TailFur.001",
)
EAR_FUR_OBJECTS = ("EarFur.001",)
WHISKER_OBJECTS = ("Whisker.001",)
COMPENSATION_OBJECTS = (
    "BaseFur.003",
    "BaseFur.004",
    "BaseFur.005",
    "LegFur.001",
)


def identify_preset() -> dict[str, str]:
    source_name = os.path.basename(bpy.data.filepath)
    preset = PRESETS.get(source_name)
    if not preset:
        raise RuntimeError(
            "Expected the BrownFur FGcamB or FScamC source, got: " + source_name
        )
    return preset


def active_mouse_audit() -> dict[str, object]:
    body = bpy.data.objects.get("MS_Mouse")
    if not body or body.type != "MESH":
        raise RuntimeError("Animated MS_Mouse body not found")
    armature_targets = [
        modifier.object.name
        for modifier in body.modifiers
        if modifier.type == "ARMATURE" and modifier.object
    ]
    return {
        "body": body.name,
        "parent": body.parent.name if body.parent else None,
        "armature_modifiers": armature_targets,
        "fur_objects": {
            name: {
                "surface": bpy.data.objects[name].data.surface.name,
                "parent": bpy.data.objects[name].parent.name,
            }
            for name in BODY_FUR_OBJECTS + EAR_FUR_OBJECTS + WHISKER_OBJECTS
        },
    }


def main() -> None:
    preset = identify_preset()
    root = os.path.dirname(SCRIPT_DIR)
    output_blend = os.path.join(SCRIPT_DIR, preset["output"])
    report_dir = os.path.join(root, "output", "blackfur_presets", preset["label"])
    report_path = os.path.join(report_dir, "build_report.json")
    os.makedirs(report_dir, exist_ok=True)

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

    limb_distribution = set_duplicate_density("LegFur.001", 5, radius=0.02)
    body = bpy.data.objects["MS_Mouse"]
    group = append_compensation_group()
    retarget_count = retarget_group_object_info(group, body)
    compensation = []
    for object_name in COMPENSATION_OBJECTS:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise RuntimeError(f"Missing active fur object: {object_name}")
        compensation.append(add_compensation_modifier(obj, group))

    audit = animation_audit()
    scene = bpy.context.scene
    scene.frame_start = int(round(audit["action_range"][0]))
    scene.frame_end = int(round(audit["action_range"][1]))
    scene.frame_set(scene.frame_start)

    report = {
        "preset": preset["label"],
        "source": bpy.data.filepath,
        "donor": DONOR_BLEND,
        "output": output_blend,
        "materials": assigned,
        "limb_distribution": limb_distribution,
        "compensation_group": COMPENSATION_GROUP,
        "compensation": compensation,
        "object_info_nodes_retargeted": retarget_count,
        "animation": audit,
        "active_mouse": active_mouse_audit(),
    }

    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    print("MGF_BLACKFUR_PRESET_REPORT=" + json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()

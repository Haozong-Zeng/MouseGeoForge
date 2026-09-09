"""Prepare the current SceneOnly MoSeq project for Core-23 dataset rendering."""

from pathlib import Path
import sys

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fix_sceneonly_brown_fur_lighting as brown_lighting  # noqa: E402
import fix_sceneonly_glass_reflection_balance as glass_balance  # noqa: E402


DEPTH_IGNORE_COLLECTION = "MGF_DepthIgnoreTransparentGlass"
DEPTH_IGNORE_OBJECTS = (
    "Part_GlassCylinder160mm.001",
    "Part_AntiReflectiveGlass.001",
)


def world_dimensions(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return [
        max(point[axis] for point in corners) - min(point[axis] for point in corners)
        for axis in range(3)
    ]


def validate_millimeter_scale(scene):
    scale_length = float(scene.unit_settings.scale_length)
    if (
        scene.unit_settings.system != "METRIC"
        or scene.unit_settings.length_unit != "MILLIMETERS"
        or abs(scale_length - 0.001) >= 1e-8
    ):
        raise ValueError(
            "Expected raw-millimeter scene units: METRIC/MILLIMETERS/0.001, got "
            f"{scene.unit_settings.system}/{scene.unit_settings.length_unit}/{scale_length}"
        )
    cylinder_dimensions = world_dimensions(bpy.data.objects["Part_GlassCylinder160mm.001"])
    mouse_dimensions = world_dimensions(bpy.data.objects["MS_Mouse.001"])
    cylinder_diameter = min(cylinder_dimensions[:2])
    mouse_length = max(mouse_dimensions)
    if not 280.0 <= cylinder_diameter <= 320.0:
        raise ValueError(f"Unexpected cylinder diameter: {cylinder_diameter:.3f} mm")
    if not 60.0 <= mouse_length <= 140.0:
        raise ValueError(f"Unexpected mouse length: {mouse_length:.3f} mm")
    return cylinder_dimensions, mouse_dimensions


def ensure_depth_ignore_collection():
    collection = bpy.data.collections.get(DEPTH_IGNORE_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(DEPTH_IGNORE_COLLECTION)
        bpy.context.scene.collection.children.link(collection)
    for name in DEPTH_IGNORE_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise KeyError(f"Missing transparent depth-ignore object: {name}")
        if collection.objects.get(obj.name) is None:
            collection.objects.link(obj)
    return collection


def main():
    brown_lighting.configure_fur()
    brown_lighting.configure_lights()
    glass_balance.restore_ceiling_emission()
    for material_name, settings in glass_balance.GLASS_SETTINGS.items():
        glass_balance.configure_glass(material_name, **settings)
    collection = ensure_depth_ignore_collection()
    scene = bpy.context.scene
    cylinder_dimensions, mouse_dimensions = validate_millimeter_scale(scene)
    scene["mgf_dataset_render_preset"] = "sceneonly_moseq_core23_8view_v1"
    scene["mgf_depth_ignore_collection"] = collection.name
    scene["mgf_world_coordinate_unit"] = "millimeter"
    scene["mgf_scene_scale_to_raw_mm"] = 1000.0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print(
        "Prepared SceneOnly MoSeq Core23 render: "
        f"depth_ignore={[obj.name for obj in collection.objects]} "
        f"cylinder_mm={[round(value, 3) for value in cylinder_dimensions]} "
        f"mouse_mm={[round(value, 3) for value in mouse_dimensions]}"
    )
    print(f"Saved: {bpy.data.filepath}")


if __name__ == "__main__":
    main()

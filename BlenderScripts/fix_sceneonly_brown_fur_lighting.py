"""Apply the brown-fur and reflection-safe lighting setup to SceneOnly MoSeq.

Run with Blender, for example:
  blender --background scene.blend --python fix_sceneonly_brown_fur_lighting.py
"""

import math

import bpy
from mathutils import Vector


BROWN_MATERIAL = "MGF_BrownFur_SceneOnly"
STAGE_LIGHTS = tuple(f"MGF_StageEmitter_{index:02d}" for index in range(1, 4))
BOTTOM_LIGHTS = tuple(f"MGF_BottomFillArea_{index:02d}" for index in range(1, 4))


def set_ray_visibility(light_object):
    """Keep emitter geometry out of glass reflections and transmissions."""
    for attribute in (
        "visible_camera",
        "visible_glossy",
        "visible_transmission",
        "visible_volume_scatter",
    ):
        if hasattr(light_object, attribute):
            setattr(light_object, attribute, False)


def replace_light_data(light_object, light_type):
    """Replace the datablock because Blender 5.1 keeps type-specific RNA classes."""
    old_data = light_object.data
    if old_data.type == light_type:
        return old_data
    new_data = bpy.data.lights.new(name=f"{light_object.name}_{light_type}", type=light_type)
    light_object.data = new_data
    if old_data.users == 0:
        bpy.data.lights.remove(old_data)
    return new_data


def aim_at(light_object, target):
    light_object.rotation_euler = (Vector(target) - light_object.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()


def configure_fur():
    material = bpy.data.materials.get(BROWN_MATERIAL)
    if material is None or not material.use_nodes:
        raise RuntimeError(f"Missing node material: {BROWN_MATERIAL}")

    hair = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED"),
        None,
    )
    if hair is None:
        raise RuntimeError(f"{BROWN_MATERIAL} has no Principled Hair BSDF")

    # Linear RGB. This renders as a medium-dark, neutral brown under Filmic/AgX.
    hair.inputs["Color"].default_value = (0.18, 0.060, 0.020, 1.0)
    hair.inputs["Roughness"].default_value = 0.62
    if "Radial Roughness" in hair.inputs:
        hair.inputs["Radial Roughness"].default_value = 0.36
    if "Random Color" in hair.inputs:
        hair.inputs["Random Color"].default_value = 0.07


def configure_lights():
    # The three lights positioned at the modeled stage-light outlets.
    for name in STAGE_LIGHTS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "LIGHT":
            raise RuntimeError(f"Missing stage light object: {name}")
        light = replace_light_data(obj, "SPOT")
        light.energy = 130000.0
        light.color = (1.0, 0.98, 0.95)
        light.shadow_soft_size = 22.0
        light.spot_size = math.radians(62.0)
        light.spot_blend = 0.65
        if hasattr(light, "volume_factor"):
            light.volume_factor = 0.0
        set_ray_visibility(obj)

    # Three low, broad area fills lift the underside without flattening the fur.
    # Area lights are appropriate for this millimetre-scale scene; their emitter
    # shapes remain hidden by the object ray-visibility settings below.
    for index, name in enumerate(BOTTOM_LIGHTS):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "LIGHT":
            raise RuntimeError(f"Missing bottom fill object: {name}")
        light = replace_light_data(obj, "AREA")
        light.energy = 150000.0
        light.color = (0.96, 0.98, 1.0)
        light.shape = "DISK"
        light.size = 65.0
        angle = math.radians(90.0 + index * 120.0)
        obj.location = (120.0 * math.cos(angle), 120.0 * math.sin(angle), 360.0)
        aim_at(obj, (0.0, 0.0, 370.0))
        if hasattr(light, "volume_factor"):
            light.volume_factor = 0.0
        set_ray_visibility(obj)

    # Existing classroom lights were producing large white ellipses through the
    # glass cylinder. Hide only the emitter from camera/glossy/transmission rays;
    # diffuse illumination and shadows remain enabled.
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            set_ray_visibility(obj)


def main():
    configure_fur()
    configure_lights()
    scene = bpy.context.scene
    scene["mgf_brown_lighting_revision"] = 9
    scene["mgf_brown_lighting_notes"] = (
        "Brown hair; 3 modeled-outlet spots; 3 bottom area fills; "
        "light emitters hidden from camera/glossy/transmission rays"
    )
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print(f"Saved reflection-safe brown-fur lighting: {bpy.data.filepath}")


if __name__ == "__main__":
    main()

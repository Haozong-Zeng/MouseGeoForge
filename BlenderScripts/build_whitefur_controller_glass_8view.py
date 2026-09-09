#!/usr/bin/env python3
"""Build a glass-platform, bounded-depth, canonical 8-view WhiteFur scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


CAMERA_NAMES = tuple(f"Camera.{index:03d}" for index in range(1, 9))
CAMERA_ROLES = (
    "horizontal_front",
    "horizontal_back_left",
    "horizontal_back_right",
    "elevated_front",
    "elevated_back_right",
    "elevated_back_left",
    "bottom",
    "top",
)
RIG_COLLECTION_NAMES = {
    1: "MGF_Rig_Camera.001_Horizontal_Front",
    2: "MGF_Rig_Camera.002_Horizontal_BackLeft",
    3: "MGF_Rig_Camera.003_Horizontal_BackRight",
    4: "MGF_Rig_Camera.004_Horizontal_Back",
    5: "MGF_Rig_Camera.005_Horizontal_FrontRight",
    6: "MGF_Rig_Camera.006_Horizontal_FrontLeft",
    7: "MGF_Rig_Camera.007_Bottom",
}
RIG_ANGLE_DEGREES = {1: 90.0, 2: -150.0, 3: -30.0, 4: -90.0, 5: 30.0, 6: 150.0}
ENVIRONMENT_NAMES = ("Part_GlassCylinder160mm", "Building_Floor")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args_after_dash())


def load_donor_objects(donor_path: Path, names: tuple[str, ...]) -> dict[str, bpy.types.Object]:
    with bpy.data.libraries.load(str(donor_path), link=False) as (data_from, data_to):
        missing = [name for name in names if name not in data_from.objects]
        if missing:
            raise KeyError(f"Donor objects missing: {missing}")
        data_to.objects = list(names)
    return dict(zip(names, data_to.objects, strict=True))


def replace_object_data_and_transform(target: bpy.types.Object, donor: bpy.types.Object) -> None:
    old_data = target.data
    target.data = donor.data.copy()
    target.matrix_basis = donor.matrix_basis.copy()
    target.animation_data_clear()
    for constraint in list(target.constraints):
        target.constraints.remove(constraint)
    if old_data and old_data.users == 0:
        bpy.data.meshes.remove(old_data)


def make_glass_material(
    name: str,
    tint: tuple[float, float, float, float],
    roughness: float,
    glass_fraction: float,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (*tint[:3], 0.3)
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"

    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (420, 0)
    camera_mix = tree.nodes.new("ShaderNodeMixShader")
    camera_mix.location = (80, -120)
    camera_mix.inputs[0].default_value = glass_fraction
    ray_mix = tree.nodes.new("ShaderNodeMixShader")
    ray_mix.location = (250, 0)
    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.location = (-120, 180)
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-120, 80)
    transparent.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    glass = tree.nodes.new("ShaderNodeBsdfGlass")
    glass.location = (-120, -100)
    glass.inputs[0].default_value = tint
    glass.inputs[1].default_value = roughness
    glass.inputs[2].default_value = 1.45
    tree.links.new(transparent.outputs[0], camera_mix.inputs[1])
    tree.links.new(glass.outputs[0], camera_mix.inputs[2])
    tree.links.new(light_path.outputs["Is Camera Ray"], ray_mix.inputs[0])
    tree.links.new(glass.outputs[0], ray_mix.inputs[1])
    tree.links.new(camera_mix.outputs[0], ray_mix.inputs[2])
    tree.links.new(ray_mix.outputs[0], output.inputs[0])
    return material


def make_viewport_glass_material(name: str) -> bpy.types.Material:
    """Create glass that stays transparent in Material Preview as well as Cycles."""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (0.55, 0.78, 0.92, 0.12)
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    if hasattr(material, "use_transparency_overlap"):
        material.use_transparency_overlap = False

    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    mix = tree.nodes.new("ShaderNodeMixShader")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    glass = tree.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs[0].default_value = (0.82, 0.94, 1.0, 1.0)
    glass.inputs[1].default_value = 0.035
    glass.inputs[2].default_value = 1.45
    mix.inputs[0].default_value = 0.16
    tree.links.new(transparent.outputs[0], mix.inputs[1])
    tree.links.new(glass.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], output.inputs[0])
    return material


def make_matte_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.82
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Alpha"].default_value = 1.0
    return material


def assign_single_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.hide_render = False
    obj.hide_viewport = False
    obj.hide_set(False)


def correct_support_glass_cylinder_stack(scene: bpy.types.Scene) -> dict[str, float | int]:
    def z_bounds(obj: bpy.types.Object) -> tuple[float, float]:
        values = [(obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box]
        return min(values), max(values)

    support = bpy.data.objects["Part_CNC-10mmGlassSupport"]
    glass_objects = [
        bpy.data.objects["Part_AntiReflectiveGlass"],
        bpy.data.objects["Part_AntiReflectiveGlass_Circle"],
    ]
    cylinder = bpy.data.objects["Part_GlassCylinder160mm"]

    support_top = z_bounds(support)[1]
    glass_bottom = min(z_bounds(obj)[0] for obj in glass_objects)
    glass_delta = support_top - glass_bottom
    for obj in glass_objects:
        obj.location.z += glass_delta
        obj["MGF_stack_role"] = "glass_on_support"
        obj["MGF_stack_support"] = support.name

    bpy.context.view_layer.update()
    glass_top = max(z_bounds(obj)[1] for obj in glass_objects)
    cylinder_delta = glass_top - z_bounds(cylinder)[0]
    cylinder.location.z += cylinder_delta
    cylinder["MGF_stack_role"] = "cylinder_on_glass"
    cylinder["MGF_stack_support"] = "Part_AntiReflectiveGlass_Circle"

    armature = bpy.data.objects.get("DeformationSystem.001")
    targets: set[bpy.types.Object] = set()
    if armature is not None:
        for constraint in armature.constraints:
            target = getattr(constraint, "target", None)
            if target is not None and target != armature:
                targets.add(target)
        for pose_bone in armature.pose.bones:
            for constraint in pose_bone.constraints:
                target = getattr(constraint, "target", None)
                if target is not None and target != armature:
                    targets.add(target)
    for target in targets:
        target.delta_location.z += glass_delta
        target["MGF_glass_stack_offset_z"] = float(target.get("MGF_glass_stack_offset_z", 0.0)) + glass_delta

    scene["MGF_physical_stack"] = "support -> coplanar glass plates -> cylinder and mouse"
    scene["MGF_glass_stack_delta_z"] = float(glass_delta)
    scene["MGF_cylinder_stack_delta_z"] = float(cylinder_delta)
    scene["MGF_mouse_stack_control_targets"] = len(targets)
    return {
        "support_top_z": support_top,
        "glass_bottom_z": support_top,
        "glass_top_z": glass_top,
        "cylinder_bottom_z": glass_top,
        "glass_delta_z": glass_delta,
        "cylinder_delta_z": cylinder_delta,
        "mouse_control_targets": len(targets),
    }


def install_render_helper() -> None:
    source = '''from pathlib import Path
import bpy

CAMERAS = [f"Camera.{index:03d}" for index in range(1, 9)]
SELF_RIG_COLLECTIONS = {
    "Camera.001": "MGF_Rig_Camera.001_Horizontal_Front",
    "Camera.002": "MGF_Rig_Camera.002_Horizontal_BackLeft",
    "Camera.003": "MGF_Rig_Camera.003_Horizontal_BackRight",
    "Camera.004": "MGF_Rig_Camera.004_Horizontal_Back",
    "Camera.005": "MGF_Rig_Camera.005_Horizontal_FrontRight",
    "Camera.006": "MGF_Rig_Camera.006_Horizontal_FrontLeft",
    "Camera.007": "MGF_Rig_Camera.007_Bottom",
    "Camera.008": None,
}
scene = bpy.context.scene
output_root = Path(bpy.path.abspath("//multiview_render")) / f"frame_{scene.frame_current:06d}"
output_root.mkdir(parents=True, exist_ok=True)

if hasattr(scene, "node_tree"):
    scene.use_nodes = True
    tree = scene.node_tree
else:
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new(f"{scene.name}_MGF_Compositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
render_layers = tree.nodes.get("MGF_Render8ViewsLayers") or tree.nodes.new("CompositorNodeRLayers")
render_layers.name = "MGF_Render8ViewsLayers"
output = tree.nodes.get("MGF_Render8ViewsOutput") or tree.nodes.new("CompositorNodeOutputFile")
output.name = "MGF_Render8ViewsOutput"
for node in tree.nodes:
    if node.type == "OUTPUT_FILE" and node != output:
        node.mute = True
output.mute = False
if hasattr(output, "base_path"):
    output.base_path = str(output_root)
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    image_input = output.inputs[0]
else:
    output.directory = str(output_root)
    output.file_output_items.clear()
    item = output.file_output_items.new(socket_type="RGBA", name="Image")
    item.override_node_format = True
    item.format.file_format = "PNG"
    item.format.color_mode = "RGBA"
    image_input = output.inputs["Image"]
for link in list(tree.links):
    if link.to_node == output:
        tree.links.remove(link)
tree.links.new(render_layers.outputs["Image"], image_input)
if hasattr(scene.render, "use_compositing"):
    scene.render.use_compositing = True

for view_index, camera_name in enumerate(CAMERAS):
    stem = f"view_{view_index:02d}_{camera_name}"
    scene.camera = bpy.data.objects[camera_name]
    if hasattr(output, "base_path"):
        output.file_slots[0].path = stem
    else:
        output.file_name = stem
    collection_name = SELF_RIG_COLLECTIONS[camera_name]
    collection = bpy.data.collections.get(collection_name) if collection_name else None
    previous = collection.hide_render if collection else None
    try:
        if collection:
            collection.hide_render = True
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
    finally:
        if collection:
            collection.hide_render = previous
        bpy.context.view_layer.update()
print(f"Rendered 8 views to {output_root}")
'''
    text = bpy.data.texts.get("MGF_Render8Views.py") or bpy.data.texts.new("MGF_Render8Views.py")
    text.clear()
    text.write(source)


def ensure_composite_output(scene: bpy.types.Scene) -> None:
    if not hasattr(bpy.types, "CompositorNodeComposite"):
        if hasattr(scene.render, "use_compositing"):
            scene.render.use_compositing = False
        return
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        tree = scene.node_tree
    else:
        tree = getattr(scene, "compositing_node_group", None)
        if tree is None:
            tree = bpy.data.node_groups.new(f"{scene.name}_MGF_Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
    render_layers = next((node for node in tree.nodes if node.type == "R_LAYERS"), None)
    if render_layers is None:
        render_layers = tree.nodes.new("CompositorNodeRLayers")
        render_layers.name = "MGF_RenderLayers"
    composite = next((node for node in tree.nodes if node.type == "COMPOSITE"), None)
    if composite is None:
        composite = tree.nodes.new("CompositorNodeComposite")
        composite.name = "MGF_Composite"
    for link in list(tree.links):
        if link.to_node == composite:
            tree.links.remove(link)
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True


def find_layer_collection(layer_collection, name: str):
    if layer_collection.name == name:
        return layer_collection
    for child in layer_collection.children:
        found = find_layer_collection(child, name)
        if found is not None:
            return found
    return None


def configure_render_visibility_and_lighting(scene: bpy.types.Scene) -> None:
    visible_collections = ("Collection", "Mouse_White", "Devices", "MGF_8View_Cameras")
    hidden_render_collections = (
        "Mouse_Brown",
        "LED.001",
        "LED.002",
        "LED.003",
        "PseudoKeypoints.001",
        "PseudoKeypoints.002",
    )
    for name in visible_collections:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_render = False
            collection.hide_viewport = False
        layer_collection = find_layer_collection(bpy.context.view_layer.layer_collection, name)
        if layer_collection is not None:
            layer_collection.exclude = False
            layer_collection.hide_viewport = False
    for name in hidden_render_collections:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_render = True

    # Preserve the complete physical rig. Only deprecated duplicates and Moseq
    # background geometry are suppressed.
    environment_collection = bpy.data.collections.get("Collection")
    if environment_collection is not None:
        for obj in environment_collection.objects:
            if obj.type == "MESH" and obj.name != "Part_Cylinder":
                obj.hide_render = False
                obj.hide_viewport = False
                obj.hide_set(False)
    for name in ("background_average", "moseq_background", "black_tube", "Part_Cylinder"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.hide_set(True)

    sun = bpy.data.objects.get("日光")
    if sun is not None and sun.type == "LIGHT":
        sun.hide_render = False
        sun.hide_viewport = False
        sun.hide_set(False)
        sun.data.energy = 3.0
        sun.data.angle = 0.12

    bottom_fill = bpy.data.objects.get("MGF_BottomFill")
    if bottom_fill is None:
        bottom_fill_data = bpy.data.lights.new("MGF_BottomFill", type="SUN")
        bottom_fill = bpy.data.objects.new("MGF_BottomFill", bottom_fill_data)
        (bpy.data.collections.get("Collection") or scene.collection).objects.link(bottom_fill)
    bottom_fill.rotation_euler = (math.pi, 0.0, 0.0)
    bottom_fill.data.energy = 1.2
    bottom_fill.data.angle = 0.2
    bottom_fill.hide_render = False
    bottom_fill.hide_viewport = False
    bottom_fill.hide_set(False)

    scene.render.film_transparent = False
    if scene.world and scene.world.use_nodes and scene.world.node_tree:
        background = next((node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND"), None)
        if background is not None:
            background.inputs[0].default_value = (0.055, 0.055, 0.055, 1.0)
            background.inputs[1].default_value = 0.35


def configure_viewport_and_cameras() -> None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            overlay = area.spaces.active.overlay
            overlay.show_overlays = True
            overlay.show_outline_selected = True
            overlay.show_extras = True
            overlay.show_relationship_lines = False
            overlay.show_text = True
    for camera_name in CAMERA_NAMES:
        camera = bpy.data.objects[camera_name]
        camera.hide_render = False
        camera.hide_viewport = False
        camera.hide_set(False)
        camera.show_in_front = True
        camera.display_type = "WIRE"
        camera.data.display_size = 5.0
        camera.data.show_name = True
        camera.data.show_limits = False


def organize_camera_mechanics(scene: bpy.types.Scene) -> dict[str, int]:
    root = bpy.data.collections.get("MGF_Camera_Mechanics")
    if root is None:
        root = bpy.data.collections.new("MGF_Camera_Mechanics")
    if not any(child == root for child in scene.collection.children):
        scene.collection.children.link(root)
    root.hide_render = False
    root.hide_viewport = False

    groups: dict[int, bpy.types.Collection] = {}
    for index, name in RIG_COLLECTION_NAMES.items():
        collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if not any(child == collection for child in root.children):
            for parent in bpy.data.collections:
                if parent != root and any(child == collection for child in parent.children):
                    parent.children.unlink(collection)
            if any(child == collection for child in scene.collection.children):
                scene.collection.children.unlink(collection)
            root.children.link(collection)
        collection.hide_render = False
        collection.hide_viewport = False
        collection["MGF_camera_name"] = f"Camera.{index:03d}"
        collection["MGF_role"] = "camera_specific_mechanical_rig"
        groups[index] = collection

    def object_center(obj: bpy.types.Object) -> Vector:
        if obj.type == "MESH":
            return obj.matrix_world @ (sum((Vector(vertex) for vertex in obj.bound_box), Vector()) / 8.0)
        return obj.matrix_world.translation.copy()

    def circular_distance(first: float, second: float) -> float:
        return abs((first - second + 180.0) % 360.0 - 180.0)

    def nearest_horizontal(point: Vector) -> int:
        angle = math.degrees(math.atan2(point.y, point.x))
        return min(
            RIG_ANGLE_DEGREES,
            key=lambda index: circular_distance(angle, RIG_ANGLE_DEGREES[index]),
        )

    assigned: set[str] = set()
    group_members: dict[int, list[str]] = {index: [] for index in groups}

    def assign(obj: bpy.types.Object, index: int) -> None:
        if obj.name in assigned:
            return
        target = groups[index]
        if target.objects.get(obj.name) is None:
            target.objects.link(obj)
        for old_collection in list(obj.users_collection):
            if old_collection != target:
                old_collection.objects.unlink(obj)
        obj["MGF_mechanical_camera_collection"] = target.name
        assigned.add(obj.name)
        group_members[index].append(obj.name)

    main_collection = bpy.data.collections.get("Collection")
    side_tokens = (
        "part_4inchopticpost",
        "part_8inchopticpost",
        "part_90degreeclamp",
        "part_picamside",
        "part_sidecamscrew",
        "rotatableopticclamp",
        "imx296 dual support",
    )
    bottom_names = {
        "Part_BottomCamScrew.001",
        "Part_Breadboard",
        "Part_OpticStand",
        "Part_PiCamBottom",
        "Gigabit_Network_Switch_Black.001",
        "Network.001",
        "Metal.001",
        "Light.001",
        "Label.001",
        "Base.001",
    }
    if main_collection is not None:
        for obj in list(main_collection.objects):
            lower_name = obj.name.lower()
            if obj.name.startswith("Part_12inchOpticPost") or obj.name in bottom_names:
                assign(obj, 7)
            elif any(token in lower_name for token in side_tokens):
                assign(obj, nearest_horizontal(object_center(obj)))

    devices = bpy.data.collections.get("Devices")
    source_names = {
        "Arduino",
        "Raspberry_Pi4.001",
        "Raspberry_Pi4.002",
        "Raspberry_Pi4.003",
        "Raspberry_Pi4.004",
        "EthernetCable.001",
        "EthernetCable.002",
        "EthernetCable.007",
        "DualCameras",
    }
    sources = [bpy.data.collections.get(name) for name in source_names]
    sources = [collection for collection in sources if collection is not None]
    if devices is not None:
        for obj in list(devices.objects):
            assign(obj, 7)

    for collection in sources:
        for obj in list(collection.objects):
            if obj.name in assigned:
                continue
            if collection.name in {"Arduino", "Raspberry_Pi4.001", "EthernetCable.007"}:
                index = 7
            elif collection.name in {"EthernetCable.001", "EthernetCable.002"}:
                index = 1
            elif collection.name == "DualCameras":
                if obj.name == "DualCam.002" or (obj.parent and obj.parent.name == "DualCam.002"):
                    index = 2
                elif obj.name == "DualCam.001" or (obj.parent and obj.parent.name == "DualCam.001"):
                    index = 3
                else:
                    index = nearest_horizontal(object_center(obj))
            else:
                point = object_center(obj)
                radius = math.hypot(point.x, point.y)
                index = 7 if radius < 180.0 or point.z < 180.0 else nearest_horizontal(point)
            assign(obj, index)

    for name in source_names:
        collection = bpy.data.collections.get(name)
        if collection and len(collection.objects) == 0 and len(collection.children) == 0:
            bpy.data.collections.remove(collection)
    devices = bpy.data.collections.get("Devices")
    if devices and len(devices.objects) == 0 and len(devices.children) == 0:
        bpy.data.collections.remove(devices)

    root["MGF_grouping_method"] = "azimuth around arena; central and low electronics assigned to Bottom"
    root["MGF_group_count"] = 7
    scene["MGF_camera_mechanics_collections"] = ";".join(RIG_COLLECTION_NAMES.values())
    scene["MGF_render_rig_policy"] = "Hide the active camera rig collection while rendering"

    # This upper connector shell exists in the source file but is hidden,
    # leaving one of the two front Ethernet plugs visibly incomplete.
    ethernet_upper = bpy.data.objects.get("ethernet_assem(Step_ap203).017")
    if ethernet_upper is not None:
        ethernet_upper.hide_viewport = False
        ethernet_upper.hide_set(False)
        ethernet_upper.hide_render = False
        ethernet_upper["MGF_visibility_fix"] = (
            "Restored upper counterpart of ethernet_assem(Step_ap203).006"
        )
    return {RIG_COLLECTION_NAMES[index]: len(groups[index].all_objects) for index in groups}


def rounded_matrix(matrix) -> list[list[float]]:
    return [[round(float(value), 8) for value in row] for row in matrix]


def main() -> None:
    args = parse_args()
    source_path = Path(bpy.data.filepath).resolve()
    donor_path = args.donor.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = CAMERA_NAMES + ENVIRONMENT_NAMES
    donors = load_donor_objects(donor_path, names)

    camera_collection = bpy.data.collections.get("MGF_8View_Cameras")
    if camera_collection is None:
        camera_collection = bpy.data.collections.new("MGF_8View_Cameras")
        bpy.context.scene.collection.children.link(camera_collection)

    for view_index, (camera_name, role) in enumerate(zip(CAMERA_NAMES, CAMERA_ROLES, strict=True)):
        target = bpy.data.objects.get(camera_name)
        if target is None or target.type != "CAMERA":
            raise KeyError(f"Target camera missing: {camera_name}")
        donor = donors[camera_name]
        old_data = target.data
        target.data = donor.data.copy()
        target.matrix_basis = donor.matrix_basis.copy()
        target.animation_data_clear()
        for constraint in list(target.constraints):
            target.constraints.remove(constraint)
        if old_data and old_data.users == 0:
            bpy.data.cameras.remove(old_data)
        if camera_collection.objects.get(target.name) is None:
            camera_collection.objects.link(target)
        target["mgf_view_index"] = view_index
        target["mgf_view_role"] = role
        target["mgf_camera_donor"] = str(donor_path)

    bpy.data.objects["Camera.007"].data.lens = 16.0
    bpy.data.objects["Camera.007"]["mgf_scene_adaptation"] = "wider bottom view retained from WhiteFur source"
    bpy.data.objects["Camera.008"].location.z = 780.5519409179688
    bpy.data.objects["Camera.008"]["mgf_scene_adaptation"] = "higher top view retained from WhiteFur source"

    cylinder = bpy.data.objects.get("Part_GlassCylinder160mm")
    floor = bpy.data.objects.get("Building_Floor")
    if cylinder is None or floor is None:
        raise KeyError("Target Part_GlassCylinder160mm or Building_Floor is missing")
    replace_object_data_and_transform(cylinder, donors["Part_GlassCylinder160mm"])
    replace_object_data_and_transform(floor, donors["Building_Floor"])

    cylinder_material = make_glass_material(
        "MGF_Glass_Cylinder_LowOcclusion",
        (0.82, 0.96, 1.0, 1.0),
        roughness=0.045,
        glass_fraction=0.0,
    )
    floor_material = make_matte_material("MGF_DepthFloor_Matte", (0.16, 0.18, 0.20, 1.0))
    assign_single_material(cylinder, cylinder_material)
    assign_single_material(floor, floor_material)
    floor.scale.x *= 2600.0 / floor.dimensions.x
    floor.scale.y *= 2600.0 / floor.dimensions.y
    floor["MGF_role"] = "opaque_depth_floor"
    floor["MGF_depth_cap_dimensions_xy"] = [2600.0, 2600.0]

    platform_material = make_viewport_glass_material("MGF_Glass_MousePlatform_Transparent")
    for name in ("Part_AntiReflectiveGlass", "Part_AntiReflectiveGlass_Circle"):
        platform = bpy.data.objects.get(name)
        if platform is None:
            continue
        assign_single_material(platform, platform_material)
        platform.display_type = "WIRE"
        platform.show_wire = True
        platform.show_all_edges = True
        platform["MGF_role"] = "transparent_mouse_support_platform"

    cylinder.display_type = "WIRE"
    cylinder.show_wire = True
    cylinder.show_all_edges = True

    ceiling = bpy.data.objects.get("Part_TopCover")
    if ceiling is not None:
        ceiling["MGF_original_z"] = float(ceiling.get("MGF_original_z", ceiling.location.z))
        ceiling.location.z = 900.0
        ceiling.scale.x *= 2600.0 / ceiling.dimensions.x
        ceiling.scale.y *= 2600.0 / ceiling.dimensions.y
        assign_single_material(
            ceiling,
            make_matte_material("MGF_DepthCeiling_Matte", (0.55, 0.57, 0.60, 1.0)),
        )
        ceiling["MGF_role"] = "opaque_depth_ceiling_above_top_camera"
        ceiling["MGF_depth_cap_dimensions_xy"] = [2600.0, 2600.0]

    opaque_cylinder = bpy.data.objects.get("Part_Cylinder")
    if opaque_cylinder is not None:
        opaque_cylinder.hide_render = True
        opaque_cylinder.hide_viewport = True

    scene = bpy.context.scene
    scene.camera = bpy.data.objects["Camera.008"]
    scene["mgf_multiview_camera_names"] = json.dumps(CAMERA_NAMES)
    scene["mgf_multiview_camera_roles"] = json.dumps(CAMERA_ROLES)
    scene["mgf_multiview_donor"] = str(donor_path)
    scene["mgf_glass_environment"] = True
    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = max(scene.cycles.transparent_max_bounces, 16)
        scene.cycles.transmission_bounces = max(scene.cycles.transmission_bounces, 12)

    stack_report = correct_support_glass_cylinder_stack(scene)
    rig_counts = organize_camera_mechanics(scene)
    configure_render_visibility_and_lighting(scene)
    configure_viewport_and_cameras()
    ensure_composite_output(scene)
    install_render_helper()
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), check_existing=False)

    manifest = {
        "format": "mousegeoforge_whitefur_controller_glass_8view_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(source_path),
        "donor_blend": str(donor_path),
        "output_blend": str(output_path),
        "active_camera": scene.camera.name,
        "cameras": [
            {
                "name": name,
                "view_index": index,
                "role": CAMERA_ROLES[index],
                "lens_mm": float(bpy.data.objects[name].data.lens),
                "sensor_width_mm": float(bpy.data.objects[name].data.sensor_width),
                "matrix_world": rounded_matrix(bpy.data.objects[name].matrix_world),
            }
            for index, name in enumerate(CAMERA_NAMES)
        ],
        "glass_objects": {
            cylinder.name: {
                "material": cylinder_material.name,
                "dimensions": [float(value) for value in cylinder.dimensions],
            },
            floor.name: {
                "material": floor_material.name,
                "dimensions": [float(value) for value in floor.dimensions],
            },
        },
        "embedded_render_helper": "MGF_Render8Views.py",
        "camera_mechanics_collections": rig_counts,
        "physical_stack": stack_report,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

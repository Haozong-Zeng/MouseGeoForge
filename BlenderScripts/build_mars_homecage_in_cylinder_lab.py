#!/usr/bin/env python3
"""Build a MARS-homecage multiview project inside the Cylinder laboratory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from math import radians
from mathutils import Matrix, Vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DONOR = PROJECT_ROOT / "backup" / "blender" / "MARS homecage.blend"
DEFAULT_OUTPUT = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Homecage\PyMouseStudio-MARSHomecage-CylinderLab-Multiview.blend"
)
DONOR_SCENE = "Scene"
CAGE_OBJECT_NAME = "MGF_MARS_HomecageShell"
OLD_ACTIVITY_PREFIXES = (
    "Part_GlassCylinder160mm",
    "Part_AntiReflectiveGlass",
    "Part_Cylinder",
    "Part_CNC-10mmGlassSupport",
)
OLD_MOUSE_COLLECTIONS = (
    "Mouse_Brown",
    "Mouse_White",
    "moseq",
    "PseudoKeypoints.001",
    "PseudoKeypoints.002",
    "MGF_FullTrajectory_Controls",
)
OLD_CAMERA_COLLECTIONS = (
    "MGF_8View_Cameras",
    "MGF_Active_8View_Cameras",
    "MGF_BottomView_ThreePointFill",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--float-z",
        type=float,
        default=None,
        help="Explicit cage-bottom Z in mm; defaults to the detected tabletop.",
    )
    parser.add_argument("--table-clearance", type=float, default=0.5)
    parser.add_argument(
        "--material-profile",
        choices=("capture_clear", "physical", "opaque_debug"),
        default="capture_clear",
    )
    return parser.parse_args(args_after_dash())


def remove_objects(objects: set[bpy.types.Object]) -> int:
    count = 0
    for obj in list(objects):
        if obj and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
            count += 1
    return count


def hierarchy(root: bpy.types.Object) -> set[bpy.types.Object]:
    result = {root}
    pending = list(root.children)
    while pending:
        obj = pending.pop()
        if obj in result:
            continue
        result.add(obj)
        pending.extend(obj.children)
    return result


def remove_collection_and_objects(name: str) -> int:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return 0
    objects = set(collection.all_objects)
    removed = remove_objects(objects)
    if name in bpy.data.collections:
        bpy.data.collections.remove(collection)
    return removed


def clear_previous_activity(scene: bpy.types.Scene) -> dict[str, object]:
    removed_names: list[str] = []
    source_root = bpy.data.objects.get("MGF_SourcePose_PerFramePlacement")
    if source_root is not None:
        removed_names.extend(obj.name for obj in hierarchy(source_root))
        remove_objects(hierarchy(source_root))

    for collection_name in (*OLD_MOUSE_COLLECTIONS, *OLD_CAMERA_COLLECTIONS):
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            removed_names.extend(obj.name for obj in collection.all_objects)
            remove_collection_and_objects(collection_name)

    for obj in list(bpy.data.objects):
        if any(obj.name.startswith(prefix) for prefix in OLD_ACTIVITY_PREFIXES):
            removed_names.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)

    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            removed_names.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)

    if scene.camera and scene.camera.name not in bpy.data.objects:
        scene.camera = None
    return {
        "removed_count": len(set(removed_names)),
        "removed_activity": sorted(
            name for name in set(removed_names) if name.startswith(OLD_ACTIVITY_PREFIXES)
        ),
    }


def remove_cylinder_camera_rig() -> dict[str, object]:
    rig_collection = next(
        (
            collection
            for collection in bpy.data.collections
            if any(obj.name.startswith("Part_Breadboard") for obj in collection.objects)
            and any(obj.name.startswith("Part_PiCamSide") for obj in collection.objects)
        ),
        None,
    )
    if rig_collection is None:
        return {"collection": None, "removed_count": 0, "removed_names": []}
    collection_name = rig_collection.name
    removed_names = sorted(obj.name for obj in rig_collection.all_objects)
    remove_collection_and_objects(collection_name)
    return {
        "collection": collection_name,
        "removed_count": len(removed_names),
        "removed_names": removed_names,
    }


def object_world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def find_tabletop() -> tuple[bpy.types.Object, float]:
    bpy.context.view_layer.update()
    candidates: list[tuple[float, bpy.types.Object]] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        materials = {
            slot.material.name.lower()
            for slot in obj.material_slots
            if slot.material is not None
        }
        if not any("table" in name for name in materials):
            continue
        minimum, maximum = object_world_bounds(obj)
        if minimum.x <= 0.0 <= maximum.x and minimum.y <= 0.0 <= maximum.y:
            if maximum.x - minimum.x > 400.0 and maximum.y - minimum.y > 400.0:
                candidates.append((float(maximum.z), obj))
    if not candidates:
        raise RuntimeError("Could not find a tabletop below the homecage center")
    tabletop_z, tabletop = max(candidates, key=lambda record: record[0])
    return tabletop, tabletop_z


def append_donor_collections(
    scene: bpy.types.Scene,
    donor: Path,
) -> dict[str, bpy.types.Collection]:
    scenes_before = set(bpy.data.scenes)
    with bpy.data.libraries.load(str(donor), link=False) as (available, requested):
        if DONOR_SCENE not in available.scenes:
            raise KeyError(f"MARS donor is missing {DONOR_SCENE}")
        requested.scenes = [DONOR_SCENE]
    donor_scene = next(
        appended_scene
        for appended_scene in bpy.data.scenes
        if appended_scene not in scenes_before
    )
    donor_root_children = list(donor_scene.collection.children)
    cage = next(
        (
            collection
            for collection in donor_root_children
            if any(
                obj.type == "MESH"
                and any(
                    slot.material and slot.material.name.startswith("有瑕疵的逼真玻璃")
                    for slot in obj.material_slots
                )
                for obj in collection.objects
            )
        ),
        None,
    )
    scene_parts = next(
        (
            collection
            for collection in donor_root_children
            if any(
                obj.type == "ARMATURE"
                for child in collection.children
                for obj in child.all_objects
            )
        ),
        None,
    )
    if cage is None or scene_parts is None:
        raise KeyError("Could not identify the MARS cage and mouse root collections")

    loaded = {"cage": cage, "scene_parts": scene_parts}
    for collection in loaded.values():
        scene.collection.children.link(collection)
    bpy.data.scenes.remove(donor_scene)
    return loaded


def new_child_collection(parent: bpy.types.Collection, name: str) -> bpy.types.Collection:
    old = bpy.data.collections.get(name)
    if old is not None:
        remove_collection_and_objects(name)
    collection = bpy.data.collections.new(name)
    parent.children.link(collection)
    return collection


def unlink_object_from_all_collections(obj: bpy.types.Object) -> None:
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)


def move_object_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    unlink_object_from_all_collections(obj)
    collection.objects.link(obj)


def unlink_root_collection(scene: bpy.types.Scene, collection: bpy.types.Collection) -> None:
    if collection.name in scene.collection.children:
        scene.collection.children.unlink(collection)


def organize_homecage_collections(
    scene: bpy.types.Scene,
    loaded: dict[str, bpy.types.Collection],
) -> tuple[bpy.types.Collection, dict[str, bpy.types.Collection], set[bpy.types.Object]]:
    root = bpy.data.collections.new("MGF_MARS_Homecage")
    scene.collection.children.link(root)
    shell = new_child_collection(root, "MGF_Homecage_Shell")
    inlet = new_child_collection(root, "MGF_Homecage_Inlet")
    interior = new_child_collection(root, "MGF_Homecage_Interior")

    cage_parts = loaded["cage"]
    bpy.context.view_layer.update()
    imported_objects: set[bpy.types.Object] = set()
    cage_meshes = [obj for obj in cage_parts.objects if obj.type == "MESH"]
    shell_object = max(cage_meshes, key=lambda obj: float(obj.dimensions.z))
    shell_origin = shell_object.matrix_world.translation.copy()
    shell_object.name = CAGE_OBJECT_NAME
    for obj in list(cage_parts.objects):
        if obj == shell_object:
            move_object_to_collection(obj, shell)
            imported_objects.add(obj)
        elif (
            obj.type == "MESH"
            and float(obj.dimensions.x) > 0.25
            and float(obj.dimensions.y) > 0.19
            and float(obj.dimensions.z) < 0.001
            and abs(float(obj.matrix_world.translation.x - shell_origin.x)) < 0.05
            and abs(float(obj.matrix_world.translation.y - shell_origin.y)) < 0.05
        ):
            obj.name = f"MGF_MARS_HomecageFloor_{len(interior.objects):02d}"
            move_object_to_collection(obj, interior)
            imported_objects.add(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
    unlink_root_collection(scene, cage_parts)
    if cage_parts.name in bpy.data.collections:
        bpy.data.collections.remove(cage_parts)

    scene_parts = loaded["scene_parts"]
    bpy.context.view_layer.update()
    inlet_index = 0
    for obj in list(scene_parts.objects):
        center_x = float(obj.matrix_world.translation.x)
        if (
            obj.type == "MESH"
            and max(float(value) for value in obj.dimensions) < 0.10
            and 0.10 < center_x < 0.20
        ):
            obj.name = f"MGF_MARS_Inlet_{inlet_index:02d}"
            inlet_index += 1
            move_object_to_collection(obj, inlet)
            imported_objects.add(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
    if inlet_index < 2:
        raise RuntimeError(f"Expected the MARS side inlet, found {inlet_index} parts")

    mouse_candidates = sorted(
        (
            child
            for child in scene_parts.children
            if any(obj.type == "ARMATURE" for obj in child.all_objects)
        ),
        key=lambda collection: collection.name,
    )
    if not mouse_candidates:
        raise KeyError("MARS Scene has no Mouse_Black child collection")
    mouse = mouse_candidates[0]
    scene_parts.children.unlink(mouse)
    mouse.name = "MGF_MARS_Mouse"
    root.children.link(mouse)
    imported_objects.update(mouse.all_objects)
    for unused_mouse in mouse_candidates[1:]:
        remove_collection_and_objects(unused_mouse.name)
    unlink_root_collection(scene, scene_parts)
    if scene_parts.name in bpy.data.collections:
        bpy.data.collections.remove(scene_parts)

    categories = {
        "shell": shell,
        "inlet": inlet,
        "interior": interior,
        "mouse": mouse,
    }
    return root, categories, imported_objects


def place_imported_homecage(
    scene: bpy.types.Scene,
    objects: set[bpy.types.Object],
    float_z: float,
) -> bpy.types.Object:
    cage = bpy.data.objects.get(CAGE_OBJECT_NAME)
    if cage is None:
        raise KeyError(CAGE_OBJECT_NAME)
    bpy.context.view_layer.update()
    source_width = float(cage.dimensions.x)
    if source_width <= 0:
        raise RuntimeError("Homecage has invalid source width")
    scale_factor = 1000.0
    cage_corners = [cage.matrix_world @ Vector(corner) for corner in cage.bound_box]
    cage_min = Vector(
        tuple(min(point[axis] for point in cage_corners) for axis in range(3))
    )
    cage_max = Vector(
        tuple(max(point[axis] for point in cage_corners) for axis in range(3))
    )
    source_pivot = Vector(
        (
            0.5 * (cage_min.x + cage_max.x),
            0.5 * (cage_min.y + cage_max.y),
            cage_min.z,
        )
    )

    mouse_mesh = next(
        (
            obj
            for obj in objects
            if obj.type == "MESH" and obj.name.startswith("MS_Mouse")
        ),
        None,
    )
    if mouse_mesh is None:
        raise RuntimeError("No MARS mouse mesh was imported")
    mouse_corners = [mouse_mesh.matrix_world @ Vector(corner) for corner in mouse_mesh.bound_box]
    mouse_center = Vector(
        tuple(
            0.5
            * (
                min(point[axis] for point in mouse_corners)
                + max(point[axis] for point in mouse_corners)
            )
            for axis in range(3)
        )
    )
    mouse_xy_alignment = Vector(
        (source_pivot.x - mouse_center.x, source_pivot.y - mouse_center.y, 0.0)
    )
    mouse_root = mouse_mesh
    while mouse_root.parent is not None:
        mouse_root = mouse_root.parent
    mouse_root.matrix_world = Matrix.Translation(mouse_xy_alignment) @ mouse_root.matrix_world
    bpy.context.view_layer.update()

    placement = bpy.data.objects.new("MGF_Homecage_Placement", None)
    scene.collection.objects.link(placement)
    placement.location = (0.0, 0.0, float(float_z))
    placement.scale = (scale_factor,) * 3
    root_names = []
    for obj in objects:
        if obj.parent is not None:
            continue
        root_names.append(obj.name)
        world = obj.matrix_world.copy()
        obj.parent = placement
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = Matrix.Translation(-source_pivot) @ world
    placement.empty_display_type = "ARROWS"
    placement.empty_display_size = 40.0
    placement["MGF_float_z_mm"] = float(float_z)
    placement["MGF_source_cage_width"] = source_width
    placement["MGF_meter_to_millimeter_scale"] = scale_factor
    placement["MGF_mouse_xy_alignment_source_units"] = list(mouse_xy_alignment)
    placement["MGF_coordinate_convention"] = "+Z up; inlet is +X"
    placement["MGF_managed_root_names"] = json.dumps(sorted(root_names))
    placement["MGF_note"] = "Marker only: donor rig hierarchy is intentionally unchanged."
    return placement


def principled(material: bpy.types.Material) -> bpy.types.Node | None:
    if not material.use_nodes or material.node_tree is None:
        return None
    return next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )


def set_input(node: bpy.types.Node, name: str, value: object) -> None:
    socket = node.inputs.get(name)
    if socket is None:
        socket = next((candidate for candidate in node.inputs if candidate.name == name), None)
    if socket is not None:
        socket.default_value = value


def material_variant(
    source: bpy.types.Material,
    name: str,
    *,
    alpha: float,
    transmission: float,
    roughness: float,
    ior: float,
) -> bpy.types.Material:
    old = bpy.data.materials.get(name)
    if old is not None:
        bpy.data.materials.remove(old)
    material = source.copy()
    material.name = name
    material.use_fake_user = True
    material.diffuse_color[3] = alpha
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED" if alpha < 1.0 else "DITHERED"
    node = principled(material)
    if node is None:
        raise RuntimeError(f"No Principled BSDF found in {source.name}")
    set_input(node, "Weight", 1.0)
    set_input(node, "Alpha", alpha)
    set_input(node, "Transmission Weight", transmission)
    set_input(node, "Roughness", roughness)
    set_input(node, "IOR", ior)
    set_input(node, "Coat Weight", 0.08)
    material["MGF_alpha"] = alpha
    material["MGF_transmission"] = transmission
    material["MGF_profile"] = name
    return material


def configure_shell_materials(
    shell: bpy.types.Collection,
    profile: str,
) -> dict[str, bpy.types.Material]:
    shell_mesh = next((obj for obj in shell.all_objects if obj.type == "MESH"), None)
    if shell_mesh is None or not shell_mesh.material_slots or shell_mesh.material_slots[0].material is None:
        raise KeyError("MARS donor glass material was not appended")
    source = shell_mesh.material_slots[0].material
    variants = {
        "capture_clear": material_variant(
            source,
            "MGF_Homecage_CaptureClear",
            alpha=0.22,
            transmission=0.65,
            roughness=0.15,
            ior=1.47,
        ),
        "physical": material_variant(
            source,
            "MGF_Homecage_Physical",
            alpha=0.50,
            transmission=0.90,
            roughness=0.08,
            ior=1.47,
        ),
        "opaque_debug": material_variant(
            source,
            "MGF_Homecage_OpaqueDebug",
            alpha=1.0,
            transmission=0.0,
            roughness=0.32,
            ior=1.47,
        ),
    }
    active = variants[profile]
    for obj in shell.all_objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(active)
        obj["MGF_material_profile"] = profile
    return variants


def configured_hair_copy(
    source: bpy.types.Material,
    name: str,
    *,
    color: tuple[float, float, float, float],
    roughness: float,
    coat: float,
    random_color: float,
) -> bpy.types.Material:
    old = bpy.data.materials.get(name)
    if old is not None:
        bpy.data.materials.remove(old)
    material = source.copy()
    material.name = name
    material.use_fake_user = True
    hair = next(
        (
            node
            for node in material.node_tree.nodes
            if node.type == "BSDF_HAIR_PRINCIPLED"
        ),
        None,
    )
    if hair is None:
        raise RuntimeError(f"No Principled Hair BSDF found in {source.name}")
    hair.parametrization = "COLOR"
    set_input(hair, "Color", color)
    set_input(hair, "Roughness", roughness)
    set_input(hair, "Radial Roughness", 0.34)
    set_input(hair, "Coat", coat)
    set_input(hair, "IOR", 1.55)
    set_input(hair, "Random Color", random_color)
    set_input(hair, "Random Roughness", 0.10)
    material["MGF_C57_reference"] = "C57BL/6 warm black coat with broad fiber highlights"
    return material


def configure_c57_fur(mouse: bpy.types.Collection) -> dict[str, bpy.types.Material]:
    source = bpy.data.materials.get("Black_Whiskers.002")
    if source is None:
        source = next(
            (
                slot.material
                for obj in mouse.all_objects
                if obj.type == "CURVES"
                for slot in obj.material_slots
                if slot.material is not None
                and any(
                    node.type == "BSDF_HAIR_PRINCIPLED"
                    for node in slot.material.node_tree.nodes
                )
            ),
            None,
        )
    if source is None:
        raise RuntimeError("Could not find the MARS black-fur material")
    coat = configured_hair_copy(
        source,
        "MGF_C57_Coat",
        color=(0.0110, 0.0068, 0.0050, 1.0),
        roughness=0.44,
        coat=0.18,
        random_color=0.12,
    )
    whiskers = configured_hair_copy(
        source,
        "MGF_C57_Whiskers",
        color=(0.0025, 0.0020, 0.0018, 1.0),
        roughness=0.38,
        coat=0.04,
        random_color=0.04,
    )
    for obj in mouse.all_objects:
        if obj.type != "CURVES" or not obj.material_slots:
            continue
        if obj.name.startswith("Mouth"):
            continue
        obj.data.materials.clear()
        obj.data.materials.append(
            whiskers if obj.name.startswith("Whisker") else coat
        )
    return {"coat": coat, "whiskers": whiskers}


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def create_homecage_led_rig(
    root: bpy.types.Collection,
    cage_bottom_z: float,
) -> dict[str, bpy.types.Object]:
    collection = new_child_collection(root, "MGF_Homecage_LED_Rig")
    target = Vector((0.0, 0.0, cage_bottom_z + 48.0))
    locations = {
        "negx_negy": (-220.0, -180.0, cage_bottom_z + 340.0),
        "negx_posy": (-220.0, 180.0, cage_bottom_z + 340.0),
        "posx_negy": (220.0, -180.0, cage_bottom_z + 340.0),
        "posx_posy": (220.0, 180.0, cage_bottom_z + 340.0),
    }
    lights: dict[str, bpy.types.Object] = {}
    for label, location in locations.items():
        data = bpy.data.lights.new(f"HomecageLED_{label}", "AREA")
        data.energy = 800_000.0
        data.color = (1.0, 0.94, 0.86)
        data.shape = "RECTANGLE"
        data.size = 110.0
        data.size_y = 70.0
        data.spread = radians(105.0)
        light = bpy.data.objects.new(data.name, data)
        collection.objects.link(light)
        light.location = location
        point_camera(light, target)
        light["MGF_role"] = "diffused_corner_LED"
        light["MGF_nominal_CCT_K"] = 5200
        light["MGF_target_mm"] = list(target)
        lights[label] = light
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.name.startswith("MGF_StageEmitter_"):
            obj.hide_render = True
            obj.hide_viewport = True
    return lights


def create_camera(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    target: Vector,
    *,
    lens: float,
    role: str,
) -> bpy.types.Object:
    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    data.clip_start = 1.0
    data.clip_end = 5000.0
    data.display_size = 30.0
    data.show_passepartout = True
    data.passepartout_alpha = 0.65
    camera = bpy.data.objects.new(name, data)
    collection.objects.link(camera)
    camera.location = location
    point_camera(camera, target)
    camera["MGF_role"] = role
    camera["MGF_target_mm"] = list(target)
    return camera


def create_multiview_cameras(
    scene: bpy.types.Scene,
    root: bpy.types.Collection,
    cage_bottom_z: float,
    cage_height: float,
) -> dict[str, bpy.types.Object]:
    collection = new_child_collection(root, "MGF_Homecage_Cameras")
    top_target = Vector((0.0, 0.0, cage_bottom_z + 30.0))
    side_target = Vector((0.0, 0.0, cage_bottom_z + 58.0))
    lens = 28.0
    cameras = {
        "top": create_camera(
            collection,
            "HomecageCam_Top",
            (0.0, 0.0, cage_bottom_z + 390.0),
            top_target,
            lens=lens,
            role="top",
        ),
        "side_neg_x": create_camera(
            collection,
            "HomecageCam_SideNegX",
            (-330.0, 0.0, cage_bottom_z + 75.0),
            side_target,
            lens=lens,
            role="side_-X_opposite_inlet",
        ),
        "side_neg_y": create_camera(
            collection,
            "HomecageCam_SideNegY",
            (0.0, -330.0, cage_bottom_z + 75.0),
            side_target,
            lens=lens,
            role="side_-Y",
        ),
        "side_pos_y": create_camera(
            collection,
            "HomecageCam_SidePosY",
            (0.0, 330.0, cage_bottom_z + 75.0),
            side_target,
            lens=lens,
            role="side_+Y",
        ),
    }
    for camera in cameras.values():
        camera["MGF_view_layer"] = "MGF_Capture_Clean"
        camera["MGF_excludes_inlet_axis"] = camera["MGF_role"] != "side_+X_inlet"
    scene.camera = cameras["top"]
    return cameras


def find_layer_collection(
    root: bpy.types.LayerCollection,
    name: str,
) -> bpy.types.LayerCollection | None:
    if root.collection.name == name:
        return root
    for child in root.children:
        found = find_layer_collection(child, name)
        if found is not None:
            return found
    return None


def configure_view_layers(scene: bpy.types.Scene) -> None:
    standard = scene.view_layers[0]
    standard.name = "MGF_Standard"
    capture = scene.view_layers.get("MGF_Capture_Clean")
    if capture is None:
        capture = scene.view_layers.new("MGF_Capture_Clean")
    bottom = scene.view_layers.get("MGF_Bottom_NoInterior")
    if bottom is not None and len(scene.view_layers) > 1:
        scene.view_layers.remove(bottom)


def configure_scene(scene: bpy.types.Scene, profile: str) -> None:
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.look = "Medium Low Contrast"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.transmission_bounces = 12
    scene["MGF_project_kind"] = "MARS homecage in Cylinder laboratory"
    scene["MGF_homecage_material_profile"] = profile
    scene["MGF_camera_layout"] = "top + side(-X) + side(-Y) + side(+Y); no +X inlet view"
    scene["MGF_camera_intrinsics_note"] = "All four cameras use 28 mm lens and 36 mm sensor width."


def disconnect_missing_images() -> list[dict[str, object]]:
    """Remove stale donor texture links so the output has no broken dependencies."""
    missing: list[dict[str, object]] = []
    for image in list(bpy.data.images):
        if image.source != "FILE" or image.packed_file is not None or not image.filepath:
            continue
        resolved = Path(bpy.path.abspath(image.filepath))
        if resolved.is_file():
            continue
        users: list[str] = []
        for material in bpy.data.materials:
            if not material.use_nodes or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type != "TEX_IMAGE" or node.image != image:
                    continue
                users.append(material.name)
                for output in node.outputs:
                    for link in list(output.links):
                        material.node_tree.links.remove(link)
                node.image = None
        missing.append(
            {
                "image": image.name,
                "path": str(resolved),
                "materials_falling_back_to_defaults": sorted(set(users)),
            }
        )
        if image.name in bpy.data.images:
            bpy.data.images.remove(image)
    return missing


def main() -> None:
    args = parse_args()
    donor = args.donor.resolve()
    output = args.output.resolve()
    if not donor.is_file():
        raise FileNotFoundError(donor)
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.frame_set(1)
    scene.frame_start = 1
    laboratory_source = bpy.data.filepath
    cleanup = clear_previous_activity(scene)
    cylinder_rig_cleanup = remove_cylinder_camera_rig()
    tabletop, tabletop_z = find_tabletop()
    cage_bottom_z = (
        float(args.float_z)
        if args.float_z is not None
        else tabletop_z + float(args.table_clearance)
    )
    loaded = append_donor_collections(scene, donor)
    root, categories, imported_objects = organize_homecage_collections(scene, loaded)
    placement = place_imported_homecage(scene, imported_objects, cage_bottom_z)
    variants = configure_shell_materials(categories["shell"], args.material_profile)
    fur_materials = configure_c57_fur(categories["mouse"])
    led_lights = create_homecage_led_rig(root, cage_bottom_z)
    bpy.context.view_layer.update()
    cage_height = float(bpy.data.objects[CAGE_OBJECT_NAME].dimensions.z)
    cameras = create_multiview_cameras(scene, root, cage_bottom_z, cage_height)
    configure_view_layers(scene)
    configure_scene(scene, args.material_profile)
    missing_images = disconnect_missing_images()
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)

    scene["MGF_homecage_donor"] = str(donor)
    scene["MGF_homecage_placement"] = placement.name
    scene["MGF_missing_donor_images_replaced"] = len(missing_images)
    bpy.data.use_autopack = False
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)

    manifest = {
        "output": str(output),
        "laboratory_source": laboratory_source,
        "homecage_donor": str(donor),
        "cage_bottom_z_mm": cage_bottom_z,
        "placement_mode": "explicit_float_z" if args.float_z is not None else "tabletop",
        "tabletop": {
            "object": tabletop.name,
            "surface_z_mm": tabletop_z,
            "clearance_mm": float(args.table_clearance),
        },
        "source_scene": DONOR_SCENE,
        "source_shell_object": CAGE_OBJECT_NAME,
        "cage_dimensions_mm": list(bpy.data.objects[CAGE_OBJECT_NAME].dimensions),
        "material_profile": args.material_profile,
        "material_variants": {name: material.name for name, material in variants.items()},
        "fur_materials": {
            name: material.name for name, material in fur_materials.items()
        },
        "led_rig": {
            "collection": "MGF_Homecage_LED_Rig",
            "configuration": "four diffused corner panels angled toward cage center",
            "nominal_cct_k": 5200,
            "lights": {
                name: {
                    "object": light.name,
                    "location": list(light.location),
                    "energy": light.data.energy,
                    "size_mm": [light.data.size, light.data.size_y],
                }
                for name, light in led_lights.items()
            },
        },
        "collections": {name: collection.name for name, collection in categories.items()},
        "cameras": {
            name: {
                "object": camera.name,
                "location": list(camera.location),
                "lens_mm": camera.data.lens,
                "role": camera["MGF_role"],
            }
            for name, camera in cameras.items()
        },
        "cleanup": cleanup,
        "cylinder_rig_cleanup": cylinder_rig_cleanup,
        "missing_images_replaced": missing_images,
        "capture_view_layer": "MGF_Capture_Clean",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

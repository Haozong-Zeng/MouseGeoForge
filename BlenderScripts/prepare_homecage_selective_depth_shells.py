#!/usr/bin/env python3
"""Create per-side-camera shell proxies with only the near wall removed."""

from __future__ import annotations

from pathlib import Path

import bpy


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
PROJECT_NAME = "Homecage-BlackFur-11_01_22_12wk_m0-Core23-Sample10-Frames200-399"
EXPECTED_BLEND = ROOT / "dataset" / "Homecage" / f"{PROJECT_NAME}.blend"
SOURCE_OBJECT = "MGF_MARS_HomecageShell"
OPAQUE_MATERIAL = "MGF_DepthProxy_Opaque"

PROXIES = {
    "SideNegX": {
        "collection": "MGF_DepthProxy_SideNegX",
        "object": "MGF_DepthShellProxy_SideNegX",
        "removed_face": 0,
        "camera": "HomecageCam_SideNegX",
    },
    "SideNegY": {
        "collection": "MGF_DepthProxy_SideNegY",
        "object": "MGF_DepthShellProxy_SideNegY",
        "removed_face": 12,
        "camera": "HomecageCam_SideNegY",
    },
    "SidePosY": {
        "collection": "MGF_DepthProxy_SidePosY",
        "object": "MGF_DepthShellProxy_SidePosY",
        "removed_face": 1,
        "camera": "HomecageCam_SidePosY",
    },
}


def remove_collection_and_objects(name: str) -> None:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def get_opaque_material() -> bpy.types.Material:
    material = bpy.data.materials.get(OPAQUE_MATERIAL)
    if material is None:
        material = bpy.data.materials.new(OPAQUE_MATERIAL)
    material.use_nodes = True
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        principled.inputs["Roughness"].default_value = 1.0
        if principled.inputs.get("Alpha") is not None:
            principled.inputs["Alpha"].default_value = 1.0
    return material


def create_proxy(source: bpy.types.Object, config: dict, material: bpy.types.Material) -> None:
    collection = bpy.data.collections.new(config["collection"])
    bpy.context.scene.collection.children.link(collection)
    collection.hide_render = False

    kept_polygons = [
        polygon
        for polygon in source.data.polygons
        if polygon.index != config["removed_face"]
    ]
    vertices = [tuple(vertex.co) for vertex in source.data.vertices]
    faces = [tuple(polygon.vertices) for polygon in kept_polygons]
    mesh = bpy.data.meshes.new(f"{config['object']}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    proxy = bpy.data.objects.new(config["object"], mesh)
    collection.objects.link(proxy)
    proxy.matrix_world = source.matrix_world.copy()
    proxy.data.materials.append(material)
    for polygon in proxy.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = False
    proxy.hide_render = True
    proxy.hide_viewport = True
    proxy.display_type = "WIRE"
    proxy["mgf_depth_proxy"] = True
    proxy["mgf_source_object"] = source.name
    proxy["mgf_camera"] = config["camera"]
    proxy["mgf_removed_source_face_index"] = config["removed_face"]
    proxy["mgf_policy"] = "ignore_only_wall_between_camera_and_mouse"


def main() -> None:
    opened = Path(bpy.data.filepath).resolve()
    if opened != EXPECTED_BLEND.resolve():
        raise ValueError(f"Expected {EXPECTED_BLEND}, opened {opened}")
    source = bpy.data.objects.get(SOURCE_OBJECT)
    if source is None or source.type != "MESH":
        raise KeyError(f"Missing mesh object {SOURCE_OBJECT!r}")
    if len(source.data.polygons) != 14:
        raise ValueError(
            f"Expected 14 shell polygons before applying reviewed face mapping; "
            f"found {len(source.data.polygons)}"
        )

    for config in PROXIES.values():
        remove_collection_and_objects(config["collection"])
    material = get_opaque_material()
    for config in PROXIES.values():
        create_proxy(source, config, material)

    scene = bpy.context.scene
    scene["MGF_side_depth_wall_policy"] = "ignore_only_wall_between_camera_and_mouse"
    scene["MGF_side_depth_proxy_version"] = "v1_faces_0_12_1"
    bpy.ops.wm.save_as_mainfile(filepath=str(EXPECTED_BLEND))
    print("SELECTIVE_DEPTH_PROXIES_READY")
    for key, config in PROXIES.items():
        print(
            f"{key}: camera={config['camera']} collection={config['collection']} "
            f"removed_face={config['removed_face']}"
        )


if __name__ == "__main__":
    main()

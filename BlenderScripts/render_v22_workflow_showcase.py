import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


VIEW_MATRIX = Matrix(
    (
        (-0.7356073856, -0.3513280153, 0.5791809559, 197.5683593750),
        (0.6774082184, -0.3815117180, 0.6289412379, 266.2572326660),
        (-0.0000003862, 0.8549957275, 0.5186352730, 182.5000000000),
        (0.0, 0.0, 0.0, 1.0),
    )
)

HAIR_OBJECTS = (
    "BaseFur.001",
    "BaseFur.002",
    "BaseFur.003",
    "EarFur.001",
    "EyesFur.001",
    "LegFur.001",
    "Mouth.001",
    "TailFur.001",
    "Whisker.001",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=63)
    parser.add_argument("--end-frame", type=int, default=163)
    parser.add_argument("--single-frame", type=int)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def show(obj, render=True):
    obj.hide_viewport = False
    obj.hide_render = not render
    try:
        obj.hide_set(False)
    except RuntimeError:
        pass


def look_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, target, energy, size):
    data = bpy.data.lights.new(name=f"{name}_Data", type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def configure_visibility():
    for name in HAIR_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            show(obj)

    # Camera.001 is represented by the corrected animated combined model.
    animated_model = bpy.data.objects.get("MGF_CamB_PhysicalModel")
    if animated_model is None:
        raise RuntimeError("V22 animated Camera.001 physical model is missing")
    show(animated_model)

    for suffix in (".002", ".003"):
        for stem in (
            "Part_CameraIO",
            "Part_CameraLens",
            "Part_CameraMount",
            "Part_CameraPCB",
            "Part_CamLen",
        ):
            obj = bpy.data.objects.get(stem + suffix)
            if obj is not None:
                show(obj)

    # Do not reveal the static Camera.001 source parts underneath the animated model.
    for stem in (
        "Part_CameraIO",
        "Part_CameraLens",
        "Part_CameraMount",
        "Part_CameraPCB",
        "Part_CamLen",
    ):
        obj = bpy.data.objects.get(stem + ".001")
        if obj is not None:
            obj.hide_viewport = True
            obj.hide_render = True


def configure_observer_camera(scene):
    data = bpy.data.cameras.new("MGF_WorkflowObserver_Data")
    data.lens = 50.0
    data.sensor_width = 36.0
    data.sensor_fit = "AUTO"
    data.clip_start = 0.1
    data.clip_end = 2000.0
    camera = bpy.data.objects.new("MGF_WorkflowObserver", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = VIEW_MATRIX
    scene.camera = camera
    return camera


def configure_clay_render(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1372
    scene.render.resolution_y = 890
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False

    material = bpy.data.materials.new("MGF_WorkflowClay")
    material.use_nodes = True
    bsdf = next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )
    bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.42, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.68
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    bpy.context.view_layer.material_override = material

    world = bpy.data.worlds.new("MGF_WorkflowWorld")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.025, 0.025, 0.025, 1.0)
    background.inputs["Strength"].default_value = 0.35
    scene.world = world

    target = Vector((58.0, 112.0, 58.0))
    add_area_light(
        "MGF_WorkflowKey",
        Vector((175.0, 225.0, 285.0)),
        target,
        1900.0,
        150.0,
    )
    add_area_light(
        "MGF_WorkflowFill",
        Vector((-90.0, 160.0, 155.0)),
        target,
        950.0,
        180.0,
    )


def main():
    args = parse_args()
    scene = bpy.context.scene
    configure_visibility()
    configure_observer_camera(scene)
    configure_clay_render(scene)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.single_frame is not None:
        scene.frame_set(args.single_frame)
        scene.render.filepath = str(output_dir / f"frame_{args.single_frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    else:
        scene.frame_start = args.start_frame
        scene.frame_end = args.end_frame
        scene.render.filepath = str(output_dir / "frame_")
        bpy.ops.render.render(animation=True)

    print(
        {
            "output_dir": str(output_dir),
            "frame_range": [args.start_frame, args.end_frame],
            "single_frame": args.single_frame,
            "animated_camera_parent": bpy.data.objects["Camera.001"].parent.name,
            "animated_model_parent": bpy.data.objects["MGF_CamB_PhysicalModel"].parent.name,
        }
    )


if __name__ == "__main__":
    main()

import argparse
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


VIEW_LOCATION = Vector((57.9204, 114.6114, 57.4504))
VIEW_DISTANCE = 241.1129
VIEW_ROTATION = Quaternion((0.316827, 0.178374, 0.457018, 0.811750))

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
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def set_visible(obj, visible):
    obj.hide_viewport = not visible
    try:
        obj.hide_set(not visible)
    except RuntimeError:
        pass


def configure_scene(args):
    scene = bpy.context.scene
    for name in HAIR_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            set_visible(obj, True)

    animated_model = bpy.data.objects.get("MGF_CamB_PhysicalModel")
    if animated_model is None:
        raise RuntimeError("V22 animated Camera.001 physical model is missing")
    set_visible(animated_model, True)

    # The moving combined model replaces the static .001 source assembly.
    for suffix in (".001", ".002", ".003"):
        for stem in (
            "Part_CameraIO",
            "Part_CameraLens",
            "Part_CameraMount",
            "Part_CameraPCB",
            "Part_CamLen",
        ):
            obj = bpy.data.objects.get(stem + suffix)
            if obj is not None:
                set_visible(obj, suffix != ".001")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.frame_start = args.start_frame
    scene.frame_end = args.end_frame
    scene.frame_set(args.start_frame)
    scene.render.resolution_x = 1372
    scene.render.resolution_y = 890
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output_dir / "frame_")
    scene.render.use_file_extension = True
    return scene, output_dir


def render_when_ui_is_ready(args):
    try:
        if bpy.context.screen is None:
            return 1.0
        areas = [area for area in bpy.context.screen.areas if area.type == "VIEW_3D"]
        if not areas:
            return 1.0

        scene, output_dir = configure_scene(args)
        area = max(areas, key=lambda candidate: candidate.width * candidate.height)
        region = max(
            (region for region in area.regions if region.type == "WINDOW"),
            key=lambda candidate: candidate.width * candidate.height,
        )
        space = area.spaces.active
        space.region_3d.view_perspective = "PERSP"
        space.region_3d.view_location = VIEW_LOCATION
        space.region_3d.view_distance = VIEW_DISTANCE
        space.region_3d.view_rotation = VIEW_ROTATION
        space.lens = 50.0
        space.overlay.show_overlays = False
        shading = space.shading
        shading.type = "SOLID"
        shading.light = "STUDIO"
        shading.color_type = "SINGLE"
        shading.single_color = (0.62, 0.62, 0.62)
        shading.show_shadows = True
        shading.show_cavity = True
        shading.cavity_type = "WORLD"
        shading.show_object_outline = False

        bpy.context.view_layer.update()
        print("MGF_V22_OPENGL_RENDER_BEGIN", flush=True)
        print(
            {
                "output_dir": str(output_dir),
                "camera_parent": bpy.data.objects["Camera.001"].parent.name,
                "model_parent": bpy.data.objects["MGF_CamB_PhysicalModel"].parent.name,
            },
            flush=True,
        )
        with bpy.context.temp_override(area=area, region=region, space_data=space):
            bpy.ops.render.opengl(animation=True, view_context=True)
        print("MGF_V22_OPENGL_RENDER_END", flush=True)
    except Exception:
        traceback.print_exc()
        print("MGF_V22_OPENGL_RENDER_FAILED", flush=True)
    return None


ARGS = parse_args()
bpy.app.timers.register(lambda: render_when_ui_is_ready(ARGS), first_interval=2.0)

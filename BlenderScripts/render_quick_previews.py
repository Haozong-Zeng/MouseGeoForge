import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default=None)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--resolution", type=int, default=900)
    parser.add_argument("--hide-reference-mouse", action="store_true")
    return parser.parse_args(args_after_dash())


def set_frame(frame):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def set_action(armature_name, action_name):
    if not action_name:
        return
    armature = bpy.data.objects[armature_name]
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise RuntimeError(f"Action {action_name!r} not found.")
    armature.animation_data_create()
    armature.animation_data.action = action


def hide_reference_mouse():
    for name in ("DeformationSystem.001", "MS_Mouse.001"):
        obj = bpy.data.objects.get(name)
        if obj:
            try:
                obj.hide_set(True)
            except RuntimeError:
                pass
            obj.hide_viewport = True
            obj.hide_render = True


def configure_render(resolution):
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    for engine in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if hasattr(scene, "display") and hasattr(scene.display, "shading"):
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "OBJECT"
        shading.show_cavity = True
        shading.show_shadows = True
        if hasattr(shading, "background_type"):
            shading.background_type = "VIEWPORT"
        if hasattr(shading, "background_color"):
            shading.background_color = (1.0, 1.0, 1.0)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.color = (1.0, 1.0, 1.0)
    except Exception:
        pass


def color_preview_objects():
    palette = [
        ("MS_Mouse", (0.42, 0.28, 0.18, 1.0)),
        ("Part_PrintedMouseCup", (0.72, 0.72, 0.68, 1.0)),
        ("Part_MouseCupSide", (0.68, 0.74, 0.78, 1.0)),
        ("Part_WaterPipe", (0.12, 0.62, 0.42, 1.0)),
        ("Cranial", (0.35, 0.55, 0.95, 1.0)),
    ]
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name in {
            "MS_Mouse",
            "Part_PrintedMouseCup",
            "Part_MouseCupSide.001",
            "Part_MouseCupSide.002",
            "Part_WaterPipe",
        } or obj.name.startswith("Cranial"):
            obj.hide_viewport = False
            obj.hide_render = False
            try:
                obj.hide_set(False)
            except RuntimeError:
                pass
        color = (0.82, 0.82, 0.78, 1.0)
        for prefix, candidate in palette:
            if obj.name.startswith(prefix):
                color = candidate
                break
        obj.color = color
        for material_slot in obj.material_slots:
            material = material_slot.material
            if material:
                material.diffuse_color = color


def object_center(objects):
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return Vector((0.0, 0.0, 0.0))
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return (min_v + max_v) * 0.5


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hide_reference_mouse:
        hide_reference_mouse()
    set_action(args.armature, args.action)
    configure_render(args.resolution)
    color_preview_objects()

    frames = args.frame or [float(bpy.context.scene.frame_start)]
    cameras = []
    for name in args.camera:
        cam = bpy.data.objects.get(name)
        if cam and cam.type == "CAMERA":
            cameras.append(cam)
    if not cameras and bpy.context.scene.camera:
        cameras.append(bpy.context.scene.camera)
    if not cameras:
        raise RuntimeError("No camera available.")

    safe_action = (args.action or "no_action").replace(".", "_")
    for frame in frames:
        set_frame(frame)
        for camera in cameras:
            bpy.context.scene.camera = camera
            safe_camera = camera.name.replace(".", "_")
            filepath = out_dir / f"{safe_action}_f{int(frame):04d}_{safe_camera}_workbench.png"
            bpy.context.scene.render.filepath = str(filepath.resolve())
            bpy.ops.render.render(write_still=True)
            print(f"rendered={filepath}")


if __name__ == "__main__":
    main()

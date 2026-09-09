import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default=None)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--center-cameras", action="store_true")
    parser.add_argument("--context-object", action="append", default=[])
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--vertex-stride", type=int, default=1)
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


def evaluated_vertices_world(mesh_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        vertices = [eval_obj.matrix_world @ vertex.co for vertex in eval_mesh.vertices]
    finally:
        eval_obj.to_mesh_clear()
    return vertices


def evaluated_bbox_world(mesh_obj):
    vertices = evaluated_vertices_world(mesh_obj)
    min_v = Vector((min(v.x for v in vertices), min(v.y for v in vertices), min(v.z for v in vertices)))
    max_v = Vector((max(v.x for v in vertices), max(v.y for v in vertices), max(v.z for v in vertices)))
    return min_v, max_v, vertices


def look_at(camera, target):
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_center_camera(name, mesh_obj, view):
    min_v, max_v, _ = evaluated_bbox_world(mesh_obj)
    center = (min_v + max_v) * 0.5
    extent = max_v - min_v
    radius = max(extent.length * 0.85, 1.0)
    distance = max(radius * 3.0, 8.0)

    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(extent.x, extent.y, extent.z, 1.0) * 1.35

    if view == "front":
        offset = Vector((0.0, -distance, 0.0))
    elif view == "side":
        offset = Vector((distance, 0.0, 0.0))
    elif view == "top":
        offset = Vector((0.0, 0.0, distance))
    else:
        offset = Vector((distance, -distance, distance * 0.55))
    cam.location = center + offset
    look_at(cam, center)
    return cam


def blank_pixels(width, height, rgba=(1.0, 1.0, 1.0, 1.0)):
    return [component for _ in range(width * height) for component in rgba]


def set_pixel(pixels, width, height, x, y, color):
    if x < 0 or x >= width or y < 0 or y >= height:
        return
    idx = ((height - 1 - y) * width + x) * 4
    pixels[idx : idx + 4] = color


def draw_disc(pixels, width, height, x, y, radius, color):
    xi = int(round(x))
    yi = int(round(y))
    rr = radius * radius
    for yy in range(yi - radius, yi + radius + 1):
        for xx in range(xi - radius, xi + radius + 1):
            if (xx - xi) * (xx - xi) + (yy - yi) * (yy - yi) <= rr:
                set_pixel(pixels, width, height, xx, yy, color)


def draw_line(pixels, width, height, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = int(max(dx, dy, 1))
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 * (1.0 - t) + x1 * t))
        y = int(round(y0 * (1.0 - t) + y1 * t))
        set_pixel(pixels, width, height, x, y, color)


def project(scene, camera, world, width, height):
    ndc = world_to_camera_view(scene, camera, world)
    return ndc, ndc.x * (width - 1), ndc.y * (height - 1)


def write_image(filepath, width, height, pixels):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            offset = (y * width + x) * 4
            for value in pixels[offset : offset + 4]:
                row.append(max(0, min(255, int(round(value * 255.0)))))
        rows.append(bytes(row))

    raw = b"".join(rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    filepath.write_bytes(png)


def draw_projected_vertices(scene, camera, vertices, pixels, width, height, stride, color, radius):
    visible_vertices = 0
    xs = []
    ys = []
    for world in vertices[:: max(stride, 1)]:
        ndc, x, y = project(scene, camera, world, width, height)
        if ndc.z <= 0.0:
            continue
        if -0.05 <= ndc.x <= 1.05 and -0.05 <= ndc.y <= 1.05:
            visible_vertices += 1
            xs.append(ndc.x)
            ys.append(ndc.y)
            draw_disc(pixels, width, height, x, y, radius, color)
    return visible_vertices, xs, ys


def render_projection(scene, camera, mesh_vertices, context_vertices, armature, filepath, width, height, stride):
    pixels = blank_pixels(width, height)
    context_metrics = {}
    xs = []
    ys = []

    palette = [
        (0.72, 0.72, 0.72, 1.0),
        (0.05, 0.62, 0.92, 1.0),
        (0.95, 0.65, 0.05, 1.0),
        (0.34, 0.72, 0.18, 1.0),
        (0.72, 0.22, 0.88, 1.0),
    ]
    for index, (name, vertices) in enumerate(context_vertices.items()):
        visible, obj_xs, obj_ys = draw_projected_vertices(
            scene,
            camera,
            vertices,
            pixels,
            width,
            height,
            max(stride, 2),
            palette[index % len(palette)],
            1,
        )
        context_metrics[name] = {
            "visible_vertex_count": visible,
            "ndc_min": [min(obj_xs), min(obj_ys)] if obj_xs else None,
            "ndc_max": [max(obj_xs), max(obj_ys)] if obj_xs else None,
        }

    visible_vertices, xs, ys = draw_projected_vertices(
        scene,
        camera,
        mesh_vertices,
        pixels,
        width,
        height,
        stride,
        (0.1, 0.1, 0.1, 1.0),
        1,
    )

    bone_points = {}
    for bone in armature.data.bones:
        world = armature.matrix_world @ armature.pose.bones[bone.name].matrix.translation
        ndc, x, y = project(scene, camera, world, width, height)
        bone_points[bone.name] = {
            "world": [world.x, world.y, world.z],
            "ndc": [ndc.x, ndc.y, ndc.z],
            "pixel": [x, y],
            "visible_01": 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0 and ndc.z > 0.0,
        }

    for bone in armature.data.bones:
        if not bone.parent:
            continue
        child = bone_points.get(bone.name)
        parent = bone_points.get(bone.parent.name)
        if not child or not parent:
            continue
        if child["visible_01"] or parent["visible_01"]:
            draw_line(
                pixels,
                width,
                height,
                child["pixel"][0],
                child["pixel"][1],
                parent["pixel"][0],
                parent["pixel"][1],
                (0.85, 0.18, 0.12, 1.0),
            )

    for point in bone_points.values():
        if point["visible_01"]:
            draw_disc(
                pixels,
                width,
                height,
                point["pixel"][0],
                point["pixel"][1],
                3,
                (0.1, 0.2, 0.95, 1.0),
            )

    write_image(filepath, width, height, pixels)
    return {
        "visible_vertex_count": visible_vertices,
        "mesh_ndc_min": [min(xs), min(ys)] if xs else None,
        "mesh_ndc_max": [max(xs), max(ys)] if xs else None,
        "context_objects": context_metrics,
        "bone_heads": bone_points,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_action(args.armature, args.action)
    armature = bpy.data.objects[args.armature]
    mesh = bpy.data.objects[args.mesh]
    frames = args.frame or [float(bpy.context.scene.frame_start)]
    context_objects = []
    for name in args.context_object:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "MESH":
            context_objects.append(obj)

    cameras = []
    for name in args.camera:
        cam = bpy.data.objects.get(name)
        if cam and cam.type == "CAMERA":
            cameras.append(cam)
    if not cameras and bpy.context.scene.camera:
        cameras.append(bpy.context.scene.camera)

    if args.center_cameras:
        set_frame(frames[0])
        for view in ("front", "side", "top", "iso"):
            cameras.append(create_center_camera(f"CHECK_{view}", mesh, view))

    manifest = {
        "file": bpy.data.filepath,
        "mesh": mesh.name,
        "armature": armature.name,
        "action": armature.animation_data.action.name
        if armature.animation_data and armature.animation_data.action
        else None,
        "renders": [],
    }

    for frame in frames:
        set_frame(frame)
        min_v, max_v, vertices = evaluated_bbox_world(mesh)
        context_vertices = {
            obj.name: evaluated_vertices_world(obj)
            for obj in context_objects
        }
        for camera in cameras:
            safe_camera = camera.name.replace(".", "_")
            safe_action = (manifest["action"] or "no_action").replace(".", "_")
            filepath = out_dir / f"{safe_action}_f{int(frame):04d}_{safe_camera}_projection.png"
            metrics = render_projection(
                bpy.context.scene,
                camera,
                vertices,
                context_vertices,
                armature,
                filepath,
                args.resolution,
                args.resolution,
                args.vertex_stride,
            )
            metrics.update(
                {
                    "frame": frame,
                    "camera": camera.name,
                    "image": str(filepath),
                    "bbox_world_min": [min_v.x, min_v.y, min_v.z],
                    "bbox_world_max": [max_v.x, max_v.y, max_v.z],
                    "bbox_world_size": [(max_v - min_v).x, (max_v - min_v).y, (max_v - min_v).z],
                }
            )
            manifest["renders"].append(metrics)

    safe_manifest_action = (manifest["action"] or "no_action").replace(".", "_")
    manifest_path = out_dir / f"{safe_manifest_action}_projection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_PROJECTION_BEGIN")
    print(json.dumps({"manifest": str(manifest_path), "renders": len(manifest["renders"])}, ensure_ascii=False))
    print("MOUSE_GEO_FORGE_PROJECTION_END")


if __name__ == "__main__":
    main()

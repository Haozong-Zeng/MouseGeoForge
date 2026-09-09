import json

import bpy
from mathutils import Vector


def world_bounds(objects):
    points = []
    per_object = {}
    for obj in objects:
        if obj.type != "MESH":
            continue
        object_points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        points.extend(object_points)
        if object_points:
            min_v = Vector(tuple(min(point[i] for point in object_points) for i in range(3)))
            max_v = Vector(tuple(max(point[i] for point in object_points) for i in range(3)))
            per_object[obj.name] = {
                "center": list((min_v + max_v) * 0.5),
                "min": list(min_v),
                "max": list(max_v),
            }
    if not points:
        return None
    min_v = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    max_v = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    return {
        "center": list((min_v + max_v) * 0.5),
        "min": list(min_v),
        "max": list(max_v),
        "objects": per_object,
    }


def suffix_group(suffix):
    names = [
        f"Part_CameraIO{suffix}",
        f"Part_CameraLens{suffix}",
        f"Part_CameraMount{suffix}",
        f"Part_CameraPCB{suffix}",
        f"Part_CamLen{suffix}",
    ]
    return [bpy.data.objects[name] for name in names if name in bpy.data.objects]


def main():
    cameras = [
        obj
        for obj in bpy.data.objects
        if obj.type == "CAMERA" and not obj.name.startswith("pasted")
    ]
    groups = {}
    for suffix in (".001", ".002", ".003"):
        bounds = world_bounds(suffix_group(suffix))
        if bounds is None:
            continue
        center = Vector(bounds["center"])
        bounds["camera_distances"] = {
            camera.name: float((camera.matrix_world.translation - center).length)
            for camera in cameras
        }
        groups[suffix] = bounds
    print("MGF_CAMERA_MODEL_MAPPING_BEGIN")
    print(
        json.dumps(
            {
                "file": bpy.data.filepath,
                "cameras": {
                    camera.name: list(camera.matrix_world.translation) for camera in cameras
                },
                "groups": groups,
            },
            indent=2,
        )
    )
    print("MGF_CAMERA_MODEL_MAPPING_END")


if __name__ == "__main__":
    main()

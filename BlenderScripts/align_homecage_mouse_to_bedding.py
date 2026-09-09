"""Apply a constant Z correction so the animated mouse contacts the bedding plane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


MESH_NAME = "MS_Mouse.001"
OFFSET_NAME = "MGF_First200_Homecage_Offset"
FLOOR_PREFIX = "MGF_MARS_HomecageFloor_"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=1)
    return parser.parse_args(argv)


def evaluated_min_z(obj: bpy.types.Object) -> float:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return min(float((evaluated.matrix_world @ vertex.co).z) for vertex in mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def object_max_z(obj: bpy.types.Object) -> float:
    return max(float((obj.matrix_world @ Vector(corner)).z) for corner in obj.bound_box)


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    mesh = bpy.data.objects.get(MESH_NAME)
    offset = bpy.data.objects.get(OFFSET_NAME)
    floors = [obj for obj in bpy.data.objects if obj.name.startswith(FLOOR_PREFIX)]
    if mesh is None or mesh.type != "MESH":
        raise KeyError(MESH_NAME)
    if offset is None:
        raise KeyError(OFFSET_NAME)
    if not floors:
        raise KeyError(FLOOR_PREFIX)

    floor_z = max(object_max_z(obj) for obj in floors)
    before_min_z = evaluated_min_z(mesh)
    correction_z = floor_z - before_min_z
    old_offset_z = float(offset.location.z)
    offset.location.z += correction_z
    bpy.context.view_layer.update()
    after_min_z = evaluated_min_z(mesh)

    scene["MGF_ground_contact_floor_z_mm"] = floor_z
    scene["MGF_ground_contact_correction_z_mm"] = correction_z
    scene["MGF_ground_contact_reference_frame"] = args.frame
    output = args.output_blend.resolve()
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(
        "GROUND_CONTACT_ALIGNMENT "
        + json.dumps(
            {
                "frame": args.frame,
                "floor_z_mm": floor_z,
                "mouse_min_z_before_mm": before_min_z,
                "mouse_min_z_after_mm": after_min_z,
                "correction_z_mm": correction_z,
                "offset_z_before_mm": old_offset_z,
                "offset_z_after_mm": float(offset.location.z),
                "saved": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

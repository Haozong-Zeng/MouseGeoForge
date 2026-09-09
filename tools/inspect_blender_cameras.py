#!/usr/bin/env python3
"""Print Blender camera lens/sensor settings for the current .blend file."""

from __future__ import annotations

import argparse
import json
import math
import sys

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", default="CameraA,CameraB,CameraC,CameraD,CameraE")
    parser.add_argument("--all-cameras", action="store_true")
    parser.add_argument("--imx296-width-mm", type=float, default=1456 * 0.00345)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    payload = {
        "blend_file": bpy.data.filepath,
        "resolution": [res_x, res_y],
        "cameras": [],
    }
    if args.all_cameras:
        names = sorted(obj.name for obj in bpy.data.objects if obj.type == "CAMERA")
    else:
        names = [item.strip() for item in args.cameras.split(",") if item.strip()]
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "CAMERA":
            payload["cameras"].append({"name": name, "exists": False})
            continue
        cam = obj.data
        sensor_width = float(cam.sensor_width)
        lens = float(cam.lens)
        fx = lens / sensor_width * res_x if sensor_width else None
        payload["cameras"].append(
            {
                "name": name,
                "exists": True,
                "lens_mm": lens,
                "sensor_width_mm": sensor_width,
                "sensor_height_mm": float(cam.sensor_height),
                "sensor_fit": cam.sensor_fit,
                "angle_x_deg": math.degrees(float(cam.angle_x)),
                "angle_y_deg": math.degrees(float(cam.angle_y)),
                "fx_px_horizontal_fit": fx,
                "equiv_lens_if_sensor_36mm": lens / sensor_width * 36.0 if sensor_width else None,
                "equiv_lens_if_imx296_width": lens / sensor_width * args.imx296_width_mm if sensor_width else None,
            }
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

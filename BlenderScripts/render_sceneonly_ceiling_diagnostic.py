"""Render controlled A/B variants for SceneOnly ceiling reflections.

This script changes render visibility only in memory and never saves the blend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_sceneonly_cycles_multiview_hq as hq


CEILING_OBJECTS = ("Ceiling", "Ceiling.001", "Plane.063")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="Camera.017")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args(argv)


def white_gloss_objects() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and any(material and material.name == "white_gloss" for material in obj.data.materials)
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.frame_set(args.frame)
    render_args = argparse.Namespace(
        resolution=args.resolution,
        samples=args.samples,
        adaptive_threshold=0.015,
        exposure=0.7,
    )
    device = hq.configure_cycles(scene, render_args)
    hq.enable_fur()

    camera = bpy.data.objects.get(args.camera)
    if camera is None or camera.type != "CAMERA":
        raise KeyError(f"Missing camera {args.camera}")
    camera.hide_render = False
    scene.camera = camera

    classroom_areas = [
        obj for obj in bpy.data.objects if obj.type == "LIGHT" and obj.name.startswith("Area")
    ]
    ceilings = [bpy.data.objects[name] for name in CEILING_OBJECTS if name in bpy.data.objects]
    white_panels = white_gloss_objects()
    controlled = set(classroom_areas + ceilings + white_panels)
    original_visibility = {obj: obj.hide_render for obj in controlled}

    variants = {
        "baseline": set(),
        "no_classroom_area_lights": set(classroom_areas),
        "no_ceiling_meshes": set(ceilings),
        "no_ceiling_or_classroom_area_lights": set(ceilings + classroom_areas),
        "no_white_gloss_meshes": set(white_panels),
    }
    records = []
    for name, hidden_objects in variants.items():
        for obj, hidden in original_visibility.items():
            obj.hide_render = hidden
        for obj in hidden_objects:
            obj.hide_render = True
        bpy.context.view_layer.update()

        output = output_dir / f"{name}_{args.camera.replace('.', '_')}_frame{args.frame:04d}.png"
        scene.render.filepath = str(output)
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        records.append(
            {
                "variant": name,
                "output": str(output),
                "seconds": time.perf_counter() - started,
                "hidden": sorted(obj.name for obj in hidden_objects),
            }
        )
        print(f"CEILING_DIAGNOSTIC {json.dumps(records[-1])}", flush=True)

    for obj, hidden in original_visibility.items():
        obj.hide_render = hidden
    manifest = {
        "format": "sceneonly_ceiling_diagnostic_v1",
        "blend": bpy.data.filepath,
        "camera": args.camera,
        "frame": args.frame,
        "resolution": args.resolution,
        "samples": args.samples,
        "device": device,
        "records": records,
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

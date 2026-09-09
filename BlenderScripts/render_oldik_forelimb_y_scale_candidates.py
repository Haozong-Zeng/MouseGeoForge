"""Render old-IK forelimb rest-length scale candidates for visual comparison."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_quick_previews as quick  # noqa: E402
import sweep_oldik_forelimb_y_scale as sweep  # noqa: E402


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, action="append", required=True)
    parser.add_argument("--frame", type=float, action="append", required=True)
    parser.add_argument("--camera", default="Camera.002")
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--resolution", type=int, default=640)
    return parser.parse_args(args_after_dash())


def set_frame(frame: float):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def render_current(out_dir: Path, scale: float, frame: float, camera_name: str):
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        raise RuntimeError(f"Missing camera {camera_name!r}")
    set_frame(frame)
    bpy.context.scene.camera = camera
    safe_scale = f"s{scale:.3f}".replace(".", "p")
    safe_camera = camera.name.replace(".", "_")
    filepath = out_dir / f"{safe_scale}_f{int(frame):04d}_{safe_camera}_workbench.png"
    bpy.context.scene.render.filepath = str(filepath.resolve())
    bpy.ops.render.render(write_still=True)
    print(f"rendered={filepath}")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for scale in args.scale:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        quick.configure_render(args.resolution)
        quick.color_preview_objects()
        rig = bpy.data.objects.get(args.armature)
        if rig is None:
            raise RuntimeError(f"Missing armature {args.armature!r}")
        sweep.apply_forelimb_rest_y_length(rig, float(scale))
        for frame in args.frame:
            render_current(out_dir, float(scale), float(frame), args.camera)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render low-sample previews for the canonical cylinder 8-view rig."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy


CAMERA_NAMES = tuple(f"Camera.{index:03d}" for index in range(1, 9))


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--engine",
        choices=("CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"),
        default="CYCLES",
    )
    parser.add_argument("--cuda-device-substring", default="RTX PRO 6000")
    parser.add_argument("--camera-names", nargs="*", default=list(CAMERA_NAMES))
    parser.add_argument("--hide-cylinder", action="store_true")
    return parser.parse_args(argv)


def configure_cuda(scene: bpy.types.Scene, device_substring: str) -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    selected = []
    for device in preferences.devices:
        use = device.type == "CUDA" and device_substring.lower() in device.name.lower()
        device.use = use
        if use:
            selected.append(f"{device.type}:{device.name}")
    if not selected:
        available = [f"{device.type}:{device.name}" for device in preferences.devices]
        raise RuntimeError(f"CUDA device {device_substring!r} not found; available={available}")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    return selected


def compositor_tree(scene: bpy.types.Scene):
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        return scene.node_tree
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = bpy.data.node_groups.new(f"{scene.name}_MGF_PreviewCompositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
    return tree


def configure_file_output(scene: bpy.types.Scene, output_dir: Path, stem: str):
    tree = compositor_tree(scene)
    render_layers = tree.nodes.get("MGF_PreviewRenderLayers")
    if render_layers is None:
        render_layers = tree.nodes.new("CompositorNodeRLayers")
        render_layers.name = "MGF_PreviewRenderLayers"
    output = tree.nodes.get("MGF_PreviewOutput")
    if output is None:
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.name = "MGF_PreviewOutput"
    for node in tree.nodes:
        if node.type == "OUTPUT_FILE" and node != output:
            node.mute = True
    output.mute = False
    if hasattr(output, "base_path"):
        output.base_path = str(output_dir)
        output.file_slots[0].path = stem
        output.format.file_format = "PNG"
        output.format.color_mode = "RGBA"
        output.format.color_depth = "8"
        image_input = output.inputs[0]
    else:
        output.directory = str(output_dir)
        output.file_name = stem
        output.file_output_items.clear()
        item = output.file_output_items.new(socket_type="RGBA", name="Image")
        item.override_node_format = True
        item.format.file_format = "PNG"
        item.format.color_mode = "RGBA"
        item.format.color_depth = "8"
        image_input = output.inputs["Image"]
    for link in list(tree.links):
        if link.to_node == output:
            tree.links.remove(link)
    tree.links.new(render_layers.outputs["Image"], image_input)
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if args.hide_cylinder:
        for name in ("Part_GlassCylinder160mm", "Part_Cylinder"):
            cylinder = bpy.data.objects.get(name)
            if cylinder is not None:
                cylinder.hide_render = True
                cylinder.hide_viewport = True
                cylinder.hide_set(True)
        bpy.context.view_layer.update()
    if args.engine == "CYCLES":
        devices = configure_cuda(scene, args.cuda_device_substring)
    else:
        scene.render.engine = args.engine
        devices = [args.engine]
    if args.engine == "BLENDER_WORKBENCH":
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = True
    scene.frame_set(args.frame)
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True

    timings = []
    for view_index, camera_name in enumerate(args.camera_names):
        if camera_name not in CAMERA_NAMES:
            raise ValueError(f"Unsupported camera {camera_name!r}; expected one of {CAMERA_NAMES}")
        camera = bpy.data.objects[camera_name]
        scene.camera = camera
        output = output_dir / f"view_{view_index:02d}_{camera_name}.png"
        stem = output.stem
        for stale in output_dir.glob(f"{stem}*.png"):
            stale.unlink()
        started = time.perf_counter()
        if args.engine != "CYCLES":
            if hasattr(scene.render, "use_compositing"):
                scene.render.use_compositing = False
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
        else:
            configure_file_output(scene, output_dir, stem)
            bpy.ops.render.render(write_still=False)
        elapsed = time.perf_counter() - started
        if args.engine != "CYCLES":
            candidates = [output]
        else:
            candidates = sorted(
                [*output_dir.glob(f"{stem}*.png"), *output_dir.glob(f"{stem}*.exr")],
                key=lambda path: path.stat().st_mtime,
            )
        if not candidates:
            raise FileNotFoundError(f"Compositor did not write {stem}*.png or {stem}*.exr")
        source = candidates[-1]
        if source.suffix.lower() == ".exr":
            rendered_image = bpy.data.images.load(str(source), check_existing=False)
            try:
                rendered_image.save_render(filepath=str(output), scene=scene)
            finally:
                bpy.data.images.remove(rendered_image)
            source.unlink()
        elif source != output:
            source.replace(output)
        for extra in [*output_dir.glob(f"{stem}*.png"), *output_dir.glob(f"{stem}*.exr")]:
            if extra != output:
                extra.unlink()
        timings.append({"camera": camera_name, "seconds": elapsed, "output": str(output)})
        print(f"preview_render camera={camera_name} seconds={elapsed:.3f}", flush=True)

    manifest = {
        "blend": bpy.data.filepath,
        "frame": args.frame,
        "samples": args.samples,
        "resolution": [args.resolution, args.resolution],
        "devices": devices,
        "collection_render_visibility": {
            name: not bpy.data.collections[name].hide_render
            for name in ("Mouse_White", "Mouse_Brown", "moseq", "Collection")
            if name in bpy.data.collections
        },
        "timings": timings,
    }
    (output_dir / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backfill visible mouse body masks into an existing BlenderPyMouse export.

Run with the matching animation project loaded:

    blender source.blend --background --python export_existing_blenderpymouse_body_masks.py -- \
      --dataset-root /path/to/dataset \
      --mouse-collection Brown_Mouse

The script renders flat white mouse-associated objects against black scene
geometry. Non-mouse geometry remains visible as an occluder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix
from PIL import Image


FORMAT_NAME = "mousegeoforge_blender_visible_body_mask_v1"
PHYSICAL_CAMERA_OBJECT_PREFIXES = (
    "CameraItems.",
    "Part_CameraIO.",
    "Part_CameraLens.",
    "Part_CameraMount.",
    "Part_CameraPCB.",
    "Part_CamLen.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mouse-collection", default="Brown_Mouse")
    parser.add_argument("--output-subdir", default="body_masks")
    parser.add_argument("--camera-model-collections", default="CamA,CamB,CamC")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frames", default="", help="Optional comma-separated exact frame numbers.")
    parser.add_argument("--view-ids", default="")
    parser.add_argument(
        "--include-hair",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include CURVES fur in the silhouette. Disabled by default for speed and smooth masks.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-tolerance", type=float, default=1e-4)
    parser.add_argument("--binary-threshold", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def resolve_path(text: str) -> Path:
    return Path(bpy.path.abspath(text)).resolve()


def parse_name_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def frame_number(path: Path) -> int:
    return int(path.name.removeprefix("frame_"))


def selected_scene_roots(dataset_root: Path, args: argparse.Namespace) -> list[Path]:
    roots = sorted(
        (path for path in dataset_root.glob("frame_*") if path.is_dir()),
        key=frame_number,
    )
    if args.frame_start > 0:
        roots = [path for path in roots if frame_number(path) >= args.frame_start]
    if args.frame_end > 0:
        roots = [path for path in roots if frame_number(path) <= args.frame_end]
    if args.frames:
        requested = {int(value) for value in parse_name_list(args.frames)}
        roots = [path for path in roots if frame_number(path) in requested]
    step = max(1, int(args.frame_step))
    return roots[::step]


def collection_objects_recursive(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    objects = {obj.name: obj for obj in collection.objects}
    for child in collection.children:
        for obj in collection_objects_recursive(child):
            objects[obj.name] = obj
    return list(objects.values())


def configure_flat_mask_render(scene: bpy.types.Scene) -> str:
    engine_items = {
        item.identifier
        for item in scene.render.bl_rna.properties["engine"].enum_items
    }
    candidates = ("BLENDER_WORKBENCH", "BLENDER_WORKBENCH_NEXT")
    engine = next((name for name in candidates if name in engine_items), None)
    if engine is not None:
        scene.render.engine = engine
        shading = scene.display.shading
        shading.light = "FLAT"
        shading.color_type = "OBJECT"
        shading.show_shadows = False
        shading.show_cavity = False
        shading.show_outline = False
        shading.show_specular_highlight = False
        shading.background_type = "VIEWPORT"
        shading.background_color = (0.0, 0.0, 0.0)
        render_mode = "WORKBENCH_OBJECT_COLOR"
    else:
        eevee = next(
            (name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if name in engine_items),
            None,
        )
        if eevee is None:
            raise RuntimeError(f"No supported mask render engine: {sorted(engine_items)}")
        scene.render.engine = eevee
        scene.render.use_compositing = False
        scene.world.color = (0.0, 0.0, 0.0)
        render_mode = "EEVEE_EMISSION"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.resolution_percentage = 100

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    return render_mode


def hide_camera_models(collection_names: list[str]) -> list[str]:
    hidden: list[str] = []
    for name in collection_names:
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        collection.hide_render = True
        hidden.append(name)
        for obj in collection_objects_recursive(collection):
            obj.hide_render = True
    for obj in bpy.data.objects:
        if obj.name.startswith(PHYSICAL_CAMERA_OBJECT_PREFIXES):
            obj.hide_render = True
    return hidden


def create_emission_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if not hasattr(obj.data, "materials"):
        return
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    if hasattr(obj.data, "polygons"):
        for polygon in obj.data.polygons:
            polygon.material_index = 0


def configure_object_appearance(
    mouse_collection_name: str,
    render_mode: str,
    include_hair: bool,
) -> list[str]:
    collection = bpy.data.collections.get(mouse_collection_name)
    if collection is None:
        raise KeyError(f"Mouse collection {mouse_collection_name!r} not found")
    collection_objects = collection_objects_recursive(collection)
    if not include_hair:
        for obj in collection_objects:
            if obj.type != "CURVES":
                continue
            obj.hide_render = True
            obj.hide_viewport = True
            try:
                obj.hide_set(True)
            except RuntimeError:
                pass
            for modifier in obj.modifiers:
                modifier.show_render = False
                modifier.show_viewport = False

    target_objects = {
        obj.name: obj
        for obj in collection_objects
        if obj.type in {"MESH", "CURVE", "CURVES", "SURFACE", "META", "FONT"}
        and not obj.hide_render
    }
    if not target_objects:
        raise ValueError(f"No renderable objects in mouse collection {mouse_collection_name!r}")

    black_material = None
    white_material = None
    if render_mode == "EEVEE_EMISSION":
        black_material = create_emission_material(
            "MGF_BodyMask_Black",
            (0.0, 0.0, 0.0, 1.0),
        )
        white_material = create_emission_material(
            "MGF_BodyMask_White",
            (1.0, 1.0, 1.0, 1.0),
        )

    renderable_types = {"MESH", "CURVE", "CURVES", "SURFACE", "META", "FONT"}
    for obj in bpy.data.objects:
        if obj.type in renderable_types:
            obj.color = (0.0, 0.0, 0.0, 1.0)
            if black_material is not None:
                assign_material(obj, black_material)
    for obj in target_objects.values():
        obj.color = (1.0, 1.0, 1.0, 1.0)
        if white_material is not None:
            assign_material(obj, white_material)
    return sorted(target_objects)


def camera_extrinsic(camera_obj: bpy.types.Object) -> list[list[float]]:
    blender_to_opencv = Matrix(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    )
    world_to_bcam = camera_obj.matrix_world.inverted()
    rotation = blender_to_opencv @ world_to_bcam.to_3x3()
    translation = blender_to_opencv @ world_to_bcam.translation
    output = [[0.0] * 4 for _ in range(4)]
    for row in range(3):
        for col in range(3):
            output[row][col] = float(rotation[row][col])
        output[row][3] = float(translation[row])
    output[3][3] = 1.0
    return output


def matrix_max_abs_difference(left: list[list[float]], right: list[list[float]]) -> float:
    return max(
        abs(float(left[row][col]) - float(right[row][col]))
        for row in range(4)
        for col in range(4)
    )


def render_scene_masks(
    scene_root: Path,
    view_ids: set[int] | None,
    args: argparse.Namespace,
) -> tuple[int, int]:
    frame = frame_number(scene_root)
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    scene_meta = read_json(scene_root / "scene_meta.json")
    rendered = 0
    skipped = 0
    for view in scene_meta.get("views", []):
        view_id = int(view["view_id"])
        if view_ids is not None and view_id not in view_ids:
            continue
        camera_name = str(view["camera"])
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None or camera_obj.type != "CAMERA":
            raise KeyError(f"Camera {camera_name!r} not found for {scene_root.name}")
        difference = matrix_max_abs_difference(
            camera_extrinsic(camera_obj),
            view["extrinsic"],
        )
        if not math.isfinite(difference) or difference > args.camera_tolerance:
            raise ValueError(
                f"Camera mismatch {scene_root.name}/{camera_name}: "
                f"max_abs={difference:.6g} > {args.camera_tolerance:.6g}"
            )

        output_dir = scene_root / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{view_id:08d}.png"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        bpy.context.scene.camera = camera_obj
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FileNotFoundError(f"Mask render did not write {output_path}")
        with Image.open(output_path) as image:
            grayscale = image.convert("L")
            binary = grayscale.point(
                lambda value: 255 if value >= args.binary_threshold else 0
            )
            binary.save(output_path)
        rendered += 1
    return rendered, skipped


def update_metadata(
    dataset_root: Path,
    args: argparse.Namespace,
    target_objects: list[str],
    hidden_camera_collections: list[str],
) -> None:
    metadata_path = dataset_root / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    metadata.setdefault("layout", {})["body_masks"] = (
        f"frame_xxxxxx/{args.output_subdir}/00000000.png"
    )
    scene_roots = selected_scene_roots(dataset_root, argparse.Namespace(
        frame_start=0,
        frame_end=0,
        frame_step=1,
        frames="",
    ))
    mask_count = sum(
        1
        for scene_root in scene_roots
        for path in (scene_root / args.output_subdir).glob("*.png")
        if path.is_file() and path.stat().st_size > 0
    )
    metadata["body_mask_annotations"] = {
        "format": FORMAT_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "output_layout": f"frame_xxxxxx/{args.output_subdir}/00000000.png",
        "mouse_collection": args.mouse_collection,
        "included_objects": target_objects,
        "camera_model_collections_hidden": hidden_camera_collections,
        "render_engine": bpy.context.scene.render.engine,
        "include_hair": bool(args.include_hair),
        "binary_threshold": int(args.binary_threshold),
        "pixel_values": [0, 255],
        "mask_view_count": mask_count,
        "expected_view_count": sum(
            len(read_json(scene_root / "scene_meta.json").get("views", []))
            for scene_root in scene_roots
        ),
        "occlusion_policy": "non-mouse scene geometry remains as black foreground occluders",
    }
    write_json_atomic(metadata_path, metadata)


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    dataset_root = resolve_path(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    roots = selected_scene_roots(dataset_root, args)
    if not roots:
        raise ValueError(f"No selected frame_* scenes under {dataset_root}")
    requested_view_ids = (
        {int(value) for value in parse_name_list(args.view_ids)}
        if args.view_ids
        else None
    )

    render_mode = configure_flat_mask_render(bpy.context.scene)
    hidden_camera_collections = hide_camera_models(
        parse_name_list(args.camera_model_collections)
    )
    target_objects = configure_object_appearance(
        args.mouse_collection,
        render_mode,
        args.include_hair,
    )
    print(
        "body_mask_setup",
        f"dataset={dataset_root.name}",
        f"target_objects={','.join(target_objects)}",
        f"render_mode={render_mode}",
        f"include_hair={args.include_hair}",
        f"hidden_camera_collections={','.join(hidden_camera_collections)}",
        flush=True,
    )

    rendered_total = 0
    skipped_total = 0
    for index, scene_root in enumerate(roots, start=1):
        rendered, skipped = render_scene_masks(scene_root, requested_view_ids, args)
        rendered_total += rendered
        skipped_total += skipped
        if index == 1 or index == len(roots) or index % max(1, args.progress_every) == 0:
            print(
                "body_mask_progress",
                f"processed={index}/{len(roots)}",
                f"frame={frame_number(scene_root)}",
                f"rendered={rendered_total}",
                f"skipped={skipped_total}",
                flush=True,
            )

    update_metadata(
        dataset_root,
        args,
        target_objects,
        hidden_camera_collections,
    )
    print(
        "body_mask_complete",
        f"dataset={dataset_root}",
        f"rendered={rendered_total}",
        f"skipped={skipped_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()

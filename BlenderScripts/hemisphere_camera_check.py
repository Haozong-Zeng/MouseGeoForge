import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import software_projection_check as projection


DEFAULT_CONTEXT_OBJECTS = [
    "Part_PrintedMouseCup",
    "Part_MouseCupSide.001",
    "Part_MouseCupSide.002",
    "Part_WaterPipe",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate hemisphere-like candidate inspection cameras from the current "
            "physical/scene cameras, score their coverage, and export projection checks."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-blend", default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default=None)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--context-object", action="append", default=[])
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--vertex-stride", type=int, default=2)
    parser.add_argument("--rings", type=int, default=4)
    parser.add_argument("--azimuth-samples", type=int, default=12)
    parser.add_argument("--max-polar-deg", type=float, default=82.0)
    parser.add_argument("--radius-scale", type=float, action="append", default=[])
    parser.add_argument("--render-top", type=int, default=24)
    parser.add_argument("--selection-separation-deg", type=float, default=12.0)
    parser.add_argument("--include-seed-projections", action="store_true")
    return parser.parse_args(args_after_dash())


def set_action(armature_name, action_name):
    if not action_name:
        return None
    armature = bpy.data.objects[armature_name]
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise RuntimeError(f"Action {action_name!r} not found.")
    armature.animation_data_create()
    armature.animation_data.action = action
    return action


def set_frame(frame):
    projection.set_frame(frame)


def camera_forward(camera):
    return (camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()


def outer(a, b):
    return Matrix(
        (
            (a.x * b.x, a.x * b.y, a.x * b.z),
            (a.y * b.x, a.y * b.y, a.y * b.z),
            (a.z * b.x, a.z * b.y, a.z * b.z),
        )
    )


def estimate_camera_focus(cameras):
    identity = Matrix.Identity(3)
    system = Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    rhs = Vector((0.0, 0.0, 0.0))
    for camera in cameras:
        direction = camera_forward(camera)
        projector = identity - outer(direction, direction)
        position = camera.matrix_world.translation.copy()
        system += projector
        rhs += projector @ position
    if abs(system.determinant()) < 1.0e-8:
        return None
    return system.inverted_safe() @ rhs


def evaluated_object_vertices(obj):
    return projection.evaluated_vertices_world(obj)


def bbox_from_vertices(vertices):
    min_v = Vector((min(v.x for v in vertices), min(v.y for v in vertices), min(v.z for v in vertices)))
    max_v = Vector((max(v.x for v in vertices), max(v.y for v in vertices), max(v.z for v in vertices)))
    return min_v, max_v


def bbox_center(objects):
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        points.extend(evaluated_object_vertices(obj))
    if not points:
        return Vector((0.0, 0.0, 0.0)), 1.0
    min_v, max_v = bbox_from_vertices(points)
    return (min_v + max_v) * 0.5, max((max_v - min_v).length, 1.0)


def choose_seed_cameras(camera_names):
    if camera_names:
        cameras = []
        for name in camera_names:
            camera = bpy.data.objects.get(name)
            if camera is None or camera.type != "CAMERA":
                raise RuntimeError(f"Camera {name!r} not found.")
            cameras.append(camera)
        return cameras
    cameras = [
        obj
        for obj in bpy.data.objects
        if obj.type == "CAMERA"
        and not obj.name.startswith("HEMI_")
        and not obj.name.startswith("CHECK_")
    ]
    return sorted(cameras, key=lambda obj: obj.name)


def median(values):
    values = sorted(values)
    if not values:
        return 1.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def normalized_or_none(vector):
    if vector.length < 1.0e-8:
        return None
    return vector.normalized()


def hemisphere_basis(normal):
    up = Vector((0.0, 0.0, 1.0))
    if abs(normal.dot(up)) > 0.96:
        up = Vector((0.0, 1.0, 0.0))
    u_axis = normalized_or_none(normal.cross(up))
    if u_axis is None:
        u_axis = Vector((1.0, 0.0, 0.0))
    v_axis = normalized_or_none(normal.cross(u_axis))
    if v_axis is None:
        v_axis = Vector((0.0, 1.0, 0.0))
    return u_axis, v_axis


def candidate_name(index, ring, azimuth, scale):
    return f"HEMI_{index:03d}_r{ring:02d}_a{azimuth:02d}_s{scale:.2f}".replace(".", "p")


def nearest_seed(direction, seeds, center):
    best = None
    best_dot = -2.0
    for seed in seeds:
        seed_direction = normalized_or_none(seed.matrix_world.translation - center)
        if seed_direction is None:
            continue
        value = direction.dot(seed_direction)
        if value > best_dot:
            best = seed
            best_dot = value
    return best or seeds[0], best_dot


def look_at(camera, target):
    direction = target - camera.location
    if direction.length < 1.0e-8:
        return
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_camera_candidate(name, seed, position, target):
    camera_data = seed.data.copy()
    camera_data.name = f"{name}_data"
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = position
    look_at(camera, target)
    return camera


def build_candidates(seeds, center, rings, azimuth_samples, max_polar_deg, radius_scales):
    seed_dirs = [normalized_or_none(seed.matrix_world.translation - center) for seed in seeds]
    seed_dirs = [direction for direction in seed_dirs if direction is not None]
    normal = normalized_or_none(sum(seed_dirs, Vector((0.0, 0.0, 0.0))))
    if normal is None:
        normal = seed_dirs[0] if seed_dirs else Vector((0.0, -1.0, 0.0))
    u_axis, v_axis = hemisphere_basis(normal)
    base_radius = median([(seed.matrix_world.translation - center).length for seed in seeds])
    max_theta = math.radians(max_polar_deg)
    radius_scales = radius_scales or [1.0]
    unique = set()
    candidates = []
    index = 0
    for scale in radius_scales:
        radius = base_radius * max(scale, 0.05)
        for ring in range(max(rings, 1) + 1):
            theta = max_theta * ring / max(rings, 1)
            count = 1 if ring == 0 else max(azimuth_samples, 3)
            for azimuth in range(count):
                phi = 0.0 if count == 1 else (2.0 * math.pi * azimuth / count) + (ring * math.pi / count)
                direction = (
                    math.cos(theta) * normal
                    + math.sin(theta) * math.cos(phi) * u_axis
                    + math.sin(theta) * math.sin(phi) * v_axis
                )
                direction = direction.normalized()
                position = center + direction * radius
                key = tuple(round(value, 3) for value in position)
                if key in unique:
                    continue
                unique.add(key)
                seed, seed_dot = nearest_seed(direction, seeds, center)
                name = candidate_name(index, ring, azimuth, scale)
                candidates.append(
                    {
                        "name": name,
                        "ring": ring,
                        "azimuth": azimuth,
                        "radius_scale": scale,
                        "radius": radius,
                        "direction": direction,
                        "position": position,
                        "nearest_seed": seed,
                        "nearest_seed_alignment": seed_dot,
                    }
                )
                index += 1
    return candidates, normal, base_radius


def projection_metrics(scene, camera, vertices, stride):
    sampled = vertices[:: max(stride, 1)]
    front = 0
    inside = 0
    xs = []
    ys = []
    for world in sampled:
        ndc = world_to_camera_view(scene, camera, world)
        if ndc.z <= 0.0:
            continue
        front += 1
        xs.append(ndc.x)
        ys.append(ndc.y)
        if 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0:
            inside += 1
    if not xs:
        return {
            "sampled": len(sampled),
            "front_fraction": 0.0,
            "visible_fraction": 0.0,
            "ndc_min": None,
            "ndc_max": None,
            "bbox_area": 0.0,
            "center_offset": 1.0,
            "frame_score": 100.0,
        }
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    center_offset = math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
    visible_fraction = inside / max(len(sampled), 1)
    front_fraction = front / max(len(sampled), 1)
    crop_penalty = (1.0 - visible_fraction) * 8.0
    front_penalty = (1.0 - front_fraction) * 4.0
    small_penalty = max(0.0, 0.18 - bbox_area) * 3.0
    large_penalty = max(0.0, bbox_area - 0.92) * 3.0
    center_penalty = center_offset * 1.5
    return {
        "sampled": len(sampled),
        "front_fraction": float(front_fraction),
        "visible_fraction": float(visible_fraction),
        "ndc_min": [float(min_x), float(min_y)],
        "ndc_max": [float(max_x), float(max_y)],
        "bbox_area": float(bbox_area),
        "center_offset": float(center_offset),
        "frame_score": float(crop_penalty + front_penalty + small_penalty + large_penalty + center_penalty),
    }


def score_camera(camera, frames, mesh, stride):
    rows = []
    for frame in frames:
        set_frame(frame)
        vertices = projection.evaluated_vertices_world(mesh)
        metrics = projection_metrics(bpy.context.scene, camera, vertices, stride)
        metrics["frame"] = float(frame)
        rows.append(metrics)
    scores = [row["frame_score"] for row in rows]
    visible = [row["visible_fraction"] for row in rows]
    fill = [row["bbox_area"] for row in rows]
    return {
        "frames": rows,
        "mean_score": float(sum(scores) / max(len(scores), 1)),
        "worst_score": float(max(scores) if scores else 100.0),
        "min_visible_fraction": float(min(visible) if visible else 0.0),
        "mean_visible_fraction": float(sum(visible) / max(len(visible), 1)),
        "mean_bbox_area": float(sum(fill) / max(len(fill), 1)),
    }


def angular_distance_deg(a, b):
    dot = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
    return math.degrees(math.acos(dot))


def select_diverse_candidates(scored, count, min_separation_deg):
    selected = []
    for row in scored:
        direction = row["direction"]
        if all(angular_distance_deg(direction, chosen["direction"]) >= min_separation_deg for chosen in selected):
            selected.append(row)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for row in scored:
            if row not in selected:
                selected.append(row)
            if len(selected) >= count:
                break
    return selected


def vector_json(vector):
    return [float(vector.x), float(vector.y), float(vector.z)]


def render_projection_set(out_dir, cameras, frames, action_name, mesh, armature, context_objects, resolution, stride):
    renders = []
    safe_action = (action_name or "no_action").replace(".", "_")
    for frame in frames:
        set_frame(frame)
        mesh_vertices = projection.evaluated_vertices_world(mesh)
        context_vertices = {
            obj.name: projection.evaluated_vertices_world(obj)
            for obj in context_objects
        }
        for camera in cameras:
            safe_camera = camera.name.replace(".", "_")
            filepath = out_dir / f"{safe_action}_f{int(frame):04d}_{safe_camera}_projection.png"
            metrics = projection.render_projection(
                bpy.context.scene,
                camera,
                mesh_vertices,
                context_vertices,
                armature,
                filepath,
                resolution,
                resolution,
                stride,
            )
            metrics.update(
                {
                    "frame": float(frame),
                    "camera": camera.name,
                    "image": str(filepath),
                }
            )
            renders.append(metrics)
    return renders


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action = set_action(args.armature, args.action)
    armature = bpy.data.objects.get(args.armature)
    mesh = bpy.data.objects.get(args.mesh)
    if armature is None or mesh is None:
        raise RuntimeError("Armature or mesh not found.")
    frames = args.frame or [float(bpy.context.scene.frame_start)]
    set_frame(frames[0])

    context_names = args.context_object or DEFAULT_CONTEXT_OBJECTS
    context_objects = [
        bpy.data.objects[name]
        for name in context_names
        if name in bpy.data.objects and bpy.data.objects[name].type == "MESH"
    ]

    seeds = choose_seed_cameras(args.camera)
    if not seeds:
        raise RuntimeError("No seed cameras found.")

    fallback_center, fallback_extent = bbox_center([mesh] + context_objects)
    focus = estimate_camera_focus(seeds)
    if focus is None or (focus - fallback_center).length > fallback_extent * 5.0:
        focus = fallback_center

    candidate_specs, hemisphere_normal, base_radius = build_candidates(
        seeds,
        focus,
        args.rings,
        args.azimuth_samples,
        args.max_polar_deg,
        args.radius_scale,
    )

    scored = []
    candidate_cameras = []
    for spec in candidate_specs:
        camera = create_camera_candidate(
            spec["name"],
            spec["nearest_seed"],
            spec["position"],
            focus,
        )
        candidate_cameras.append(camera)
        score = score_camera(camera, frames, mesh, args.vertex_stride)
        row = {
            **spec,
            **score,
            "camera": camera,
        }
        scored.append(row)

    scored.sort(key=lambda row: (row["worst_score"], row["mean_score"], -row["min_visible_fraction"]))
    selected = select_diverse_candidates(
        scored,
        max(args.render_top, 0),
        args.selection_separation_deg,
    )
    render_cameras = [row["camera"] for row in selected]
    if args.include_seed_projections:
        render_cameras = seeds + render_cameras

    renders = render_projection_set(
        out_dir,
        render_cameras,
        frames,
        action.name if action else args.action,
        mesh,
        armature,
        context_objects,
        args.resolution,
        args.vertex_stride,
    )

    manifest = {
        "file": bpy.data.filepath,
        "action": action.name if action else args.action,
        "armature": armature.name,
        "mesh": mesh.name,
        "frames": [float(frame) for frame in frames],
        "seed_cameras": [
            {
                "name": camera.name,
                "location": vector_json(camera.matrix_world.translation),
                "forward": vector_json(camera_forward(camera)),
            }
            for camera in seeds
        ],
        "focus": vector_json(focus),
        "hemisphere_normal": vector_json(hemisphere_normal),
        "base_radius": float(base_radius),
        "candidate_count": len(scored),
        "selected_count": len(selected),
        "candidates": [
            {
                "name": row["name"],
                "ring": int(row["ring"]),
                "azimuth": int(row["azimuth"]),
                "radius_scale": float(row["radius_scale"]),
                "radius": float(row["radius"]),
                "location": vector_json(row["position"]),
                "direction": vector_json(row["direction"]),
                "nearest_seed": row["nearest_seed"].name,
                "nearest_seed_alignment": float(row["nearest_seed_alignment"]),
                "mean_score": row["mean_score"],
                "worst_score": row["worst_score"],
                "min_visible_fraction": row["min_visible_fraction"],
                "mean_visible_fraction": row["mean_visible_fraction"],
                "mean_bbox_area": row["mean_bbox_area"],
                "selected_for_projection": row in selected,
                "frames": row["frames"],
            }
            for row in scored
        ],
        "renders": renders,
    }
    manifest_path = out_dir / "hemisphere_camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_blend = None
    if args.save_blend:
        save_blend = Path(args.save_blend)
        save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend.resolve()))

    print("MOUSE_GEO_FORGE_HEMISPHERE_CAMERAS_BEGIN")
    print(
        json.dumps(
            {
                "seed_count": len(seeds),
                "candidate_count": len(scored),
                "selected_count": len(selected),
                "render_count": len(renders),
                "manifest": str(manifest_path),
                "save_blend": str(save_blend) if save_blend else None,
                "best": [
                    {
                        "name": row["name"],
                        "worst_score": row["worst_score"],
                        "min_visible_fraction": row["min_visible_fraction"],
                        "mean_bbox_area": row["mean_bbox_area"],
                    }
                    for row in scored[: min(5, len(scored))]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_HEMISPHERE_CAMERAS_END")


if __name__ == "__main__":
    main()

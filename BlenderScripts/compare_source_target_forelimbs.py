import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontlimb_quality_check as qc
import migrate_animations_to_ik as mig
import probe_forelimb_interpolation as interp
import software_projection_check as projection
import source_mesh_match_refine as smrefine


DEFAULT_CONTEXT_OBJECTS = [
    "Part_PrintedMouseCup",
    "Part_MouseCupSide.001",
    "Part_MouseCupSide.002",
    "Part_WaterPipe",
]

DEFAULT_CAMERAS = [
    "HEMI_004_r01_a03_s1p00",
    "HEMI_000_r00_a00_s1p00",
    "HEMI_003_r01_a02_s1p00",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare target IK forelimb mesh against the source hand-animated mesh."
    )
    parser.add_argument("--source-blend", required=True)
    parser.add_argument("--target-blend", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--source-mesh", default="MS_Mouse")
    parser.add_argument("--source-action", default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--quality-json", default=None)
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--context-object", action="append", default=[])
    parser.add_argument("--top-worst", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--vertex-stride", type=int, default=2)
    parser.add_argument("--arm-volume-bins", type=int, default=6)
    parser.add_argument("--arm-volume-trim", type=float, default=0.12)
    return parser.parse_args(args_after_dash())


def choose_frames(args):
    frames = {float(frame) for frame in args.frame}
    if args.quality_json:
        quality = json.loads(Path(args.quality_json).read_text(encoding="utf-8"))
        action = next(row for row in quality["actions"] if row["action"] == args.action)
        for key in ("worst_collapse", "worst_arm_volume", "worst_clearance"):
            for row in action.get(key, [])[: max(args.top_worst, 0)]:
                if "frame" in row:
                    frames.add(float(row["frame"]))
        for row in action.get("warning_frames", []):
            if "frame" in row:
                frames.add(float(row["frame"]))
    if not frames:
        frames.add(float(bpy.context.scene.frame_start))
    return sorted(frames)


def set_action(armature, action_name):
    armature.animation_data_create()
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Action {action_name!r} not found.")
        armature.animation_data.action = action
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    actions = list(bpy.data.actions)
    if not actions:
        raise RuntimeError("No action found.")
    action = sorted(actions, key=lambda item: item.name)[0]
    armature.animation_data.action = action
    return action


def collect_source(args, frames):
    print(f"[source] opening {args.source_blend}")
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.source_blend).resolve()))
    armature = mig.find_armature(args.source_armature, require_action=False)
    mesh = bpy.data.objects.get(args.source_mesh)
    if mesh is None:
        raise RuntimeError(f"Source mesh {args.source_mesh!r} not found.")
    action = set_action(armature, args.source_action)

    bone_names = sorted(
        set(mig.BODY_TAIL_ALIGNMENT_BONES)
        | set(mig.HEAD_REFERENCE_BONES)
        | {f"Scapula_{side}" for side in ("L", "R")}
        | {f"Wrist_{side}" for side in ("L", "R")}
        | {f"Fingers1_{side}" for side in ("L", "R")}
        | {
            f"{prefix}_{side}"
            for side in ("L", "R")
            for prefix in ("IndexFinger3", "MiddleFinger3", "RingFinger3", "PinkyFinger3")
        }
    )
    payload = {
        "blend": str(Path(args.source_blend).resolve()),
        "mesh_vertex_count": len(mesh.data.vertices),
        "action": action.name,
        "frames": {},
    }
    for index, frame in enumerate(frames, start=1):
        qc.set_frame(frame)
        vertices, _polygons = qc.evaluated_mesh_world(mesh)
        heads, tails = smrefine.source_bone_points_world(armature, bone_names)
        payload["frames"][int(round(frame))] = {
            "vertices": vertices,
            "heads": heads,
            "tails": tails,
        }
        if index == 1 or index == len(frames) or index % 10 == 0:
            print(f"  source frame {index}/{len(frames)}")
    return payload


def selected_indices(arm_indices, paw_indices, side, include_paw):
    indices = set(arm_indices[side]["upper"]) | set(arm_indices[side]["lower"])
    if include_paw:
        indices |= set(paw_indices[side])
    return sorted(indices)


def aligned_source_vertices(source_frame, alignment):
    return [alignment @ vertex for vertex in source_frame["vertices"]]


def stats_for_distances(distances, diagonal):
    if not distances:
        return {
            "count": 0,
            "rms": None,
            "nrmse": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(distances)
    squared = sum(value * value for value in distances)
    p95_index = min(int(round((len(ordered) - 1) * 0.95)), len(ordered) - 1)
    rms = math.sqrt(squared / len(distances))
    return {
        "count": len(distances),
        "rms": float(rms),
        "nrmse": float(rms / max(diagonal, 1.0e-8)),
        "mean": float(sum(distances) / len(distances)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def point_error(target_vertices, source_vertices, indices):
    source_points = []
    distances = []
    for index in indices:
        if index >= len(target_vertices) or index >= len(source_vertices):
            continue
        source = source_vertices[index]
        source_points.append(source)
        distances.append((target_vertices[index] - source).length)
    diagonal = smrefine.bbox_diagonal(source_points)
    return stats_for_distances(distances, diagonal)


def chain_features(shoulder, elbow, palm):
    upper = elbow - shoulder
    lower = palm - elbow
    reach = palm - shoulder
    chain = upper.length + lower.length
    if chain < 1.0e-8:
        return {"available": False}
    angle = qc.angle_degrees(shoulder - elbow, palm - elbow)
    return {
        "available": True,
        "shoulder": [float(value) for value in shoulder],
        "elbow": [float(value) for value in elbow],
        "palm": [float(value) for value in palm],
        "upper_length": float(upper.length),
        "lower_length": float(lower.length),
        "reach": float(reach.length),
        "reach_ratio": float(reach.length / chain),
        "elbow_angle_deg": None if angle is None else float(angle),
    }


def source_chain(source_frame, alignment, side):
    shoulder = source_frame["heads"].get(f"Scapula_{side}")
    elbow = source_frame["tails"].get(f"Scapula_{side}")
    palm = source_frame["tails"].get(f"Wrist_{side}")
    if shoulder is None or elbow is None or palm is None:
        return {"available": False}
    return chain_features(alignment @ shoulder, alignment @ elbow, alignment @ palm)


def target_chain(armature, side):
    shoulder = qc.pose_bone_head_world(armature, f"Scapula_{side}")
    elbow = qc.pose_bone_tail_world(armature, f"Scapula_{side}")
    palm = qc.pose_bone_tail_world(armature, f"Wrist_{side}")
    if shoulder is None or elbow is None or palm is None:
        return {"available": False}
    return chain_features(shoulder, elbow, palm)


def volume_for_chain(vertices, indices, chain, bins, trim):
    if not chain.get("available"):
        return {"available": False}
    shoulder = Vector(chain["shoulder"])
    elbow = Vector(chain["elbow"])
    palm = Vector(chain["palm"])
    upper = qc.volume_proxy_for_segment(vertices, indices["upper"], shoulder, elbow, bins, trim)
    lower = qc.volume_proxy_for_segment(vertices, indices["lower"], elbow, palm, bins, trim)
    return {
        "available": bool(upper.get("available") or lower.get("available")),
        "upper": upper,
        "lower": lower,
    }


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def draw_chain(scene, camera, pixels, width, height, points, color):
    if not points:
        return
    projected = []
    for point in points:
        ndc, x, y = projection.project(scene, camera, point, width, height)
        projected.append((ndc, x, y))
        if ndc.z > 0.0 and -0.05 <= ndc.x <= 1.05 and -0.05 <= ndc.y <= 1.05:
            projection.draw_disc(pixels, width, height, x, y, 4, color)
    for first, second in zip(projected, projected[1:]):
        if first[0].z > 0.0 or second[0].z > 0.0:
            projection.draw_line(
                pixels,
                width,
                height,
                first[1],
                first[2],
                second[1],
                second[2],
                color,
            )


def render_overlay(
    filepath,
    scene,
    camera,
    target_vertices,
    source_vertices,
    target_chain_points,
    source_chain_points,
    context_vertices,
    indices_by_side,
    width,
    height,
    stride,
):
    pixels = projection.blank_pixels(width, height)
    palette = [
        (0.72, 0.72, 0.72, 1.0),
        (0.05, 0.62, 0.92, 1.0),
        (0.95, 0.65, 0.05, 1.0),
        (0.34, 0.72, 0.18, 1.0),
    ]
    for index, vertices in enumerate(context_vertices.values()):
        projection.draw_projected_vertices(
            scene,
            camera,
            vertices,
            pixels,
            width,
            height,
            max(stride, 3),
            palette[index % len(palette)],
            1,
        )
    projection.draw_projected_vertices(
        scene,
        camera,
        target_vertices,
        pixels,
        width,
        height,
        max(stride * 3, 6),
        (0.1, 0.1, 0.1, 0.35),
        1,
    )
    for side, indices in indices_by_side.items():
        source_points = [source_vertices[index] for index in indices if index < len(source_vertices)]
        target_points = [target_vertices[index] for index in indices if index < len(target_vertices)]
        source_color = (0.0, 0.72, 0.22, 0.9) if side == "L" else (0.0, 0.55, 0.95, 0.9)
        target_color = (0.92, 0.12, 0.05, 0.9) if side == "L" else (0.86, 0.2, 0.78, 0.9)
        projection.draw_projected_vertices(
            scene, camera, source_points, pixels, width, height, max(stride, 1), source_color, 2
        )
        projection.draw_projected_vertices(
            scene, camera, target_points, pixels, width, height, max(stride, 1), target_color, 2
        )
        draw_chain(scene, camera, pixels, width, height, source_chain_points[side], source_color)
        draw_chain(scene, camera, pixels, width, height, target_chain_points[side], target_color)
    projection.write_image(filepath, width, height, pixels)


def chain_points_from_record(record):
    if not record.get("available"):
        return []
    return [Vector(record["shoulder"]), Vector(record["elbow"]), Vector(record["palm"])]


def main():
    args = parse_args()
    frames = choose_frames(args)
    source_payload = collect_source(args, frames)

    print(f"[target] opening {args.target_blend}")
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.target_blend).resolve()))
    armature = bpy.data.objects.get(args.armature)
    mesh = bpy.data.objects.get(args.mesh)
    if armature is None or mesh is None:
        raise RuntimeError("Target armature or mesh not found.")
    action = bpy.data.actions.get(args.action)
    if action is None:
        raise RuntimeError(f"Target action {args.action!r} not found.")
    qc.set_action(armature, action)

    if len(mesh.data.vertices) != source_payload["mesh_vertex_count"]:
        raise RuntimeError(
            f"Source/target topology mismatch: {source_payload['mesh_vertex_count']} vs {len(mesh.data.vertices)}"
        )

    arm_indices, _arm_groups = qc.collect_arm_vertex_indices(mesh, 0.001)
    paw_indices, _paw_groups = qc.collect_paw_vertex_indices(mesh, False, 0.001)
    context_names = args.context_object or DEFAULT_CONTEXT_OBJECTS
    context_objects = [
        bpy.data.objects[name]
        for name in context_names
        if name in bpy.data.objects and bpy.data.objects[name].type == "MESH"
    ]
    camera_names = args.camera or DEFAULT_CAMERAS
    cameras = [
        bpy.data.objects[name]
        for name in camera_names
        if name in bpy.data.objects and bpy.data.objects[name].type == "CAMERA"
    ]
    if not cameras:
        raise RuntimeError("No overlay cameras available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    render_rows = []
    for frame in frames:
        print(f"[compare] frame={frame}")
        qc.set_frame(frame)
        frame_key = int(round(frame))
        source_frame = source_payload["frames"].get(frame_key)
        if source_frame is None:
            continue

        target_vertices, _target_polygons = qc.evaluated_mesh_world(mesh)
        target_heads = smrefine.target_body_heads(armature)
        alignment, alignment_scale, alignment_count = smrefine.alignment_from_source_to_target(
            source_frame,
            target_heads,
            allow_scale=False,
        )
        source_vertices = aligned_source_vertices(source_frame, alignment)
        context_vertices = {
            obj.name: projection.evaluated_vertices_world(obj)
            for obj in context_objects
        }

        source_chain_by_side = {}
        target_chain_by_side = {}
        target_chain_points = {}
        source_chain_points = {}
        indices_by_side = {}
        side_rows = {}
        for side in ("L", "R"):
            arm_only = selected_indices(arm_indices, paw_indices, side, include_paw=False)
            arm_and_paw = selected_indices(arm_indices, paw_indices, side, include_paw=True)
            indices_by_side[side] = arm_and_paw
            source_chain_by_side[side] = source_chain(source_frame, alignment, side)
            target_chain_by_side[side] = target_chain(armature, side)
            source_chain_points[side] = chain_points_from_record(source_chain_by_side[side])
            target_chain_points[side] = chain_points_from_record(target_chain_by_side[side])

            source_volume = volume_for_chain(
                source_vertices,
                arm_indices[side],
                source_chain_by_side[side],
                args.arm_volume_bins,
                args.arm_volume_trim,
            )
            target_volume = volume_for_chain(
                target_vertices,
                arm_indices[side],
                target_chain_by_side[side],
                args.arm_volume_bins,
                args.arm_volume_trim,
            )
            upper_ratio = safe_ratio(
                target_volume.get("upper", {}).get("volume_proxy"),
                source_volume.get("upper", {}).get("volume_proxy"),
            )
            lower_ratio = safe_ratio(
                target_volume.get("lower", {}).get("volume_proxy"),
                source_volume.get("lower", {}).get("volume_proxy"),
            )
            reach_delta = None
            elbow_delta = None
            if source_chain_by_side[side].get("available") and target_chain_by_side[side].get("available"):
                reach_delta = (
                    target_chain_by_side[side]["reach_ratio"]
                    - source_chain_by_side[side]["reach_ratio"]
                )
                if (
                    source_chain_by_side[side].get("elbow_angle_deg") is not None
                    and target_chain_by_side[side].get("elbow_angle_deg") is not None
                ):
                    elbow_delta = (
                        target_chain_by_side[side]["elbow_angle_deg"]
                        - source_chain_by_side[side]["elbow_angle_deg"]
                    )
            side_rows[side] = {
                "arm_error": point_error(target_vertices, source_vertices, arm_only),
                "arm_paw_error": point_error(target_vertices, source_vertices, arm_and_paw),
                "source_chain": source_chain_by_side[side],
                "target_chain": target_chain_by_side[side],
                "reach_ratio_delta": None if reach_delta is None else float(reach_delta),
                "elbow_angle_delta_deg": None if elbow_delta is None else float(elbow_delta),
                "source_volume": source_volume,
                "target_volume": target_volume,
                "target_over_source_volume_ratio": {
                    "upper": upper_ratio,
                    "lower": lower_ratio,
                    "worst": min(
                        [value for value in (upper_ratio, lower_ratio) if value is not None],
                        default=None,
                    ),
                },
            }

        safe_action = args.action.replace(".", "_")
        renders = []
        for camera in cameras:
            safe_camera = camera.name.replace(".", "_")
            filepath = out_dir / f"{safe_action}_f{int(round(frame)):04d}_{safe_camera}_source_target_overlay.png"
            render_overlay(
                filepath,
                bpy.context.scene,
                camera,
                target_vertices,
                source_vertices,
                target_chain_points,
                source_chain_points,
                context_vertices,
                indices_by_side,
                args.resolution,
                args.resolution,
                args.vertex_stride,
            )
            renders.append(str(filepath))
            render_rows.append({"frame": float(frame), "camera": camera.name, "image": str(filepath)})

        rows.append(
            {
                "frame": float(frame),
                "alignment_scale": float(alignment_scale),
                "alignment_point_count": int(alignment_count),
                "sides": side_rows,
                "renders": renders,
            }
        )

    worst = []
    for row in rows:
        for side, side_row in row["sides"].items():
            worst.append(
                {
                    "frame": row["frame"],
                    "side": side,
                    "arm_nrmse": side_row["arm_error"]["nrmse"],
                    "arm_paw_nrmse": side_row["arm_paw_error"]["nrmse"],
                    "reach_ratio_delta": side_row["reach_ratio_delta"],
                    "elbow_angle_delta_deg": side_row["elbow_angle_delta_deg"],
                    "volume_ratio_worst": side_row["target_over_source_volume_ratio"]["worst"],
                }
            )
    worst.sort(
        key=lambda item: (
            item["arm_paw_nrmse"] is None,
            -(item["arm_paw_nrmse"] or 0.0),
            item["frame"],
            item["side"],
        )
    )

    report = {
        "source": {
            "blend": source_payload["blend"],
            "action": source_payload["action"],
        },
        "target": {
            "blend": str(Path(args.target_blend).resolve()),
            "action": action.name,
            "armature": armature.name,
            "mesh": mesh.name,
        },
        "settings": vars(args),
        "frames": [float(frame) for frame in frames],
        "rows": rows,
        "worst_by_arm_paw_nrmse": worst[:20],
        "renders": render_rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("MOUSE_GEO_FORGE_SOURCE_TARGET_COMPARE_BEGIN")
    print(
        json.dumps(
            {
                "frame_count": len(frames),
                "row_count": len(rows),
                "render_count": len(render_rows),
                "report": str(report_path),
                "worst": worst[:8],
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_SOURCE_TARGET_COMPARE_END")


if __name__ == "__main__":
    main()

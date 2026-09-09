"""Audit local mesh waviness against a source animation with matching topology."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_v3_self_collision as selfaudit  # noqa: E402
from fit_v3_group_to_source_mesh import bbox_center_object  # noqa: E402


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--source-action", required=True)
    parser.add_argument("--target-blend", type=Path, required=True)
    parser.add_argument("--target-action", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--target-armature", default="\u8001\u9f20_rig")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--source-cup", default="Part_PrintedMouseCup")
    parser.add_argument("--target-cup", default="Part_PrintedMouseCup")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame", type=int, action="append", default=[])
    parser.add_argument("--min-weight", type=float, default=0.001)
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args(args_after_dash())


def set_frame(frame: int) -> None:
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()


def action_range(action, scene, start, end, step: int, explicit: list[int]) -> list[int]:
    if explicit:
        return sorted({int(frame) for frame in explicit})
    a_start, a_end = action.frame_range
    first = int(math.floor(a_start)) if start is None else int(start)
    last = int(math.ceil(a_end)) if end is None else int(end)
    if last < first:
        first, last = int(scene.frame_start), int(scene.frame_end)
    return list(range(first, last + 1, max(int(step), 1)))


def vec_json(vector: Vector, digits: int = 5) -> list[float]:
    return [round(float(value), digits) for value in vector]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def evaluated_mesh_payload(obj) -> tuple[list[Vector], list[Vector], list[list[int]]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        vertices = [eval_obj.matrix_world @ vertex.co for vertex in mesh.vertices]
        normals = [(eval_obj.matrix_world.to_3x3() @ vertex.normal).normalized() for vertex in mesh.vertices]
        polygons = [[int(index) for index in poly.vertices] for poly in mesh.polygons]
        return vertices, normals, polygons
    finally:
        eval_obj.to_mesh_clear()


def build_adjacency(polygons: list[list[int]], vertex_count: int) -> list[list[int]]:
    neighbors = [set() for _ in range(vertex_count)]
    for poly in polygons:
        count = len(poly)
        for index, vertex in enumerate(poly):
            prev_vertex = poly[(index - 1) % count]
            next_vertex = poly[(index + 1) % count]
            if prev_vertex != vertex:
                neighbors[vertex].add(prev_vertex)
            if next_vertex != vertex:
                neighbors[vertex].add(next_vertex)
    return [sorted(row) for row in neighbors]


def laplacian_roughness(vertices: list[Vector], normals: list[Vector], adjacency: list[list[int]]) -> list[dict]:
    rows = []
    for index, vertex in enumerate(vertices):
        neighbors = adjacency[index]
        if not neighbors:
            rows.append({"normal_lap": 0.0, "lap_len": 0.0, "mean_edge": 0.0})
            continue
        average = Vector((0.0, 0.0, 0.0))
        edge_sum = 0.0
        for neighbor in neighbors:
            average += vertices[neighbor]
            edge_sum += (vertices[neighbor] - vertex).length
        average /= len(neighbors)
        lap = vertex - average
        normal = normals[index]
        normal_lap = abs(lap.dot(normal))
        rows.append(
            {
                "normal_lap": float(normal_lap),
                "lap_len": float(lap.length),
                "mean_edge": float(edge_sum / len(neighbors)),
            }
        )
    return rows


def high_frequency_displacement(
    source_rel: list[Vector],
    target_rel: list[Vector],
    target_normals: list[Vector],
    adjacency: list[list[int]],
) -> list[dict]:
    displacements = [target - source for source, target in zip(source_rel, target_rel)]
    rows = []
    for index, displacement in enumerate(displacements):
        neighbors = adjacency[index]
        if not neighbors:
            residual = Vector((0.0, 0.0, 0.0))
        else:
            average = Vector((0.0, 0.0, 0.0))
            for neighbor in neighbors:
                average += displacements[neighbor]
            average /= len(neighbors)
            residual = displacement - average
        normal = target_normals[index]
        rows.append(
            {
                "hf_disp": float(residual.length),
                "hf_disp_normal": float(abs(residual.dot(normal))),
                "disp_len": float(displacement.length),
            }
        )
    return rows


def open_scene(path: Path, armature_name: str, mesh_name: str, cup_name: str, action_name: str):
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    rig = bpy.data.objects.get(armature_name)
    mesh = bpy.data.objects.get(mesh_name)
    cup = bpy.data.objects.get(cup_name)
    action = bpy.data.actions.get(action_name)
    if rig is None or mesh is None or cup is None or action is None:
        raise RuntimeError(f"Missing rig/mesh/cup/action in {path}")
    rig.animation_data_create()
    rig.animation_data.action = action
    return rig, mesh, cup, action


def vertex_groups_for(mesh_obj, vertex_index: int, limit: int = 6) -> list[dict]:
    groups = mesh_obj.vertex_groups
    memberships = []
    for membership in mesh_obj.data.vertices[vertex_index].groups:
        group = groups[membership.group]
        memberships.append({"group": group.name, "weight": round(float(membership.weight), 5)})
    memberships.sort(key=lambda item: item["weight"], reverse=True)
    return memberships[:limit]


def collect_payload(path, armature, mesh_name, cup_name, action_name, frames):
    _rig, mesh, cup, _action = open_scene(path, armature, mesh_name, cup_name, action_name)
    payload = {"frames": {}, "vertex_count": len(mesh.data.vertices), "mesh_groups": [g.name for g in mesh.vertex_groups]}
    polygons_for_adjacency = None
    for index, frame in enumerate(frames, start=1):
        set_frame(frame)
        vertices, normals, polygons = evaluated_mesh_payload(mesh)
        cup_center = bbox_center_object(cup)
        payload["frames"][frame] = {
            "vertices": vertices,
            "vertices_rel": [vertex - cup_center for vertex in vertices],
            "normals": normals,
        }
        if polygons_for_adjacency is None:
            polygons_for_adjacency = polygons
        if index == 1 or index == len(frames) or index % 50 == 0:
            print(f"[collect] {Path(path).name} {index}/{len(frames)} frame={frame}")
    payload["polygons"] = polygons_for_adjacency or []
    return payload


def filter_values(rows: list[dict], indices: list[int], key: str) -> list[float]:
    return [float(rows[index][key]) for index in indices if index < len(rows)]


def main() -> None:
    args = parse_args()
    _rig, mesh, _cup, target_action = open_scene(
        args.target_blend, args.target_armature, args.mesh, args.target_cup, args.target_action
    )
    frames = action_range(target_action, bpy.context.scene, args.frame_start, args.frame_end, args.frame_step, args.frame)
    part_specs = selfaudit.build_part_specs(mesh, args.min_weight)
    target_group_names = [g.name for g in mesh.vertex_groups]

    source_payload = collect_payload(
        args.source_blend, args.source_armature, args.mesh, args.source_cup, args.source_action, frames
    )
    target_payload = collect_payload(
        args.target_blend, args.target_armature, args.mesh, args.target_cup, args.target_action, frames
    )
    if source_payload["vertex_count"] != target_payload["vertex_count"]:
        raise RuntimeError("Source and target topology mismatch.")
    adjacency = build_adjacency(source_payload["polygons"], source_payload["vertex_count"])

    rows = []
    top_rows = []
    part_accum: dict[str, dict[str, list[float]]] = {
        name: {"rough_delta": [], "hf_disp": [], "hf_disp_normal": [], "target_normal_lap": [], "source_normal_lap": []}
        for name in part_specs
    }
    for frame in frames:
        source_frame = source_payload["frames"][frame]
        target_frame = target_payload["frames"][frame]
        source_rough = laplacian_roughness(source_frame["vertices_rel"], source_frame["normals"], adjacency)
        target_rough = laplacian_roughness(target_frame["vertices_rel"], target_frame["normals"], adjacency)
        hf = high_frequency_displacement(
            source_frame["vertices_rel"], target_frame["vertices_rel"], target_frame["normals"], adjacency
        )
        for part, spec in part_specs.items():
            indices = [int(index) for index in spec["indices"]]
            rough_delta = [
                max(0.0, target_rough[index]["normal_lap"] - source_rough[index]["normal_lap"])
                for index in indices
                if index < len(target_rough)
            ]
            part_accum[part]["rough_delta"].extend(rough_delta)
            for key in ("hf_disp", "hf_disp_normal"):
                part_accum[part][key].extend(filter_values(hf, indices, key))
            part_accum[part]["target_normal_lap"].extend(filter_values(target_rough, indices, "normal_lap"))
            part_accum[part]["source_normal_lap"].extend(filter_values(source_rough, indices, "normal_lap"))
            rows.append(
                {
                    "frame": int(frame),
                    "part": part,
                    "vertex_count": len(indices),
                    "rough_delta": summarize_values(rough_delta),
                    "hf_disp": summarize_values(filter_values(hf, indices, "hf_disp")),
                    "hf_disp_normal": summarize_values(filter_values(hf, indices, "hf_disp_normal")),
                    "target_normal_lap": summarize_values(filter_values(target_rough, indices, "normal_lap")),
                    "source_normal_lap": summarize_values(filter_values(source_rough, indices, "normal_lap")),
                }
            )
        for vertex_index in range(source_payload["vertex_count"]):
            rough_delta = max(
                0.0, target_rough[vertex_index]["normal_lap"] - source_rough[vertex_index]["normal_lap"]
            )
            score = max(rough_delta, hf[vertex_index]["hf_disp_normal"])
            if score <= 0.0:
                continue
            top_rows.append(
                {
                    "frame": int(frame),
                    "vertex": int(vertex_index),
                    "score": float(score),
                    "rough_delta": float(rough_delta),
                    "hf_disp": float(hf[vertex_index]["hf_disp"]),
                    "hf_disp_normal": float(hf[vertex_index]["hf_disp_normal"]),
                    "target_normal_lap": float(target_rough[vertex_index]["normal_lap"]),
                    "source_normal_lap": float(source_rough[vertex_index]["normal_lap"]),
                    "target_rel": vec_json(target_frame["vertices_rel"][vertex_index]),
                }
            )

    top_rows.sort(key=lambda item: item["score"], reverse=True)
    # Reopen target once for vertex-group labels without retaining evaluated mesh objects.
    _rig, target_mesh, _cup, _action = open_scene(
        args.target_blend, args.target_armature, args.mesh, args.target_cup, args.target_action
    )
    for item in top_rows[: args.top]:
        item["target_groups"] = vertex_groups_for(target_mesh, item["vertex"])

    part_summary = {
        part: {key: summarize_values(values) for key, values in accum.items()} for part, accum in part_accum.items()
    }
    report = {
        "source_blend": str(args.source_blend.resolve()),
        "source_action": args.source_action,
        "target_blend": str(args.target_blend.resolve()),
        "target_action": args.target_action,
        "frames": [int(frame) for frame in frames],
        "settings": {
            "frame_step": args.frame_step,
            "min_weight": args.min_weight,
            "metric_notes": {
                "rough_delta": "positive increase of target normal-direction Laplacian over source",
                "hf_disp": "local high-frequency residual of target-source displacement",
                "hf_disp_normal": "normal-direction component of high-frequency displacement",
            },
        },
        "target_group_count": len(target_group_names),
        "part_summary": part_summary,
        "rows": rows,
        "top_vertices": top_rows[: args.top],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(frames)}, indent=2))


if __name__ == "__main__":
    main()

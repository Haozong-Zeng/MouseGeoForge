"""Sweep constant local-Y forelimb length/scale on the old IK WaterReaching rig."""

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

import frontlimb_quality_check as fq  # noqa: E402


FORELIMB_SCALE_BONES = ("Scapula_L", "Scapula_R")
FORELIMB_CHAIN_BASES = ("Scapula", "Wrist")
HEAD_GROUP_PREFIXES = ("Head", "Jaw", "Ear", "Eye", "Mustache", "Neck")
TORSO_GROUP_PREFIXES = ("Root", "RootPart", "Spine", "Chest", "Neck", "Head", "Jaw")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default=None)
    parser.add_argument("--frame", type=float, action="append", default=[166.0, 239.0])
    parser.add_argument(
        "--scale",
        type=float,
        action="append",
        default=[],
        help="Candidate local-Y scale. Defaults to 1.00..1.25 sweep.",
    )
    parser.add_argument("--selected-scale", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=("rest-length", "pose-scale"),
        default="rest-length",
        help="rest-length edits bone head-tail length; pose-scale only changes PoseBone.scale.y.",
    )
    parser.add_argument("--min-weight", type=float, default=0.001)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args(args_after_dash())


def default_scales() -> list[float]:
    return [1.00, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.22, 1.25]


def action_fcurves(action) -> list:
    fcurves = []
    if getattr(action, "fcurves", None):
        fcurves.extend(action.fcurves)
    if getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def choose_action(rig, requested: str | None):
    if requested:
        action = bpy.data.actions.get(requested)
        if action is None:
            raise RuntimeError(f"Missing action {requested!r}")
        return action
    if rig.animation_data and rig.animation_data.action:
        return rig.animation_data.action
    actions = sorted([action for action in bpy.data.actions if action.name.startswith("MIG_")], key=lambda a: a.name)
    if not actions:
        raise RuntimeError("No action found.")
    return actions[0]


def set_action(rig, action):
    rig.animation_data_create()
    rig.animation_data.action = action


def set_frame(frame: float):
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def apply_forelimb_y_scale(rig, scale_y: float):
    for bone_name in FORELIMB_SCALE_BONES:
        pb = rig.pose.bones.get(bone_name)
        if pb is None:
            continue
        pb.scale = (1.0, scale_y, 1.0)
        for constraint in pb.constraints:
            if constraint.type != "LIMIT_SCALE":
                continue
            constraint.use_min_x = True
            constraint.use_min_y = True
            constraint.use_min_z = True
            constraint.use_max_x = True
            constraint.use_max_y = True
            constraint.use_max_z = True
            constraint.min_x = 1.0
            constraint.max_x = 1.0
            constraint.min_y = scale_y
            constraint.max_y = scale_y
            constraint.min_z = 1.0
            constraint.max_z = 1.0
            constraint.owner_space = "LOCAL"
    bpy.context.view_layer.update()


def descendants_by_parent(rig) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {bone.name: [] for bone in rig.data.bones}
    for bone in rig.data.bones:
        if bone.parent:
            children.setdefault(bone.parent.name, []).append(bone.name)
    return children


def collect_descendants(root: str, children: dict[str, list[str]]) -> list[str]:
    out = []
    stack = list(children.get(root, []))
    while stack:
        name = stack.pop(0)
        out.append(name)
        stack.extend(children.get(name, []))
    return out


def apply_forelimb_rest_y_length(rig, scale_y: float):
    """Scale selected forelimb rest lengths and translate downstream chains.

    Blender bones use local Y along head->tail.  We preserve each descendant's
    original orientation/length unless the descendant itself is explicitly
    listed in FORELIMB_SCALE_BONES.
    """

    children = descendants_by_parent(rig)
    original = {}
    connect = {}
    parent = {}
    for bone in rig.data.bones:
        original[bone.name] = {
            "head": bone.head_local.copy(),
            "tail": bone.tail_local.copy(),
        }
        connect[bone.name] = bool(bone.use_connect)
        parent[bone.name] = bone.parent.name if bone.parent else None

    new_positions = {
        name: {"head": row["head"].copy(), "tail": row["tail"].copy()}
        for name, row in original.items()
    }

    scale_targets = set(FORELIMB_SCALE_BONES)
    for side in ("L", "R"):
        accumulated = Vector((0.0, 0.0, 0.0))
        chain = [f"{base}_{side}" for base in FORELIMB_CHAIN_BASES]
        for index, bone_name in enumerate(chain):
            if bone_name not in original:
                continue
            if bone_name not in scale_targets:
                continue
            old_head = original[bone_name]["head"]
            old_tail = original[bone_name]["tail"]
            old_vec = old_tail - old_head
            new_head = old_head + accumulated
            new_tail = new_head + old_vec * scale_y
            extension = new_tail - (old_tail + accumulated)
            new_positions[bone_name] = {"head": new_head, "tail": new_tail}
            later_scaled = set(chain[index + 1 :]).intersection(scale_targets)
            for descendant in collect_descendants(bone_name, children):
                if descendant in later_scaled:
                    continue
                new_positions[descendant]["head"] += extension
                new_positions[descendant]["tail"] += extension
            accumulated += extension

    impacted = set()
    for root in FORELIMB_SCALE_BONES:
        if root in original:
            impacted.add(root)
            impacted.update(collect_descendants(root, children))

    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for name in impacted:
            if name in rig.data.edit_bones:
                rig.data.edit_bones[name].use_connect = False
        for name in impacted:
            eb = rig.data.edit_bones.get(name)
            if eb is None:
                continue
            eb.head = new_positions[name]["head"]
            eb.tail = new_positions[name]["tail"]
        for name in impacted:
            eb = rig.data.edit_bones.get(name)
            if eb is None:
                continue
            parent_name = parent.get(name)
            if connect.get(name) and parent_name in new_positions:
                parent_tail = new_positions[parent_name]["tail"]
                if (new_positions[name]["head"] - parent_tail).length < 1.0e-5:
                    eb.use_connect = True
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()


def selected_indices_by_group_prefix(mesh_obj, prefixes: tuple[str, ...], min_weight: float) -> set[int]:
    group_ids = set()
    for group in mesh_obj.vertex_groups:
        if group.name.startswith(prefixes):
            group_ids.add(group.index)
    indices: set[int] = set()
    for vertex in mesh_obj.data.vertices:
        for membership in vertex.groups:
            if membership.group in group_ids and membership.weight >= min_weight:
                indices.add(vertex.index)
                break
    return indices


def submesh_bvh(vertices, polygons, indices: set[int]):
    sub_vertices, sub_polygons = fq.submesh_for_indices(vertices, polygons, indices, "any")
    return fq.make_bvh(sub_vertices, sub_polygons), len(sub_vertices), len(sub_polygons)


def nearest_distance_between_sets(vertices, a_indices: set[int], b_indices: set[int]) -> float | None:
    if not a_indices or not b_indices:
        return None
    best = None
    b_points = [vertices[index] for index in b_indices if index < len(vertices)]
    for index in a_indices:
        if index >= len(vertices):
            continue
        point = vertices[index]
        for other in b_points:
            dist = (point - other).length
            if best is None or dist < best:
                best = float(dist)
    return best


def part_self_overlap(vertices, polygons, paw_indices, body_indices):
    paw_bvh, paw_v, paw_p = submesh_bvh(vertices, polygons, paw_indices)
    body_bvh, body_v, body_p = submesh_bvh(vertices, polygons, body_indices)
    overlap = 0
    if paw_bvh is not None and body_bvh is not None:
        overlap = len(paw_bvh.overlap(body_bvh))
    return {
        "overlap_count": int(overlap),
        "min_vertex_distance": nearest_distance_between_sets(vertices, paw_indices, body_indices),
        "paw_vertices": paw_v,
        "paw_polygons": paw_p,
        "body_vertices": body_v,
        "body_polygons": body_p,
    }


def segment_volume(armature, vertices, side: str, indices, bins=6, trim=0.12) -> dict:
    scapula = f"Scapula_{side}"
    wrist = f"Wrist_{side}"
    shoulder = fq.pose_bone_head_world(armature, scapula)
    elbow = fq.pose_bone_tail_world(armature, scapula)
    palm = fq.pose_bone_tail_world(armature, wrist)
    if shoulder is None or elbow is None or palm is None:
        return {"available": False}
    upper = fq.volume_proxy_for_segment(vertices, indices[side]["upper"], shoulder, elbow, bins, trim)
    lower = fq.volume_proxy_for_segment(vertices, indices[side]["lower"], elbow, palm, bins, trim)
    total = None
    if upper.get("volume_proxy") is not None and lower.get("volume_proxy") is not None:
        total = upper["volume_proxy"] + lower["volume_proxy"]
    return {
        "available": bool(upper.get("available") and lower.get("available")),
        "upper": upper,
        "lower": lower,
        "total_volume_proxy": total,
    }


def evaluate_scale(
    rig,
    mesh,
    action,
    frames: list[float],
    scale_y: float,
    paw_indices_by_side,
    arm_indices,
    head_indices: set[int],
    torso_indices: set[int],
    obstacles,
    use_pose_scale: bool = True,
) -> dict:
    if use_pose_scale:
        apply_forelimb_y_scale(rig, scale_y)
    rows = []
    for frame in frames:
        set_action(rig, action)
        set_frame(frame)
        if use_pose_scale:
            apply_forelimb_y_scale(rig, scale_y)
        vertices, polygons = fq.evaluated_mesh_world(mesh)
        pose_by_side = {side: fq.forelimb_pose_metrics(rig, side) for side in ("L", "R")}
        volume_by_side = {
            side: segment_volume(rig, vertices, side, arm_indices)
            for side in ("L", "R")
        }
        clearance_by_side = {
            side: fq.paw_clearance_for_side(
                vertices,
                polygons,
                paw_indices_by_side[side],
                side,
                obstacles,
                "any",
            )
            for side in ("L", "R")
        }
        self_by_side = {}
        for side in ("L", "R"):
            paw_indices = set(paw_indices_by_side[side])
            self_by_side[side] = {
                "head": part_self_overlap(vertices, polygons, paw_indices, head_indices),
                "torso": part_self_overlap(vertices, polygons, paw_indices, torso_indices),
            }
        rows.append(
            {
                "frame": float(frame),
                "forelimb_pose": pose_by_side,
                "forelimb_volume": volume_by_side,
                "paw_clearance": clearance_by_side,
                "self_overlap": self_by_side,
            }
        )

    right_reach = [
        row["forelimb_pose"]["R"].get("reach_ratio")
        for row in rows
        if row["forelimb_pose"]["R"].get("reach_ratio") is not None
    ]
    right_elbow = [
        row["forelimb_pose"]["R"].get("elbow_angle_deg")
        for row in rows
        if row["forelimb_pose"]["R"].get("elbow_angle_deg") is not None
    ]
    external_overlap = sum(
        int(row["paw_clearance"][side]["overlap_count"])
        for row in rows
        for side in ("L", "R")
    )
    head_overlap = sum(
        int(row["self_overlap"][side]["head"]["overlap_count"])
        for row in rows
        for side in ("L", "R")
    )
    torso_overlap = sum(
        int(row["self_overlap"][side]["torso"]["overlap_count"])
        for row in rows
        for side in ("L", "R")
    )
    min_external_clearance = min(
        float(row["paw_clearance"][side]["min_distance"])
        for row in rows
        for side in ("L", "R")
        if row["paw_clearance"][side]["min_distance"] is not None
    )

    max_reach = max(right_reach) if right_reach else None
    max_elbow = max(right_elbow) if right_elbow else None
    min_reach = min(right_reach) if right_reach else None
    score = 0.0
    if max_reach is not None:
        score += max(0.0, max_reach - 0.94) * 250.0
        score += max(0.0, 0.80 - min_reach) * 80.0
    if max_elbow is not None:
        score += max(0.0, max_elbow - 158.0) * 2.0
    score += external_overlap * 100.0
    score += head_overlap * 8.0
    score += torso_overlap * 4.0
    score += max(0.0, 0.25 - min_external_clearance) * 50.0
    score += (scale_y - 1.0) * 4.0

    return {
        "scale_y": float(scale_y),
        "score": float(score),
        "summary": {
            "right_reach_ratio_max": None if max_reach is None else float(max_reach),
            "right_reach_ratio_min": None if min_reach is None else float(min_reach),
            "right_elbow_angle_max": None if max_elbow is None else float(max_elbow),
            "external_overlap_count": int(external_overlap),
            "self_head_overlap_count": int(head_overlap),
            "self_torso_overlap_count": int(torso_overlap),
            "min_external_clearance": float(min_external_clearance),
        },
        "frames": rows,
    }


def clear_scale_fcurves(action, bone_names: tuple[str, ...]):
    fcurves = action_fcurves(action)
    wanted = {f'pose.bones["{name}"].scale' for name in bone_names}
    for fcurve in list(fcurves):
        if fcurve.data_path in wanted:
            try:
                action.fcurves.remove(fcurve)
            except Exception:
                pass


def bake_constant_scale_keys(rig, action, scale_y: float, frames: list[float]):
    set_action(rig, action)
    clear_scale_fcurves(action, FORELIMB_SCALE_BONES)
    key_frames = sorted({float(bpy.context.scene.frame_start), float(bpy.context.scene.frame_end), *frames})
    for frame in key_frames:
        set_frame(frame)
        apply_forelimb_y_scale(rig, scale_y)
        for bone_name in FORELIMB_SCALE_BONES:
            pb = rig.pose.bones.get(bone_name)
            if pb is not None:
                pb.keyframe_insert(data_path="scale", frame=frame)
    apply_forelimb_y_scale(rig, scale_y)


def prepare_scene(args):
    rig = bpy.data.objects.get(args.armature)
    mesh = bpy.data.objects.get(args.mesh)
    if rig is None or mesh is None:
        raise RuntimeError("Missing armature or mesh")
    action = choose_action(rig, args.action)
    set_action(rig, action)
    paw_indices_by_side, paw_groups = fq.collect_paw_vertex_indices(
        mesh, include_wrist=True, min_weight=args.min_weight
    )
    arm_indices, arm_groups = fq.collect_arm_vertex_indices(mesh, args.min_weight)
    obstacle_names = [name for name in fq.DEFAULT_OBSTACLES if bpy.data.objects.get(name) is not None]
    obstacles, missing_obstacles = fq.obstacle_records(obstacle_names)
    head_indices = selected_indices_by_group_prefix(mesh, HEAD_GROUP_PREFIXES, args.min_weight)
    torso_indices = selected_indices_by_group_prefix(mesh, TORSO_GROUP_PREFIXES, args.min_weight)
    return {
        "rig": rig,
        "mesh": mesh,
        "action": action,
        "paw_indices_by_side": paw_indices_by_side,
        "paw_groups": paw_groups,
        "arm_indices": arm_indices,
        "arm_groups": arm_groups,
        "obstacle_names": obstacle_names,
        "obstacles": obstacles,
        "missing_obstacles": missing_obstacles,
        "head_indices": head_indices,
        "torso_indices": torso_indices,
    }


def main() -> None:
    args = parse_args()
    frames = sorted({float(frame) for frame in args.frame})
    scales = args.scale or default_scales()
    candidates = []

    for scale_y in scales:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        scene = prepare_scene(args)
        if args.mode == "rest-length":
            apply_forelimb_rest_y_length(scene["rig"], float(scale_y))
        candidates.append(
            evaluate_scale(
                scene["rig"],
                scene["mesh"],
                scene["action"],
                frames,
                float(scale_y),
                scene["paw_indices_by_side"],
                scene["arm_indices"],
                scene["head_indices"],
                scene["torso_indices"],
                scene["obstacles"],
                use_pose_scale=(args.mode == "pose-scale"),
            )
        )
        print(f"[sweep] scale_y={scale_y:.3f} score={candidates[-1]['score']:.4f}")

    ranked = sorted(candidates, key=lambda row: (row["score"], row["scale_y"]))
    selected_scale = float(args.selected_scale) if args.selected_scale else float(ranked[0]["scale_y"])
    selected = next(row for row in candidates if abs(row["scale_y"] - selected_scale) < 1.0e-8)

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = prepare_scene(args)
    if args.mode == "rest-length":
        apply_forelimb_rest_y_length(scene["rig"], selected_scale)
    else:
        bake_constant_scale_keys(scene["rig"], scene["action"], selected_scale, frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    report = {
        "input": str(args.blend.resolve()),
        "output": str(args.output.resolve()),
        "mode": args.mode,
        "action": scene["action"].name,
        "armature": scene["rig"].name,
        "mesh": scene["mesh"].name,
        "frames": frames,
        "scaled_bones": list(FORELIMB_SCALE_BONES),
        "scale_candidates": [float(value) for value in scales],
        "selected_scale_y": selected_scale,
        "selected_summary": selected["summary"],
        "candidate_summaries": [
            {
                "scale_y": row["scale_y"],
                "score": row["score"],
                **row["summary"],
            }
            for row in ranked
        ],
        "candidates": candidates,
        "groups": {
            "paw": scene["paw_groups"],
            "arm": scene["arm_groups"],
            "head_index_count": len(scene["head_indices"]),
            "torso_index_count": len(scene["torso_indices"]),
        },
        "obstacles": {
            "used": scene["obstacle_names"],
            "missing": scene["missing_obstacles"],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OLDIK_FORELIMB_Y_SCALE_SWEEP_BEGIN")
    print(json.dumps({key: report[key] for key in ("output", "selected_scale_y", "selected_summary", "candidate_summaries")}, ensure_ascii=False, indent=2))
    print("OLDIK_FORELIMB_Y_SCALE_SWEEP_END")


if __name__ == "__main__":
    main()

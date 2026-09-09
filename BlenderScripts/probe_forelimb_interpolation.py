import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontlimb_quality_check as qc


DEFAULT_OBSTACLES = [
    "Part_MouseCupSide.001",
    "Part_MouseCupSide.002",
    "Part_PrintedMouseCup",
]


FINGER_PREFIXES = [
    "Fingers1",
    "IndexFinger1",
    "IndexFinger2",
    "IndexFinger3",
    "MiddleFinger1",
    "MiddleFinger2",
    "MiddleFinger3",
    "RingFinger1",
    "RingFinger2",
    "RingFinger3",
    "PinkyFinger1",
    "PinkyFinger2",
    "PinkyFinger3",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--obstacle", action="append", default=[])
    parser.add_argument("--volume-threshold", type=float, default=0.72)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--include-fingers", action="store_true")
    parser.add_argument("--arm-volume-bins", type=int, default=6)
    parser.add_argument("--arm-volume-trim", type=float, default=0.12)
    return parser.parse_args(args_after_dash())


def side_bone_names(side, include_fingers):
    names = [f"Scapula_{side}", f"Wrist_{side}"]
    if include_fingers:
        names.extend(f"{prefix}_{side}" for prefix in FINGER_PREFIXES)
    return names


def decompose_pose_bases(armature, bone_names):
    bases = {}
    for name in bone_names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        loc, rot, scale = pose_bone.matrix_basis.decompose()
        bases[name] = (loc.copy(), rot.copy(), scale.copy())
    return bases


def apply_interpolated_bases(armature, before, after, t):
    for name, before_parts in before.items():
        after_parts = after.get(name)
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None or after_parts is None:
            continue
        loc_a, rot_a, scale_a = before_parts
        loc_b, rot_b, scale_b = after_parts
        loc = loc_a.lerp(loc_b, t)
        rot = rot_a.slerp(rot_b, t)
        scale = scale_a.lerp(scale_b, t)
        pose_bone.matrix_basis = Matrix.LocRotScale(loc, rot, scale)
    bpy.context.view_layer.update()


def frame_rows_by_number(quality_action):
    return {
        int(round(float(row["frame"]))): row
        for row in quality_action["frames"]
    }


def ratio_for(row, side):
    volume = row.get("forelimb_volume", {}).get(side, {})
    ratio = volume.get("worst_volume_ratio")
    return None if ratio is None else float(ratio)


def nearest_good_frames(frames, rows_by_frame, frame, side, threshold):
    previous = None
    following = None
    for candidate in reversed([value for value in frames if value < frame]):
        row = rows_by_frame[candidate]
        ratio = ratio_for(row, side)
        flags = set(row.get("flags", []))
        if ratio is not None and ratio >= threshold and f"{side}_paw_cup_overlap" not in flags:
            previous = candidate
            break
    for candidate in [value for value in frames if value > frame]:
        row = rows_by_frame[candidate]
        ratio = ratio_for(row, side)
        flags = set(row.get("flags", []))
        if ratio is not None and ratio >= threshold and f"{side}_paw_cup_overlap" not in flags:
            following = candidate
            break
    return previous, following


def volume_ratios(volume, reference):
    ratios = {}
    worst_ratio = None
    worst_segment = None
    for segment, value in (
        ("upper", volume.get("upper", {}).get("volume_proxy")),
        ("lower", volume.get("lower", {}).get("volume_proxy")),
        ("total", volume.get("total_volume_proxy")),
    ):
        ratio = None
        ref = reference.get(segment)
        if ref and value is not None:
            ratio = float(value / ref)
            if worst_ratio is None or ratio < worst_ratio:
                worst_ratio = ratio
                worst_segment = segment
        ratios[segment] = ratio
    return ratios, worst_ratio, worst_segment


def evaluate_side(armature, mesh_obj, selected_indices, arm_indices, obstacles, side, reference, args):
    vertices, polygons = qc.evaluated_mesh_world(mesh_obj)
    volume = qc.forelimb_volume_metrics(
        vertices,
        armature,
        side,
        arm_indices,
        args.arm_volume_bins,
        args.arm_volume_trim,
    )
    ratios, worst_ratio, worst_segment = volume_ratios(volume, reference)
    clearance = qc.paw_clearance_for_side(
        vertices,
        polygons,
        selected_indices[side],
        side,
        obstacles,
        "any",
    )
    pose = qc.forelimb_pose_metrics(armature, side)
    return {
        "worst_volume_ratio": worst_ratio,
        "worst_volume_segment": worst_segment,
        "upper_volume_ratio": ratios.get("upper"),
        "lower_volume_ratio": ratios.get("lower"),
        "total_volume_ratio": ratios.get("total"),
        "overlap_count": clearance.get("overlap_count"),
        "min_distance": clearance.get("min_distance"),
        "reach_ratio": pose.get("reach_ratio"),
        "elbow_angle_deg": pose.get("elbow_angle_deg"),
    }


def main():
    args = parse_args()
    quality = json.loads(Path(args.quality_json).read_text(encoding="utf-8"))
    quality_action = next(
        action for action in quality["actions"] if action["action"] == args.action
    )
    rows_by_frame = frame_rows_by_number(quality_action)
    frames = sorted(rows_by_frame)

    targets = []
    for frame, row in rows_by_frame.items():
        for side in ("L", "R"):
            ratio = ratio_for(row, side)
            if ratio is not None and ratio < args.volume_threshold:
                targets.append((ratio, frame, side))
    targets.sort(key=lambda item: (item[0], item[1], item[2]))
    targets = targets[: args.top]

    armature = bpy.data.objects[args.armature]
    mesh_obj = bpy.data.objects[args.mesh]
    qc.set_action(armature, bpy.data.actions[args.action])
    selected_indices, _paw_groups = qc.collect_paw_vertex_indices(mesh_obj, False, 0.001)
    arm_indices, _arm_groups = qc.collect_arm_vertex_indices(mesh_obj, 0.001)
    obstacles, missing = qc.obstacle_records(args.obstacle or DEFAULT_OBSTACLES)
    volume_references = quality_action["arm_volume_reference"]
    if missing:
        print(f"[warn] missing obstacles: {missing}")

    samples = []
    improved = 0
    made_overlap = 0
    for _ratio, frame, side in targets:
        before_frame, after_frame = nearest_good_frames(
            frames,
            rows_by_frame,
            frame,
            side,
            args.volume_threshold,
        )
        if before_frame is None or after_frame is None:
            samples.append(
                {
                    "frame": frame,
                    "side": side,
                    "skipped": True,
                    "reason": "missing_neighbor",
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                }
            )
            continue

        bone_names = side_bone_names(side, args.include_fingers)
        qc.set_frame(before_frame)
        before_bases = decompose_pose_bases(armature, bone_names)
        qc.set_frame(after_frame)
        after_bases = decompose_pose_bases(armature, bone_names)

        qc.set_frame(frame)
        baseline = evaluate_side(
            armature,
            mesh_obj,
            selected_indices,
            arm_indices,
            obstacles,
            side,
            volume_references[side],
            args,
        )
        t = (frame - before_frame) / max(after_frame - before_frame, 1.0)
        apply_interpolated_bases(armature, before_bases, after_bases, t)
        interpolated = evaluate_side(
            armature,
            mesh_obj,
            selected_indices,
            arm_indices,
            obstacles,
            side,
            volume_references[side],
            args,
        )
        if (
            interpolated["worst_volume_ratio"] is not None
            and baseline["worst_volume_ratio"] is not None
            and interpolated["worst_volume_ratio"] > baseline["worst_volume_ratio"]
        ):
            improved += 1
        if baseline["overlap_count"] == 0 and interpolated["overlap_count"] > 0:
            made_overlap += 1
        samples.append(
            {
                "frame": frame,
                "side": side,
                "before_frame": before_frame,
                "after_frame": after_frame,
                "t": t,
                "baseline": baseline,
                "interpolated": interpolated,
                "volume_delta": None
                if baseline["worst_volume_ratio"] is None
                or interpolated["worst_volume_ratio"] is None
                else interpolated["worst_volume_ratio"] - baseline["worst_volume_ratio"],
                "overlap_delta": interpolated["overlap_count"] - baseline["overlap_count"],
            }
        )

    result = {
        "file": bpy.data.filepath,
        "action": args.action,
        "include_fingers": args.include_fingers,
        "volume_threshold": args.volume_threshold,
        "target_count": len(targets),
        "sample_count": len(samples),
        "improved_count": improved,
        "made_overlap_count": made_overlap,
        "samples": samples,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_INTERPOLATION_PROBE_BEGIN")
    print(json.dumps({k: result[k] for k in ("target_count", "sample_count", "improved_count", "made_overlap_count")}, indent=2))
    print("MOUSE_GEO_FORGE_INTERPOLATION_PROBE_END")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontlimb_quality_check as qc
import probe_forelimb_interpolation as interp


DEFAULT_OBSTACLES = [
    "Part_MouseCupSide.001",
    "Part_MouseCupSide.002",
    "Part_PrintedMouseCup",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--obstacle", action="append", default=[])
    parser.add_argument("--volume-threshold", type=float, default=0.72)
    parser.add_argument("--min-volume-delta", type=float, default=0.015)
    parser.add_argument("--clearance-threshold", type=float, default=0.25)
    parser.add_argument("--allowed-clearance-drop", type=float, default=0.03)
    parser.add_argument("--top", type=int, default=999)
    parser.add_argument("--include-fingers", action="store_true")
    parser.add_argument("--arm-volume-bins", type=int, default=6)
    parser.add_argument("--arm-volume-trim", type=float, default=0.12)
    return parser.parse_args(args_after_dash())


def targets_from_quality(quality_action, threshold, top):
    rows_by_frame = interp.frame_rows_by_number(quality_action)
    targets = []
    for frame, row in rows_by_frame.items():
        for side in ("L", "R"):
            ratio = interp.ratio_for(row, side)
            if ratio is not None and ratio < threshold:
                targets.append((ratio, frame, side))
    targets.sort(key=lambda item: (item[0], item[1], item[2]))
    return targets[:top], rows_by_frame


def candidate_is_safe(baseline, candidate, args):
    if candidate["worst_volume_ratio"] is None or baseline["worst_volume_ratio"] is None:
        return False, "missing_volume"
    volume_delta = candidate["worst_volume_ratio"] - baseline["worst_volume_ratio"]
    if volume_delta < args.min_volume_delta and candidate["worst_volume_ratio"] < args.volume_threshold:
        return False, "small_volume_gain"
    if candidate["overlap_count"] > baseline["overlap_count"]:
        return False, "overlap_worse"
    if candidate["overlap_count"] > 0:
        return False, "overlap_remaining"

    base_distance = baseline["min_distance"]
    candidate_distance = candidate["min_distance"]
    if base_distance is not None and candidate_distance is not None:
        if base_distance >= args.clearance_threshold and candidate_distance < args.clearance_threshold:
            return False, "clearance_became_warning"
        if candidate_distance + args.allowed_clearance_drop < base_distance:
            return False, "clearance_drop"
    return True, "accepted"


def keyframe_bones(armature, bone_names, frame):
    for name in bone_names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        pose_bone.keyframe_insert(data_path="location", frame=frame)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pose_bone.keyframe_insert(data_path="scale", frame=frame)


def main():
    args = parse_args()
    quality = json.loads(Path(args.quality_json).read_text(encoding="utf-8"))
    quality_action = next(
        action for action in quality["actions"] if action["action"] == args.action
    )
    targets, rows_by_frame = targets_from_quality(
        quality_action,
        args.volume_threshold,
        args.top,
    )
    frames = sorted(rows_by_frame)

    armature = bpy.data.objects[args.armature]
    mesh_obj = bpy.data.objects[args.mesh]
    action = bpy.data.actions[args.action]
    qc.set_action(armature, action)
    selected_indices, _paw_groups = qc.collect_paw_vertex_indices(mesh_obj, False, 0.001)
    arm_indices, _arm_groups = qc.collect_arm_vertex_indices(mesh_obj, 0.001)
    obstacles, missing = qc.obstacle_records(args.obstacle or DEFAULT_OBSTACLES)
    volume_references = quality_action["arm_volume_reference"]

    report_rows = []
    accepted = 0
    rejected = 0
    skipped = 0
    for _ratio, frame, side in targets:
        before_frame, after_frame = interp.nearest_good_frames(
            frames,
            rows_by_frame,
            frame,
            side,
            args.volume_threshold,
        )
        if before_frame is None or after_frame is None:
            skipped += 1
            report_rows.append(
                {
                    "frame": frame,
                    "side": side,
                    "decision": "skipped",
                    "reason": "missing_neighbor",
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                }
            )
            continue

        bone_names = interp.side_bone_names(side, args.include_fingers)
        qc.set_frame(before_frame)
        before_bases = interp.decompose_pose_bases(armature, bone_names)
        qc.set_frame(after_frame)
        after_bases = interp.decompose_pose_bases(armature, bone_names)

        qc.set_frame(frame)
        original_bases = interp.decompose_pose_bases(armature, bone_names)
        baseline = interp.evaluate_side(
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
        interp.apply_interpolated_bases(armature, before_bases, after_bases, t)
        candidate = interp.evaluate_side(
            armature,
            mesh_obj,
            selected_indices,
            arm_indices,
            obstacles,
            side,
            volume_references[side],
            args,
        )
        safe, reason = candidate_is_safe(baseline, candidate, args)
        if safe:
            accepted += 1
            keyframe_bones(armature, bone_names, frame)
            decision = "accepted"
        else:
            rejected += 1
            interp.apply_interpolated_bases(armature, original_bases, original_bases, 0.0)
            decision = "rejected"

        report_rows.append(
            {
                "frame": frame,
                "side": side,
                "decision": decision,
                "reason": reason,
                "before_frame": before_frame,
                "after_frame": after_frame,
                "t": t,
                "baseline": baseline,
                "candidate": candidate,
                "volume_delta": None
                if baseline["worst_volume_ratio"] is None
                or candidate["worst_volume_ratio"] is None
                else candidate["worst_volume_ratio"] - baseline["worst_volume_ratio"],
                "overlap_delta": candidate["overlap_count"] - baseline["overlap_count"],
            }
        )

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.resolve()))
    report = {
        "file": bpy.data.filepath,
        "output": str(output_path.resolve()),
        "action": args.action,
        "settings": {
            "volume_threshold": args.volume_threshold,
            "min_volume_delta": args.min_volume_delta,
            "clearance_threshold": args.clearance_threshold,
            "allowed_clearance_drop": args.allowed_clearance_drop,
            "include_fingers": args.include_fingers,
        },
        "target_count": len(targets),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "skipped_count": skipped,
        "samples": report_rows,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_SELECTIVE_INTERPOLATION_BEGIN")
    print(
        json.dumps(
            {
                "target_count": len(targets),
                "accepted_count": accepted,
                "rejected_count": rejected,
                "skipped_count": skipped,
                "output": str(output_path.resolve()),
                "report": str(report_path.resolve()),
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_SELECTIVE_INTERPOLATION_END")


if __name__ == "__main__":
    main()

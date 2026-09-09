"""Audit hand-mouth semantics over a frame range.

This focuses on the failure mode where the migrated IK drives the palm or wrist
into the mouth while the source animation only brings the fingertips close.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_hand_mouth_and_finger_alignment as audit  # noqa: E402
import frontlimb_quality_check as qc  # noqa: E402


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--target-blend", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--target-armature", default="DeformationSystem")
    parser.add_argument("--source-action", default=None)
    parser.add_argument("--target-action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--side", action="append", choices=("L", "R"), default=[])
    parser.add_argument("--palm-min-source-ratio", type=float, default=0.72)
    parser.add_argument("--fingertip-max-source-ratio", type=float, default=1.85)
    return parser.parse_args(args_after_dash())


def action_frames(action) -> list[float]:
    frames = set()
    fcurves = []
    if getattr(action, "fcurves", None) is not None:
        fcurves.extend(action.fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    for fcurve in fcurves:
        for key in fcurve.keyframe_points:
            frames.add(round(float(key.co.x), 6))
    if frames:
        return sorted(frames)
    start, end = action.frame_range
    return [float(frame) for frame in range(math.ceil(start), math.floor(end) + 1)]


def choose_frames(args: argparse.Namespace, source_action) -> list[float]:
    if args.frame:
        return sorted(set(float(frame) for frame in args.frame))
    frames = action_frames(source_action)
    if args.frame_start is not None:
        frames = [frame for frame in frames if frame >= args.frame_start]
    if args.frame_end is not None:
        frames = [frame for frame in frames if frame <= args.frame_end]
    return frames


def set_action_by_name(armature, action_name: str | None):
    return audit.set_action(armature, action_name)


def frame_semantics(armature, side: str) -> dict:
    mouth, mouth_source = audit.mouth_proxy(armature)
    head, head_source = audit.head_proxy(armature)
    points = audit.hand_points(armature, side)
    return {
        "mouth_proxy": mouth_source,
        "head_proxy": head_source,
        "distance_to_mouth": {
            name: audit.distance(point, mouth) for name, point in points.items()
        },
        "distance_to_head": {
            name: audit.distance(point, head) for name, point in points.items()
        },
        "arm_chain": audit.arm_chain_metrics(points),
    }


def ratio(value, ref):
    if value is None or ref in (None, 0):
        return None
    return float(value / ref)


def side_row(
    source_sem: dict,
    target_sem: dict,
    palm_min_source_ratio: float,
    fingertip_max_source_ratio: float,
) -> dict:
    source_mouth = source_sem["distance_to_mouth"]
    target_mouth = target_sem["distance_to_mouth"]
    palm_ratio = ratio(target_mouth.get("palm"), source_mouth.get("palm"))
    wrist_ratio = ratio(target_mouth.get("wrist_head"), source_mouth.get("wrist_head"))
    fingertip_ratio = ratio(
        target_mouth.get("fingertip_center"),
        source_mouth.get("fingertip_center"),
    )
    fingers_tip_ratio = ratio(
        target_mouth.get("fingers_tip"),
        source_mouth.get("fingers_tip"),
    )
    flags = []
    if palm_ratio is not None and palm_ratio < palm_min_source_ratio:
        flags.append("palm_too_close_to_mouth")
    if wrist_ratio is not None and wrist_ratio < palm_min_source_ratio:
        flags.append("wrist_too_close_to_mouth")
    if fingertip_ratio is not None and fingertip_ratio > fingertip_max_source_ratio:
        flags.append("fingertip_too_far_from_mouth")
    if fingers_tip_ratio is not None and fingers_tip_ratio > fingertip_max_source_ratio:
        flags.append("fingers_tip_too_far_from_mouth")
    return {
        "source": source_sem,
        "target": target_sem,
        "ratios": {
            "palm_mouth": palm_ratio,
            "wrist_mouth": wrist_ratio,
            "fingers_tip_mouth": fingers_tip_ratio,
            "fingertip_center_mouth": fingertip_ratio,
        },
        "flags": flags,
    }


def collect_source(args: argparse.Namespace):
    bpy.ops.wm.open_mainfile(filepath=str(args.source_blend.resolve()))
    armature = bpy.data.objects.get(args.source_armature)
    if armature is None:
        raise RuntimeError(f"Missing source armature {args.source_armature!r}")
    action = set_action_by_name(armature, args.source_action)
    frames = choose_frames(args, action)
    sides = args.side or ["R"]
    out = {}
    for frame in frames:
        qc.set_frame(frame)
        frame_key = str(int(round(frame)))
        out[frame_key] = {side: frame_semantics(armature, side) for side in sides}
    return action.name, frames, sides, out


def collect_target(
    args: argparse.Namespace,
    target_blend: Path,
    frames: list[float],
    sides: list[str],
    source_rows: dict,
) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(target_blend.resolve()))
    armature = bpy.data.objects.get(args.target_armature)
    if armature is None:
        raise RuntimeError(f"Missing target armature {args.target_armature!r}")
    action = set_action_by_name(armature, args.target_action)
    rows = []
    flag_counts = {}
    for frame in frames:
        qc.set_frame(frame)
        frame_key = str(int(round(frame)))
        row = {"frame": float(frame), "sides": {}}
        for side in sides:
            target_sem = frame_semantics(armature, side)
            source_sem = source_rows[frame_key][side]
            side_metrics = side_row(
                source_sem,
                target_sem,
                args.palm_min_source_ratio,
                args.fingertip_max_source_ratio,
            )
            for flag in side_metrics["flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            row["sides"][side] = side_metrics
        rows.append(row)
    return {
        "blend": str(target_blend.resolve()),
        "action": action.name,
        "frame_count": len(frames),
        "flag_counts": flag_counts,
        "frames": rows,
    }


def main() -> None:
    args = parse_args()
    source_action, frames, sides, source_rows = collect_source(args)
    targets = [
        collect_target(args, target_blend, frames, sides, source_rows)
        for target_blend in args.target_blend
    ]
    report = {
        "source": str(args.source_blend.resolve()),
        "source_action": source_action,
        "frames": [float(frame) for frame in frames],
        "sides": sides,
        "thresholds": {
            "palm_min_source_ratio": float(args.palm_min_source_ratio),
            "fingertip_max_source_ratio": float(args.fingertip_max_source_ratio),
        },
        "targets": targets,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HAND_SEMANTIC_SERIES_AUDIT_BEGIN")
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "frame_count": len(frames),
                "targets": len(targets),
                "sides": sides,
            },
            indent=2,
        )
    )
    print("HAND_SEMANTIC_SERIES_AUDIT_END")


if __name__ == "__main__":
    main()
